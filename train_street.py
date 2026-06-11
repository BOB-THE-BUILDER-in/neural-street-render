#!/usr/bin/env python3
"""
train_street.py  --  Street neural-renderer training (v1)

Objective (the decided architecture):
  canvas  = noised real-frame latent   (Runway-style texture-noise; strength random per sample)
  steer   = clean depth + clean normals (re-injected every step -> geometry lock)
  context = past-frame latent           (temporal coherence)
  target  = clean frame latent
  loss    = flow-matching velocity (canvas->clean), conditioned on steer+context

No identity, no segmentation (streets don't need them; normals+texture carry placement).
Trains fresh from random init on the Delhi street windows.

Data: /workspace/windows/win_*.pt   each = {latent:(16,4,48,48), depth:(16,1,48,48), normals:(16,3,48,48)}
  (download from HF if not local:
     hf download bobthebuilderinternational/delhi-street-conditions --repo-type dataset --local-dir /workspace/windows)

Run:
  tmux new -s train
  python train_street.py
Checkpoints -> /workspace/checkpoints_street/  (push to HF periodically yourself).
"""
import os, glob, math, time, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
DEVICE = "cuda"
SD_SCALE = 0.18215

# ----------------------------------------------------------------------------
# Dataset: serves (clean latents, depth, normals) per 16-frame window
# ----------------------------------------------------------------------------
class StreetWindows(Dataset):
    def __init__(self, root="/workspace/windows"):
        self.files = sorted(glob.glob(os.path.join(root, "win_*.pt")))
        assert self.files, f"no windows in {root}"
        print(f"dataset: {len(self.files)} windows")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        return d["latent"].float(), d["depth"].float(), d["normals"].float()  # (16,4,48,48),(16,1,48,48),(16,3,48,48)

# ----------------------------------------------------------------------------
# Model: temporal conditional U-Net (Phase-3 lineage), street channels
#   in  = noisy(4) + past(4) + is_first(1) + depth(1) + normals(3) = 13 channels
#   out = velocity (4)
# ----------------------------------------------------------------------------
from diffusers import UNet2DConditionModel

class StreetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=48,
            in_channels=13,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(192, 384, 512, 512),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=256,
        )
        # no identity/text: feed a single learned null context token
        self.null_ctx = nn.Parameter(torch.randn(1, 1, 256) * 0.02)

    def forward(self, noisy, past, is_first, depth, normals, t):
        B = noisy.shape[0]
        isf = is_first.view(B, 1, 1, 1).expand(B, 1, 48, 48)
        x = torch.cat([noisy, past, isf, depth, normals], dim=1)   # (B,13,48,48)
        ctx = self.null_ctx.expand(B, -1, -1)
        return self.unet(x, t, encoder_hidden_states=ctx).sample

# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

def save_ckpt(model, ema, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": ema.shadow, "step": step},
               os.path.join(out_dir, f"street_step_{step:06d}.pt"))
    print(f"  saved checkpoint at step {step}", flush=True)

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(a):
    ds = StreetWindows(a.data)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=2,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    model = StreetUNet().to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    ema = EMA(model)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"training: {a.iters} steps, batch {a.batch} windows x 16 frames, fresh init", flush=True)

    step, done, t0 = 0, False, time.time()
    while not done:
        for lat, depth, normals in dl:
            lat = lat.to(DEVICE); depth = depth.to(DEVICE); normals = normals.to(DEVICE)
            B, T = lat.shape[:2]
            # pick a random target frame t in [1..T-1]; its past is frame t-1.
            # also train frame 0 (is_first) sometimes so the model can cold-start.
            ti = random.randint(0, T - 1)
            clean = lat[:, ti]                                   # (B,4,48,48) target
            if ti == 0:
                past = torch.zeros_like(clean); is_first = torch.ones(B, device=DEVICE)
            else:
                past = lat[:, ti - 1]; is_first = torch.zeros(B, device=DEVICE)
            d = depth[:, ti]; n = normals[:, ti]

            # --- texture-noise canvas: start from the clean frame + random-strength noise ---
            noise = torch.randn_like(clean)
            s = torch.rand(B, 1, 1, 1, device=DEVICE)            # strength in [0,1) per sample
            x0 = (1 - s) * clean + s * noise                     # noised texture (canvas at t=s)
            # flow-matching from x0 (at time s) toward clean (at time 1):
            #   target velocity along the straight path = clean - noise
            #   but our canvas already sits at fraction s; train v to map x_s -> clean
            # Use rectified-flow convention: z_t=(1-t)noise+t*clean, v=clean-noise, t=1-s
            t_frac = (1 - s)                                      # canvas noise level = s  ->  flow time = 1-s
            target_v = clean - noise
            tt = (t_frac.view(B) * 999).long()

            with amp():
                pred_v = model(x0, past, is_first, d, n, tt)
            loss = F.mse_loss(pred_v.float(), target_v.float())

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)

            step += 1
            if step <= 20 or step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{a.iters}  loss {loss.item():.4f}  ({el/step:.2f}s/it)", flush=True)
            if step % a.save_every == 0 or step == a.iters:
                save_ckpt(model, ema, step, a.out_dir)
            if step >= a.iters:
                done = True; break
    print("training done", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/windows")
    p.add_argument("--out_dir", default="/workspace/checkpoints_street")
    p.add_argument("--iters", type=int, default=40000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=5000)
    train(p.parse_args())
