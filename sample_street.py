#!/usr/bin/env python3
"""
sample_street.py -- visual test for the street renderer.

Takes a window, builds the texture-noise canvas at a chosen strength, steers with
that window's clean depth+normals + past frame, runs few-step generation, decodes,
and saves a side-by-side [ real | generated ] PNG per frame so you can SEE if the
model is learning to render streets. Loss can't tell you this; the image can.

Run:
  python sample_street.py --ckpt /workspace/checkpoints_street/street_step_005000.pt --win 50 --strength 0.6
"""
import os, argparse, glob
import torch, numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from train_street import StreetUNet, SD_SCALE   # reuse the model def

DEVICE = "cuda"

@torch.no_grad()
def gen(model, x0, past, is_first, depth, normals, K, strength):
    """Few-step Euler from the noised canvas (sitting at flow-time 1-strength) to clean."""
    x = x0
    t_start = 1 - strength                    # canvas noise level = strength
    dt = (1 - t_start) / K
    for i in range(K):
        t = t_start + i * dt
        tt = torch.full((x.shape[0],), t * 999, device=DEVICE).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, past, is_first, depth, normals, tt)
        x = x + dt * v
    return x

def main(a):
    # model
    model = StreetUNet().to(DEVICE).eval()
    ck = torch.load(a.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["ema"] if a.ema else ck["model"])
    print(f"loaded {a.ckpt} (step {ck.get('step')}, {'ema' if a.ema else 'raw'})")

    # decoder (SD-1.5 VAE; swap to TAESD later for speed)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()

    def dec(z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = vae.decode(z / SD_SCALE).sample
        return ((img.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[0].transpose(1, 2, 0)

    # data
    f = sorted(glob.glob(os.path.join(a.data, "win_*.pt")))[a.win]
    d = torch.load(f, map_location=DEVICE)
    lat, depth, normals = d["latent"].float(), d["depth"].float(), d["normals"].float()
    T = lat.shape[0]
    os.makedirs(a.out, exist_ok=True)

    for ti in range(T):
        clean = lat[ti:ti+1]
        past = torch.zeros_like(clean) if ti == 0 else lat[ti-1:ti]
        is_first = torch.ones(1, device=DEVICE) if ti == 0 else torch.zeros(1, device=DEVICE)
        dd, nn_ = depth[ti:ti+1], normals[ti:ti+1]

        noise = torch.randn_like(clean)
        x0 = (1 - a.strength) * clean + a.strength * noise     # noised texture canvas
        out = gen(model, x0, past, is_first, dd, nn_, a.K, a.strength)

        grid = Image.new("RGB", (384*2, 384))
        grid.paste(Image.fromarray(dec(clean)), (0, 0))        # real
        grid.paste(Image.fromarray(dec(out)),   (384, 0))      # generated
        grid.save(f"{a.out}/cmp_{ti:02d}.png")
    print(f"saved {T} frames to {a.out}/  [ real | generated ] at strength={a.strength}, K={a.K}")
    print(f"scp them to your Mac. If the RIGHT side looks like a plausible street matching the left's structure, it's learning.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="/workspace/windows")
    p.add_argument("--out", default="/workspace/samples")
    p.add_argument("--win", type=int, default=50)
    p.add_argument("--strength", type=float, default=0.6, help="texture noise (0=keep render, 1=full hallucination)")
    p.add_argument("--K", type=int, default=8, help="denoise steps for the test")
    p.add_argument("--ema", action="store_true", help="use EMA weights (smoother)")
    main(p.parse_args())
