#!/usr/bin/env python3
"""
test_image.py -- run the trained street renderer on ANY image (OOD test).

Extracts depth (DA3 metric, per-window-style normalize) + normals (DSINE) from the
input image using the SAME conventions as training, builds the noised-texture canvas,
runs the model, and saves [ input | generated ] so you can see if it generalizes to
frames it never trained on.

Run:
  python test_image.py --img /workspace/test.jpg --ckpt /workspace/checkpoints_street/street_step_040000.pt --strength 0.3
"""
import os, sys, argparse
import torch, numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from train_street import StreetUNet, SD_SCALE

DEVICE = "cuda"
DEPTH_MAX_M = 50.0  # only used if metric path needs a fallback

def load_extractors():
    sys.path.insert(0, "/workspace/Depth-Anything-3")
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large").to(device=DEVICE)
    dsine = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    return da3, dsine

@torch.no_grad()
def gen(model, x0, past, is_first, depth, normals, K, strength):
    x = x0
    t_start = 1 - strength
    dt = (1 - t_start) / K
    for i in range(K):
        t = t_start + i * dt
        tt = torch.full((x.shape[0],), t * 999, device=DEVICE).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, past, is_first, depth, normals, tt)
        x = x + dt * v
    return x

def main(a):
    # --- load image, center-crop to 384 like training ---
    im = Image.open(a.img).convert("RGB")
    w, h = im.size; s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((384, 384), Image.LANCZOS)
    im.save("/workspace/test_input384.png")

    da3, dsine = load_extractors()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()

    arr = np.asarray(im).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,3,384,384)

    # latent
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = vae.encode(x * 2 - 1).latent_dist.sample() * SD_SCALE   # (1,4,48,48)

    # depth (DA3) -> per-window(=per-image here) percentile stretch, near=bright  [MATCHES TRAINING]
    with torch.no_grad():
        pred = da3.inference(["/workspace/test_input384.png"])
    depth = torch.as_tensor(np.asarray(pred.depth), device=DEVICE).float()  # (1,384,384)
    lo = torch.quantile(depth.flatten(), 0.02); hi = torch.quantile(depth.flatten(), 0.98)
    d = ((depth - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    depth_buf = (1 - d).unsqueeze(1)
    depth_buf = torch.nn.functional.interpolate(depth_buf, size=(48, 48), mode="bilinear", align_corners=False)

    # normals (DSINE)
    with torch.inference_mode():
        n = dsine.infer_pil(im)[0]                                    # (3,384,384) [-1,1]
    normals = n.unsqueeze(0).to(DEVICE).float()
    normals = torch.nn.functional.interpolate(normals, size=(48, 48), mode="bilinear", align_corners=False)

    # model
    model = StreetUNet().to(DEVICE).eval()
    ck = torch.load(a.ckpt, map_location=DEVICE)
    model.load_state_dict(ck["ema"])
    print(f"loaded {a.ckpt} (step {ck.get('step')})")

    # noised-texture canvas (single frame -> no past, treat as is_first)
    noise = torch.randn_like(clean)
    x0 = (1 - a.strength) * clean + a.strength * noise
    past = torch.zeros_like(clean); is_first = torch.ones(1, device=DEVICE)
    out = gen(model, x0, past, is_first, depth_buf, normals, a.K, a.strength)

    def dec(z):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = vae.decode(z / SD_SCALE).sample
        return ((img.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()[0].transpose(1, 2, 0)

    grid = Image.new("RGB", (384*2, 384))
    grid.paste(im, (0, 0))
    grid.paste(Image.fromarray(dec(out)), (384, 0))
    grid.save(a.out)
    print(f"saved {a.out}  [ input | generated ]  strength={a.strength}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True)
    p.add_argument("--ckpt", default="/workspace/checkpoints_street/street_step_040000.pt")
    p.add_argument("--out", default="/workspace/ood_test.png")
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--K", type=int, default=8)
    main(p.parse_args())
