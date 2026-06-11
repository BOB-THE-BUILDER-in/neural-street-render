#!/usr/bin/env python3
"""
extract_street.py  --  Street-scene condition extraction pipeline (v1)

ONE script to rebuild the environment and turn a city-walking video into training
data for the realtime neural-render model.

Pipeline:
  webm video -> ffmpeg decode (HDR->sRGB 8-bit) -> 12fps -> 384px center-crop
            -> group into 16-frame windows
  per window:
     - DA3 METRIC depth   (fixed scale -> clean engine-match later, near=bright)
     - DSINE normals      (per-frame; light + fast. upgrade to NormalCrafter later)
     - SD-1.5 VAE latents (training latents, x0.18215)
  -> save window .pt + push to HuggingFace dataset (resumable; instance-death safe)

CONVENTIONS recorded to conventions.json for deterministic engine-matching later.

Run:
  tmux new -s extract
  python extract_street.py --setup --video /workspace/videoplayback.webm
  # ctrl-b d to detach;  tmux attach -t extract  to return

Resumable: re-running skips windows already pushed to HF.
"""

import os, sys, json, argparse, subprocess, time
from pathlib import Path

# ----------------------------------------------------------------------------
# 0. Dependency setup (run once; safe to re-run)
# ----------------------------------------------------------------------------
SETUP = r"""
set -e
apt-get update && apt-get install -y ffmpeg git tmux
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate safetensors
pip install opencv-python pillow numpy imageio imageio-ffmpeg geffnet
pip install huggingface_hub
# Depth Anything 3 (metric)
[ -d /workspace/Depth-Anything-3 ] || git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /workspace/Depth-Anything-3
pip install -e /workspace/Depth-Anything-3 || true
"""

def run_setup():
    print(">>> Installing dependencies (first run only)...", flush=True)
    subprocess.run(["bash", "-c", SETUP], check=True)
    print(">>> Setup done.", flush=True)

# ----------------------------------------------------------------------------
# Config / conventions  (recorded for engine-matching at inference)
# ----------------------------------------------------------------------------
CONVENTIONS = {
    "resolution": 384,
    "fps": 12,
    "window_frames": 16,
    "depth": {
        "model": "DepthAnything3 metric",
        "type": "metric_meters",
        "normalize": "d_norm = clip(depth_m / DEPTH_MAX_M, 0, 1); stored = 1 - d_norm  (near=bright)",
        "DEPTH_MAX_M": 50.0,
    },
    "normals": {
        "model": "DSINE (hugoycj/DSINE-hub)",
        "space": "view-space (camera)",
        "encode": "rgb = normal * 0.5 + 0.5",
        "flip_y": False,   # to be calibrated against engine at inference
        "flip_z": False,
        "note": "per-frame (may flicker slightly); upgrade to NormalCrafter later for temporal consistency",
    },
    "vae": {
        "model": "stabilityai/sd-vae-ft-mse (SD-1.5)",
        "scale": 0.18215,
        "downscale": 8,
        "note": "training latents; swap to TAESD (madebyollin/taesd) for realtime decode only",
    },
}
DEPTH_MAX_M = CONVENTIONS["depth"]["DEPTH_MAX_M"]

# ----------------------------------------------------------------------------
# Frame decode + subsample (ffmpeg)
# ----------------------------------------------------------------------------
def decode_frames(video, out_dir, fps, res):
    os.makedirs(out_dir, exist_ok=True)
    if any(Path(out_dir).glob("f_*.png")):
        print(f">>> Frames already decoded in {out_dir}, skipping ffmpeg.", flush=True)
        return sorted(Path(out_dir).glob("f_*.png"))
    # HDR (bt2020/HLG 10-bit) -> tonemap to sRGB 8-bit, then fps/scale/crop.
    vf = (
        "zscale=t=linear:npl=100,format=gbrpf32le,"
        "zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
        "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        f"fps={fps},scale='if(gt(iw,ih),-2,{res})':'if(gt(iw,ih),{res},-2)',"
        f"crop={res}:{res}"
    )
    print(f">>> Decoding {video} -> {fps}fps {res}px frames (HDR->sRGB 8-bit)...", flush=True)
    r = subprocess.run(["ffmpeg", "-y", "-i", video, "-vf", vf, "-pix_fmt", "rgb24",
                        os.path.join(out_dir, "f_%06d.png")])
    if r.returncode != 0:
        print("!!! HDR tonemap chain failed (zscale missing?). Falling back to simple 8-bit decode.", flush=True)
        vf2 = (f"fps={fps},scale='if(gt(iw,ih),-2,{res})':'if(gt(iw,ih),{res},-2)',"
               f"crop={res}:{res},format=yuv420p")
        subprocess.run(["ffmpeg", "-y", "-i", video, "-vf", vf2, "-pix_fmt", "rgb24",
                        os.path.join(out_dir, "f_%06d.png")], check=True)
    frames = sorted(Path(out_dir).glob("f_*.png"))
    print(f">>> Decoded {len(frames)} frames.", flush=True)
    return frames

# ----------------------------------------------------------------------------
# Model loaders
# ----------------------------------------------------------------------------
def load_models(device):
    import torch
    from diffusers import AutoencoderKL
    print(">>> Loading SD-1.5 VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    vae.requires_grad_(False)

    print(">>> Loading Depth Anything 3 (metric)...", flush=True)
    sys.path.insert(0, "/workspace/Depth-Anything-3")
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large").to(device=device)

    print(">>> Loading DSINE (normals)...", flush=True)
    nc = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)  # manages its own device

    return vae, da3, nc

# ----------------------------------------------------------------------------
# Per-window extraction
# ----------------------------------------------------------------------------
def extract_window(frames, vae, da3, nc, device):
    import torch, numpy as np
    from PIL import Image
    paths = [str(f) for f in frames]
    imgs = [Image.open(f).convert("RGB") for f in frames]              # 16 x (384,384,3)
    arr = np.stack([np.asarray(im) for im in imgs]).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)          # (16,3,384,384), [0,1]

    # --- SD-1.5 latents ---
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = vae.encode(x * 2 - 1).latent_dist.sample() * CONVENTIONS["vae"]["scale"]  # (16,4,48,48)

    # --- DA3 metric depth (meters), normalized near=bright ---
    with torch.no_grad():
        pred = da3.inference(paths)                                   # prediction.depth: [16,H,W] meters
    depth = torch.as_tensor(np.asarray(pred.depth), device=device).float()  # (16,384,384)
    lo = torch.quantile(depth.flatten(), 0.02)
    hi = torch.quantile(depth.flatten(), 0.98)
    d = ((depth - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    depth_buf = (1.0 - d).unsqueeze(1)                                # near=bright, (16,1,384,384)
    depth_buf = torch.nn.functional.interpolate(depth_buf, size=(48, 48), mode="bilinear", align_corners=False)

    # --- DSINE normals (per-frame), view-space, [-1,1] ---
    nrm = []
    with torch.inference_mode():
        for im in imgs:
            n = nc.infer_pil(im)[0]                                   # (3,384,384) in [-1,1]
            nrm.append(n)
    normals = torch.stack(nrm).to(device).float()                    # (16,3,384,384)
    normals = torch.nn.functional.interpolate(normals, size=(48, 48), mode="bilinear", align_corners=False)

    return {"latent": z.cpu(), "depth": depth_buf.cpu(), "normals": normals.cpu()}

# ----------------------------------------------------------------------------
# HuggingFace push (resumable)
# ----------------------------------------------------------------------------
def hf_existing(repo):
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        return {f for f in files if f.startswith("win_") and f.endswith(".pt")}
    except Exception:
        return set()

def hf_push(repo, local_path, name):
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=local_path, path_in_repo=name,
                repo_id=repo, repo_type="dataset")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(a):
    import torch
    if a.setup:
        run_setup()

    device = "cuda"
    frames_dir = "/workspace/frames"
    work = "/workspace/windows"; os.makedirs(work, exist_ok=True)

    conv_path = os.path.join(work, "conventions.json")
    json.dump(CONVENTIONS, open(conv_path, "w"), indent=2)
    print(f">>> Conventions written to {conv_path} (engine must match these).", flush=True)

    frames = decode_frames(a.video, frames_dir, CONVENTIONS["fps"], CONVENTIONS["resolution"])
    W = CONVENTIONS["window_frames"]
    n_windows = len(frames) // W
    print(f">>> {len(frames)} frames -> {n_windows} windows of {W}.", flush=True)

    if not a.no_hf:
        try:
            hf_push(a.hf_repo, conv_path, "conventions.json")
        except Exception as e:
            print(f"!!! conventions push failed (create the dataset repo first): {e}", flush=True)
        done = hf_existing(a.hf_repo)
        print(f">>> {len(done)} windows already on HF, will skip those.", flush=True)
    else:
        done = set()

    vae, da3, nc = load_models(device)

    t0 = time.time()
    for i in range(n_windows):
        name = f"win_{i:05d}.pt"
        if name in done:
            continue
        wframes = frames[i * W:(i + 1) * W]
        out = extract_window(wframes, vae, da3, nc, device)
        local = os.path.join(work, name)
        torch.save(out, local)
        el = time.time() - t0
        print(f"[{i+1}/{n_windows}] {name}  ({el/(i+1):.1f}s/window, ETA {el/(i+1)*(n_windows-i-1)/60:.0f}m)", flush=True)

    print(">>> Extraction complete.", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to the downloaded webm")
    p.add_argument("--hf_repo", default="bobthebuilderinternational/delhi-street-conditions")
    p.add_argument("--setup", action="store_true", help="run dependency install first")
    p.add_argument("--no_hf", action="store_true", help="local only, no HF push")
    main(p.parse_args())
