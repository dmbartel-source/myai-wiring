# RunPod GPU backend for MYAI Studio

Studio's self-hosted generation path: ComfyUI running on a rented RunPod
H100, driven by `tools/gpu_client.py` + `tools/gpu_workflows.py`. Open
models only — Qwen-Image-Edit-2509 for photo edits, Wan 2.2 14B for
image/text-to-video, Wan VACE 14B for model replacement in video.
Media flows only between the Studio server and this pod.

## Create the pod (RunPod console)

| Setting | Value |
|---|---|
| GPU | **H100 80GB SXM** (1x) |
| Cloud | **Secure Cloud** (datacenter tier — the reliable one) |
| Pricing | **On-demand. Never spot/interruptible** — spot pods can be reclaimed mid-render and the job dies with them |
| Network volume | **150 GB**, mounted at `/workspace` (see size note below) |
| Datacenter | Any with H100 SXM stock; pick one and stay in it (network volumes are datacenter-bound) |
| Template/image | A PyTorch image (e.g. `runpod/pytorch`) is fine — setup installs everything else |
| Exposed ports | HTTP: **8188** (ComfyUI). TCP/SSH optional |

**Volume size note:** the plan started at 75 GB, but the model set in
`models.txt` totals **~128 GB** (four Wan 2.2 FP8 files ≈ 57 GB, VACE
14B fp16 ≈ 35 GB, Qwen-Image-Edit-2509 + its VL encoder ≈ 29 GB, plus
shared encoders/VAEs). 75 GB physically cannot hold it, so the volume
is 150 GB. At ~**$0.07/GB/month** that's ≈ **$10.50/month** — and it's
the only fixed cost: stop the pod when idle and you pay nothing but the
volume.

**Pod cost while running:** H100 80GB SXM Secure Cloud on-demand runs
about **$2.89–$3.49/hr** (Oct 2026). A 15-second Wan clip (~10–20 min of
rendering) therefore costs roughly $0.50–$1.15 of pod time, and you only
pay while the pod is up.

## Bootstrap the pod (once)

SSH/web-terminal into the fresh pod and run:

```sh
# copy this runpod/ directory to the pod first (git clone the repo, or scp)
export HF_TOKEN=<your-huggingface-token>   # optional, faster downloads
bash runpod/setup_pod.sh
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188
```

`setup_pod.sh` is idempotent — re-run it after any interruption and it
only fetches what's missing. Everything (ComfyUI, custom nodes, models)
installs onto the network volume, so after the first setup a new pod in
the same datacenter is rendering again in minutes: mount the same
volume, start ComfyUI, done. No re-download.

Security note: ComfyUI has no built-in authentication and anyone with
the URL can use it. The RunPod proxy URL contains the unguessable pod
ID — treat it like a password, don't share it, and stop the pod when
you're not using it (which you want anyway, since it's billed hourly).

## Point Studio at the pod

On the **Studio server**, create `~/.nanobot/gpu.env`:

```ini
# Pod proxy URL: https://<pod-id>-8188.proxy.runpod.net
# (RunPod console: pod -> Connect -> "HTTP Service [Port 8188]" URL)
GPU_BASE_URL=https://<pod-id>-8188.proxy.runpod.net
# GPU_API_KEY=        # only if you later put the pod behind an auth proxy
# GPU_HOURLY_USD=2.89 # estimate rate; defaults to 2.89
```

The pod ID changes every time a *new* pod is created (resuming the same
pod keeps it). If Studio reports "pod unreachable", first check the pod
is actually running, then check this URL matches the console.

From then on, Studio's Create tab offers the four GPU options
(Edit Photo (Your GPU), Animate Photo (Your GPU), Text to Video
(Your GPU), Replace Model (Your GPU)) whenever media is selected as
usual. With the pod stopped they fail soft with a "pod unreachable"
message — nothing else in Studio is affected.
