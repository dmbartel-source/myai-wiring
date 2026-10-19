#!/usr/bin/env python3
# EXTRACT-KEYPOINTS-V1 — DWPose (rtmlib) keypoint extraction for stage 1.
# Reads the 39 stage-1 view images, runs rtmlib Body (DWPose, COCO-17),
# picks the highest-mean-score person per image, and writes the sidecar
# keypoints.json in EXACTLY the format stage1_fit.load_keypoints_for_view
# consumes: {file: {"keypoints": [[x, y, conf] x17] in ORIGINAL image
# pixels, "format": "coco17", "image_size": [w, h]}}.
# Extraction failing for an image is reported per-view, never fatal here;
# stage1_fit decides what missing keypoints mean (fallback exists).
# Usage: extract_keypoints.py --input DIR --out PATH [--device cuda]
#        extract_keypoints.py selftest
import argparse, json, os, sys

MARKER = "EXTRACT-KEYPOINTS-V1"

def pick_person(kps, scores):
    """kps (N,17,2), scores (N,17) -> (17,3) array for the best person."""
    import numpy as np
    if kps is None or len(kps) == 0:
        return None
    mean_scores = scores.mean(axis=1)
    best = int(np.argmax(mean_scores))
    out = np.concatenate([kps[best], scores[best][:, None]], axis=1)
    return out

def write_sidecar(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)

def selftest():
    import numpy as np
    kps = np.random.RandomState(0).rand(2, 17, 2) * 500
    scores = np.random.RandomState(1).rand(2, 17)
    scores[1] += 0.5
    picked = pick_person(kps, scores)
    assert picked.shape == (17, 3)
    assert np.allclose(picked[:, :2], kps[1]), "must pick the higher-score person"
    assert pick_person(np.zeros((0, 17, 2)), np.zeros((0, 17))) is None
    data = {"a.jpg": {"keypoints": picked.tolist(), "format": "coco17",
                      "image_size": [800, 1000]}}
    write_sidecar("/tmp/kp_selftest.json", data)
    back = json.load(open("/tmp/kp_selftest.json"))
    assert back["a.jpg"]["format"] == "coco17" and len(back["a.jpg"]["keypoints"]) == 17
    os.remove("/tmp/kp_selftest.json")
    print(f"SELFTEST-OK {MARKER} person-pick + sidecar round-trip verified")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest(); return
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    assert a.input and a.out, "--input and --out are required"
    try:
        from rtmlib import Body
    except ImportError:
        print("EXTRACT-FAIL: rtmlib not installed in this venv "
              "(uv pip install --python <venv python> rtmlib)", flush=True)
        sys.exit(1)
    from PIL import Image
    import numpy as np
    model = Body(mode="balanced", backend="onnxruntime", device=a.device)
    files = sorted(f for f in os.listdir(a.input)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    data, done, empty = {}, 0, []
    for name in files:
        path = os.path.join(a.input, name)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        kps, scores = model(np.asarray(img))
        picked = pick_person(np.asarray(kps, dtype=np.float64),
                             np.asarray(scores, dtype=np.float64)) \
            if kps is not None else None
        if picked is None:
            empty.append(name)
            print(f"KEYPOINTS-EMPTY {name}", flush=True)
            continue
        data[name] = {"keypoints": picked.tolist(), "format": "coco17",
                      "image_size": [w, h]}
        done += 1
        print(f"KEYPOINTS-OK {name} mean-conf {picked[:, 2].mean():.2f}",
              flush=True)
    write_sidecar(a.out, data)
    print(f"EXTRACTED {done}/{len(files)} empty={len(empty)} -> {a.out}",
          flush=True)
    print("EXTRACT-KEYPOINTS-DONE", flush=True)

if __name__ == "__main__":
    main()
