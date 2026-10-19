# EXPORT-SCAFFOLD-V1 — rerun the gate deep fit (GATE-FIT-V8 recipe,
# David-labeled heroes) and SAVE the fitted body: scaffold.npz
# (betas + offsets) + scaffold_body.ply (neutral-pose mesh, ASCII PLY
# with vertex normals) in the work dir; the stage script copies the
# PLY into ComfyUI/output for pickup. Metrics print for comparison
# against the gate run (front .852 / TQ .701 / side .833 / back .803,
# bust 0.94%) — a materially different print means this export is not
# the gate body and must not be shown as it.
import os, sys, json
sys.path.insert(0, "/workspace")
import numpy as np
import torch
import gate_fit as G

def write_ply(path, verts, faces, normals):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, n in zip(verts, normals):
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for fa in faces:
            f.write(f"3 {fa[0]} {fa[1]} {fa[2]}\n")

def main():
    indir, workdir = sys.argv[1], sys.argv[2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = G.find_smplx()
    print(f"smplx model dir: {model_dir}", flush=True)
    model = G.load_model(model_dir, device)
    faces_t = model.faces_tensor.to(device)
    from PIL import Image as Im
    maskdir = os.path.join(workdir, "masks")
    views = []
    for i, (slot, nm, az) in enumerate(G.HEROES):
        mp = os.path.join(maskdir, os.path.splitext(nm)[0] + "_mask.png")
        m = np.array(Im.open(mp)) > 0
        sq, _ = G.prep_mask(m)
        views.append({"idx": i, "slot": slot, "name": nm, "azim": az,
                      "mask512": sq})
        print(f"HERO {slot}: {nm} (azim {az:.0f})", flush=True)
    state = G.fit_shape(model, faces_t, views, device)
    # metrics (verification against the gate run)
    worst = {}
    for v in views:
        m = G.view_metrics(model, faces_t, state, v, device)
        worst[v["slot"]] = m["iou_primary"]
        print(f"METRIC {v['slot']}: IoU below-neck {m['iou_primary']:.3f} full {m['iou_full']:.3f}", flush=True)
    # neutral-pose body = betas + offsets, zero pose
    with torch.no_grad():
        out = model(betas=state["betas"], return_verts=True)
        verts = (out.vertices + state["offsets"])[0].cpu().numpy()
    faces = model.faces_tensor.cpu().numpy()
    # vertex normals by face-normal accumulation
    v = verts.astype(np.float64); fa = faces.astype(np.int64)
    fn = np.cross(v[fa[:, 1]] - v[fa[:, 0]], v[fa[:, 2]] - v[fa[:, 0]])
    nrm = np.zeros_like(v)
    for k in range(3):
        np.add.at(nrm, fa[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True); ln[ln == 0] = 1
    nrm = (nrm / ln).astype(np.float32)
    np.savez(os.path.join(workdir, "scaffold.npz"),
             betas=state["betas"].cpu().numpy(),
             offsets=state["offsets"].cpu().numpy())
    write_ply(os.path.join(workdir, "scaffold_body.ply"), verts, faces, nrm)
    print(f"EXPORT-DONE ply verts {len(verts)} faces {len(faces)} "
          f"metrics {json.dumps(worst)}", flush=True)

if __name__ == "__main__":
    main()
