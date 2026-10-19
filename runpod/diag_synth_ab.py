#!/usr/bin/env python3
# DIAG-SYNTH-AB-V1 — A/B diagnostic: the PROVEN gate fitter
# (gate_fit.fit_shape, unmodified) vs the stage-1 fitter
# (stage1_fit.fit_joint) on IDENTICAL synthetic views built by
# stage_synth's exact V5 construction (same seeds). Settles whether
# the synth recovery failure lives in the stage-1 fit code or in
# the synthetic data construction. Also prints the zero-body
# hard-vs-soft IoU level check. Pod-side, GPU. Read-only wrt the
# production scripts (imports them; modifies nothing).
import sys
sys.path.insert(0, "/workspace")
import numpy as np
import torch
import gate_fit as G
import stage1_fit as S


def build(device):
    model_dir = G.find_smplx()
    model = G.load_model(model_dir, device)
    faces = model.faces_tensor.to(device)
    rng = np.random.RandomState(7)
    beta_std = np.array([0.8 * (0.82 ** j) if j < 30 else 0.03
                         for j in range(100)])
    betas_true = np.clip(rng.randn(100) * beta_std, -2.0, 2.0)
    with torch.no_grad():
        out_true = model(betas=torch.tensor(
            betas_true, device=device, dtype=torch.float32).reshape(1, -1),
            return_verts=True)
        base = out_true.vertices[0]
        joints_true = out_true.joints[0].cpu().numpy()
    base_np = base.cpu().numpy()
    nrm_np = S.vertex_normals_np(base_np, faces.cpu().numpy())
    s_true = 0.035 * np.sin(3.0 * base_np[:, 1]) * np.cos(2.0 * base_np[:, 0])
    s_true = S.clamp_offsets_np(s_true)
    verts_true = S.apply_scalar_offsets(base_np, nrm_np, s_true)
    azims = [0.0, 45.0, 90.0, 105.0, 135.0, 180.0, 215.0, 270.0]

    def make_views():
        views = []
        vt = torch.tensor(verts_true, device=device, dtype=torch.float32)
        for i, az in enumerate(azims):
            with torch.no_grad():
                a = S.render_alpha_fov(vt, faces, az, device, sigma=1e-5)
            views.append({"idx": i, "file": f"synth_{int(az):03d}.png",
                          "name": f"synth_{int(az):03d}", "sequence": None,
                          "frame_index": None, "hero_role": None,
                          "azimuth_anchor_deg": az, "azim": float(az),
                          "azim_source": "anchor", "fov": S.FOV_DEFAULT,
                          "mask512": (a > 0.5).cpu().numpy(),
                          "keypoints": None})
        smplx_idx = [sd for _, sd in S.COCO_TO_SMPLX]
        kp_rng = np.random.RandomState(1234)
        kp_conf = np.array([0.45 if sd in (20, 21, 7, 8) else 0.9
                            for _, sd in S.COCO_TO_SMPLX])
        for v in views:
            with torch.no_grad():
                scr = S.project_points_fov(
                    torch.tensor(joints_true[smplx_idx], device=device,
                                 dtype=torch.float32),
                    v["azim"], device, dtx=0.0, dty=0.0, scale=1.0,
                    fov=S.FOV_DEFAULT)
            xy = scr[:, :2].cpu().numpy() + kp_rng.randn(len(smplx_idx), 2) * 2.0
            v["keypoints"] = np.concatenate([xy, kp_conf[:, None]], axis=1)
        return views

    return model, faces, betas_true, make_views


def beta_errs(bhat, btrue):
    e = np.abs(np.asarray(bhat, dtype=np.float64).reshape(-1) - btrue)
    return float(e.max()), float(e[:30].mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, faces, betas_true, make_views = build(device)
    print(f"DIAG betas_true norm {np.linalg.norm(betas_true):.3f}", flush=True)
    views0 = make_views()
    with torch.no_grad():
        zverts = model(betas=torch.zeros(1, 100, device=device),
                       return_verts=True).vertices[0]
    hs, ss = [], []
    for v in views0:
        with torch.no_grad():
            ah = S.render_alpha_fov(zverts, faces, v["azim"], device, sigma=1e-5)
            asf = S.render_alpha_fov(zverts, faces, v["azim"], device, sigma=1e-4)
        hs.append(G.iou((ah > 0.5).cpu().numpy(), v["mask512"]))
        ss.append(float(G.soft_iou(asf, torch.tensor(
            v["mask512"].astype(np.float32), device=device))))
    print(f"DIAG-LEVEL zero-body identity-cam: hard IoU mean {np.mean(hs):.3f} "
          f"| soft_iou(sigma=1e-4) mean {np.mean(ss):.3f}", flush=True)
    vg = make_views()
    st_g = G.fit_shape(model, faces, vg, device, steps_beta=400, steps_off=0)
    mx, mn = beta_errs(st_g["betas"].cpu().numpy(), betas_true)
    print(f"DIAG-GATE beta max {mx:.3f} mean30 {mn:.3f}", flush=True)
    vs = make_views()
    st_s = S.fit_joint(model, faces, vs, device, steps_a=400, steps_b=150,
                       anchor=None, height_m=1.6256)
    mx2, mn2 = beta_errs(st_s["betas"].cpu().numpy(), betas_true)
    print(f"DIAG-STAGE1 beta max {mx2:.3f} mean30 {mn2:.3f}", flush=True)
    print("DIAG-DONE", flush=True)


main()
