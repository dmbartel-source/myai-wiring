#!/usr/bin/env python3
# GATE-FIT-V7 — MYAI measured-body GATE TEST (plan: GATE_TEST_PLAN.md).
# Question: can ONE shared SMPL-X shape + per-vertex offsets match the
# silhouettes of 4 hero views (front/side/back/three-quarter) at once?
# Stages: segment -> select (azimuth grid) -> fit (beta, then offsets)
# -> metrics (below-neck IoU primary, bust projection, leave-one-out)
# -> verdict against the plan's PASS/KILL thresholds.
# Pure-metric helpers are torch-free so --selftest runs anywhere.
import argparse, json, math, os, sys

CANDIDATES = ["5766.png", "5760.png", "5761.png",
              "lib_5765_f_0001.jpg", "lib_5765_f_0012.jpg",
              "lib_5784-3_f_0019.jpg", "lib_5817_f_0013.jpg", "5769.png"]
# Heroes LABELED BY DAVID (Oct 18, 2026, via chat uploads matched to
# files by perceptual hash: front/side exact-or-near, back exact).
# Azimuths: front/back from his labels; side 105 and three-quarter
# 215 read from the pictures he showed (side = profile with slight
# back turn; TQ = back three-quarter from her left). The fit refines
# per-view orientation; a mirrored TQ would surface as a lone low IoU.
HEROES = [("front", "lib_5765_f_0001.jpg", 0.0),
          ("three-quarter", "5769.png", 215.0),
          ("side", "lib_5765_f_0012.jpg", 105.0),
          ("back", "5766.png", 180.0)]
ANGLE_SLOTS = {"front": 0.0, "three-quarter": 45.0, "side": 90.0, "back": 180.0}

# ---------------- torch-free metric helpers ----------------

def iou(a, b):
    """a, b: 2D boolean numpy arrays of the same shape."""
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 1.0

def width_profile(mask):
    """Per-row width (rightmost-leftmost+1) of a boolean mask; 0 rows empty."""
    import numpy as np
    prof = np.zeros(mask.shape[0], dtype=np.int64)
    for r in range(mask.shape[0]):
        cols = np.nonzero(mask[r])[0]
        if cols.size:
            prof[r] = cols.max() - cols.min() + 1
    return prof

def person_rows(mask):
    import numpy as np
    rows = np.nonzero(mask.any(axis=1))[0]
    return (int(rows.min()), int(rows.max())) if rows.size else (0, mask.shape[0] - 1)

def bust_delta(mask_mesh, mask_ref, chest_row_frac=0.28, band_frac=0.05):
    """Compare silhouette widths in the chest band between the rendered
    mesh silhouette and the reference mask, both in the SAME crop.
    chest band = rows around chest_row_frac of person height (from top).
    Returns mean |width delta| in band as % of person height, and the
    band's max-width delta %."""
    import numpy as np
    t0, b0 = person_rows(mask_ref)
    h = max(1, b0 - t0)
    c = t0 + chest_row_frac * h
    band = (max(t0, int(c - band_frac * h)), min(b0, int(c + band_frac * h)))
    pm = width_profile(mask_mesh)[band[0]:band[1] + 1].astype(float)
    pr = width_profile(mask_ref)[band[0]:band[1] + 1].astype(float)
    if pr.size == 0 or pr.max() == 0:
        return None
    mean_delta = float(np.abs(pm - pr).mean()) / h * 100.0
    maxw_delta = float(abs(pm.max() - pr.max())) / h * 100.0
    return {"band_rows": [int(band[0]), int(band[1])],
            "mean_delta_pct": mean_delta, "maxwidth_delta_pct": maxw_delta}

def verdict(metrics):
    """metrics: {heroes: {slot: {iou_primary, iou_full}}, bust: {...},
    loo: {iou_primary, bust_mean_delta_pct}} -> (label, reasons)."""
    reasons = []
    ious = [v["iou_primary"] for v in metrics["heroes"].values()]
    bust = metrics["bust"]["mean_delta_pct"]
    loo_iou = metrics["loo"]["iou_primary"]
    if min(ious) >= 0.95 and bust <= 2.0 and loo_iou >= 0.90:
        return "GATE-PASS", ["all hero IoUs >= 0.95, bust delta <= 2%, LOO IoU >= 0.90"]
    if min(ious) < 0.90:
        reasons.append(f"worst hero IoU {min(ious):.3f} < 0.90 after offsets")
    if bust > 2.0:
        reasons.append(f"bust projection delta {bust:.2f}% > 2%")
    if loo_iou < 0.85:
        reasons.append(f"leave-one-out IoU {loo_iou:.3f} < 0.85 (views not mutually predictive)")
    if reasons:
        return "GATE-KILL", reasons
    return "GATE-MIDDLE", [f"IoUs {['%.3f' % i for i in ious]}, bust {bust:.2f}%, LOO {loo_iou:.3f} — between thresholds; evidence goes to David"]

def selftest():
    import numpy as np
    from PIL import Image, ImageDraw
    def blob(w, h, box):
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).ellipse(box, fill=1)
        return np.array(im) > 0
    a = blob(100, 200, (30, 20, 70, 180))
    assert abs(iou(a, a) - 1.0) < 1e-9
    b = blob(100, 200, (35, 20, 75, 180))
    v = iou(a, b)
    assert 0.5 < v < 1.0, v
    assert bust_delta(a, a)["mean_delta_pct"] == 0.0
    wide = blob(100, 200, (25, 20, 75, 180))  # same center, wider body
    d = bust_delta(a, wide)
    assert d and d["mean_delta_pct"] > 0, d
    m = {"heroes": {s: {"iou_primary": 0.97, "iou_full": 0.93} for s in ANGLE_SLOTS},
         "bust": {"mean_delta_pct": 1.2}, "loo": {"iou_primary": 0.93}}
    lab, _ = verdict(m)
    assert lab == "GATE-PASS", lab
    m["heroes"]["side"]["iou_primary"] = 0.88
    lab, _ = verdict(m)
    assert lab == "GATE-KILL", lab
    print("SELFTEST-OK iou/bust/verdict helpers verified on synthetic masks")

# ---------------- stage: segment ----------------

def stage_segment(indir, outdir):
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove
    os.makedirs(outdir, exist_ok=True)
    session = new_session("isnet-general-use")
    report = {}
    for name in CANDIDATES:
        src = os.path.join(indir, name)
        if not os.path.exists(src):
            print(f"SEGMENT-MISSING {name}")
            continue
        out = remove(Image.open(src).convert("RGB"), session=session)
        alpha = np.array(out.split()[-1])
        mask = alpha > 127
        # keep the largest connected component (scipy if present, else raw)
        try:
            from scipy import ndimage
            lab, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                mask = lab == (int(np.argmax(sizes)) + 1)
        except ImportError:
            pass
        frac = float(mask.mean())
        t, b = person_rows(mask)
        dst = os.path.join(outdir, os.path.splitext(name)[0] + "_mask.png")
        tmp = dst + ".tmp.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmp)
        Image.open(tmp).verify()
        os.replace(tmp, dst)
        report[name] = {"mask_frac": frac, "person_px_h": int(b - t)}
        print(f"SEGMENT {name} frac={frac:.3f} person_h={b - t}px")
    json.dump(report, open(os.path.join(outdir, "segment_report.json"), "w"), indent=1)
    ok = len(report)
    print(f"SEGMENT-DONE {ok}/{len(CANDIDATES)}")
    if ok < 6:
        print("GATE-FAIL segment"); sys.exit(1)
    return report

# ---------------- shared torch setup ----------------

def load_model(model_dir, device, num_betas=100):
    import torch, smplx
    probe = os.path.join(model_dir, "smplx", "SMPLX_FEMALE.npz")
    flat = os.path.join(model_dir, "SMPLX_FEMALE.npz")
    assert os.path.exists(probe) or os.path.exists(flat),         f"SMPLX_FEMALE.npz not resolvable under {model_dir}"
    model = smplx.create(model_dir, model_type="smplx", gender="female",
                         num_betas=num_betas, use_pca=False,
                         ext="npz").to(device)
    return model

def find_smplx():
    import glob
    pats = ["/workspace/**/SMPLX_FEMALE.npz", "/workspace/**/smplx/models/**",
            "/workspace/**/models/smplx/**"]
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    files = [h for h in hits if os.path.basename(h) == "SMPLX_FEMALE.npz"]
    if not files:
        return None
    d = os.path.dirname(files[0])
    # smplx.create() appends the model-type folder itself: it must be
    # given the PARENT of the "smplx" folder, not the folder itself
    # (V1 bug: returned the inner folder -> "<dir>/smplx/smplx" assert).
    if os.path.basename(d) == "smplx":
        return os.path.dirname(d)
    return d

# ---------------- rendering + fitting (torch) ----------------

def prep_mask(mask_bool, size=512):
    """Crop to person bbox + 8% margin, pad square, resize to size.
    Returns (square bool mask, crop box)."""
    import numpy as np
    from PIL import Image
    t, b = person_rows(mask_bool)
    cols = np.nonzero(mask_bool.any(axis=0))[0]
    l, r = int(cols.min()), int(cols.max())
    h, w = mask_bool.shape
    mh = int(0.08 * (b - t)); mw = int(0.08 * (r - l))
    t, b = max(0, t - mh), min(h, b + mh)
    l, r = max(0, l - mw), min(w, r + mw)
    crop = mask_bool[t:b + 1, l:r + 1]
    side = max(crop.shape)
    sq = np.zeros((side, side), dtype=bool)
    sq[:crop.shape[0], :crop.shape[1]] = crop
    im = Image.fromarray((sq * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return np.array(im) > 0, (t, b, l, r)

def render_alpha(verts, faces, azim, device, size=512, dist=3.2,
                 dtx=0.0, dty=0.0, scale=1.0, sigma=1e-4):
    import torch
    from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer,
        MeshRenderer, RasterizationSettings, SoftSilhouetteShader, BlendParams,
        look_at_view_transform)
    from pytorch3d.structures import Meshes
    if not torch.is_tensor(dtx):
        dtx = torch.tensor(float(dtx), device=device)
    if not torch.is_tensor(dty):
        dty = torch.tensor(float(dty), device=device)
    if not torch.is_tensor(scale):
        scale = torch.tensor(float(scale), device=device)
    v = verts * scale
    mesh = Meshes(verts=[v], faces=[faces])
    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim, device=device)
    T = T + torch.stack([dtx.reshape(()), dty.reshape(()),
                         torch.zeros((), device=device)]).reshape(1, 3).to(T.dtype)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=38.0)
    rast = RasterizationSettings(image_size=size, blur_radius=0.0,
                                 faces_per_pixel=1, bin_size=0)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=rast),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4)))
    out = renderer(mesh)
    return out[0, ..., 3]  # alpha HxW

def project_points(verts, azim, device, size=512, dist=3.2, dtx=0.0, dty=0.0, scale=1.0):
    import torch
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim, device=device)
    T = T + torch.tensor([[dtx, dty, 0.0]], device=device, dtype=T.dtype)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=38.0)
    pts = (verts * scale).reshape(1, -1, 3)
    return cam.transform_points_screen(pts, image_size=(size, size))[0]  # (x, y-down, z)

def soft_iou(alpha, target):
    import torch
    t = target.float()
    inter = (alpha * t).sum()
    union = (alpha + t - alpha * t).sum()
    return inter / union.clamp(min=1.0)

def init_cam_measured(model, faces, view, device, rounds=20):
    """Measured camera init against the REAL render_alpha params
    (dtx, dty, scale at fixed dist 3.2). Iterates with IoU-gated
    acceptance: an update is kept only if the hard IoU improves, so
    the result is monotone by construction. World-per-pixel at
    dist 3.2 / fov 38 = 2*3.2*tan(19deg)/512 = 0.0043."""
    import torch, numpy as np
    tgt = view["mask512"]
    tys, txs = np.nonzero(tgt)
    th = float(tys.max() - tys.min() + 1)
    tc_r, tc_c = float(tys.mean()), float(txs.mean())
    with torch.no_grad():
        out = model(betas=torch.zeros(1, 100, device=device), return_verts=True)
        verts0 = out.vertices[0]
    def hard_iou(dtx, dty, scale):
        a = render_alpha(verts0, faces, view["azim"], device,
                         dtx=dtx, dty=dty, scale=scale, sigma=1e-5)
        sil = (a > 0.5).cpu().numpy()
        inter = (sil & tgt).sum()
        union = (sil | tgt).sum()
        return (inter / union if union else 0.0), sil
    dtx, dty, scale = 0.0, 0.0, 1.0
    best, _ = hard_iou(dtx, dty, scale)
    for _ in range(rounds):
        _, sil = hard_iou(dtx, dty, scale)
        rys, rxs = np.nonzero(sil)
        if len(rys) == 0:
            scale *= 1.2
            best, _ = hard_iou(dtx, dty, scale)
            continue
        rh = float(rys.max() - rys.min() + 1)
        rc_r, rc_c = float(rys.mean()), float(rxs.mean())
        cand = (dtx - (tc_c - rc_c) * 0.0043 * 0.8,
                dty + (tc_r - rc_r) * 0.0043 * 0.8,
                scale * (th / rh) ** 0.5)
        val, _ = hard_iou(*cand)
        if val >= best:
            dtx, dty, scale = cand
            best = val
        else:
            break
    print(f"INIT {view['name']}: dtx {dtx:.2f} dty {dty:.2f} "
          f"scale {scale:.2f} IoU {best:.3f}", flush=True)
    return [dtx, dty, scale]


def fit_shape(model, faces, views, device, steps_beta=400, steps_off=400,
              skip_metrics_for=None):
    """views: list of dicts {name, mask512 (np bool), azim}. One shared
    shape; per-view pose + camera offsets. Returns (state, per-view
    cam params) where state = {betas, offsets}."""
    import torch, numpy as np
    n = len(views)
    betas = torch.zeros(1, 100, device=device, requires_grad=True)
    poses = [torch.zeros(1, 63, device=device, requires_grad=True) for _ in range(n)]
    orients = [torch.zeros(1, 3, device=device, requires_grad=True) for _ in range(n)]
    print("FIT-GUARDS init=measured clamp=on deadview=on (GATE-FIT-V7)", flush=True)
    cams = []
    for v in views:
        c0 = init_cam_measured(model, faces, v, device)
        cams.append(torch.tensor(c0, device=device, requires_grad=True))
    targets = [torch.tensor(v["mask512"].astype(np.float32), device=device) for v in views]
    opt = torch.optim.Adam([betas] + poses + orients + cams, lr=0.02)
    offsets = torch.zeros(1, 10475, 3, device=device)
    for step in range(steps_beta):
        opt.zero_grad()
        loss = 0.0
        for i, v in enumerate(views):
            out = model(betas=betas, global_orient=orients[i], body_pose=poses[i],
                        return_verts=True)
            alpha = render_alpha(out.vertices[0], faces, v["azim"], device,
                                 dtx=cams[i][0], dty=cams[i][1], scale=cams[i][2])
            loss = loss + (1 - soft_iou(alpha, targets[i])) \
                 + 0.02 * (poses[i] ** 2).mean() + 0.001 * (betas ** 2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            for c in cams:
                c[0] = c[0].clamp(-1.0, 1.0)
                c[1] = c[1].clamp(-1.0, 1.0)
                c[2] = c[2].clamp(0.65, 1.6)
        if step % 100 == 0:
            print(f"  beta step {step} loss {float(loss):.4f}", flush=True)
    # dead-view guard: a view whose render barely overlaps its photo
    # after the beta stage would silently poison the metrics (the
    # Oct 18 side-view collapse). Refit such a view solo, loudly.
    with torch.no_grad():
        for i, v in enumerate(views):
            out = model(betas=betas, global_orient=orients[i], body_pose=poses[i],
                        return_verts=True)
            a = render_alpha(out.vertices[0], faces, v["azim"], device,
                             dtx=cams[i][0], dty=cams[i][1], scale=cams[i][2],
                             sigma=1e-5)
            sil = (a > 0.5).cpu().numpy()
            vi = iou(sil, v["mask512"])
            print(f"  BETA-IOU {v['name']}: {vi:.3f}", flush=True)
            if vi < 0.10:
                print(f"  VIEW-DEAD {v['name']} — solo cam/pose refit", flush=True)
                optd = torch.optim.Adam([poses[i], orients[i], cams[i]], lr=0.02)
                for _ in range(150):
                    optd.zero_grad()
                    o2 = model(betas=betas, global_orient=orients[i],
                               body_pose=poses[i], return_verts=True)
                    a2 = render_alpha(o2.vertices[0], faces, v["azim"], device,
                                      dtx=cams[i][0], dty=cams[i][1], scale=cams[i][2])
                    l2 = (1 - soft_iou(a2, targets[i])) + 0.02 * (poses[i] ** 2).mean()
                    l2.backward(); optd.step()
                    with torch.no_grad():
                        cams[i][0] = cams[i][0].clamp(-1.0, 1.0)
                        cams[i][1] = cams[i][1].clamp(-1.0, 1.0)
                        cams[i][2] = cams[i][2].clamp(0.65, 1.6)
    # offsets stage
    offsets = torch.zeros(1, 10475, 3, device=device, requires_grad=True)
    from pytorch3d.loss import mesh_laplacian_smoothing
    from pytorch3d.structures import Meshes
    opt2 = torch.optim.Adam([offsets], lr=0.01)
    base_verts = []
    with torch.no_grad():
        for i, v in enumerate(views):
            out = model(betas=betas, global_orient=orients[i], body_pose=poses[i],
                        return_verts=True)
            base_verts.append(out.vertices.detach())
    for step in range(steps_off):
        opt2.zero_grad()
        loss = 0.0
        for i, v in enumerate(views):
            verts = base_verts[i] + offsets
            alpha = render_alpha(verts[0], faces, v["azim"], device,
                                 dtx=cams[i][0], dty=cams[i][1], scale=cams[i][2])
            loss = loss + (1 - soft_iou(alpha, targets[i]))
        mesh = Meshes(verts=[base_verts[0][0] + offsets[0]], faces=[faces])
        loss = loss + 2.0 * mesh_laplacian_smoothing(mesh) + 0.5 * (offsets ** 2).mean()
        loss.backward(); opt2.step()
        if step % 100 == 0:
            print(f"  offset step {step} loss {float(loss):.4f}", flush=True)
    state = {"betas": betas.detach(), "offsets": offsets.detach(),
             "poses": [p.detach() for p in poses],
             "orients": [o.detach() for o in orients],
             "cams": [c.detach() for c in cams]}
    return state

def view_metrics(model, faces, state, view, device, size=512):
    """Hard metrics for one view under a fitted state: full + below-neck
    IoU, and (for the side view) the bust-band numbers are computed by
    the caller from the returned silhouettes."""
    import torch, numpy as np
    i = view["idx"]
    with torch.no_grad():
        out = model(betas=state["betas"], global_orient=state["orients"][i],
                    body_pose=state["poses"][i], return_verts=True)
        verts = out.vertices + state["offsets"]
        cam = state["cams"][i]
        alpha = render_alpha(verts[0], faces, view["azim"], device,
                             dtx=float(cam[0]), dty=float(cam[1]), scale=float(cam[2]),
                             sigma=1e-5)
        sil = (alpha > 0.5).cpu().numpy()
        joints = out.joints[0]
        neck = joints[12]  # SMPL-X body joint 12 = neck
        scr = project_points(neck.reshape(1, 3), view["azim"], device,
                             dtx=float(cam[0]), dty=float(cam[1]), scale=float(cam[2]))
        neck_row = int(round(float(scr[0, 1])))
    tgt = view["mask512"]
    full = iou(sil, tgt)
    sil_bn = sil.copy(); sil_bn[:neck_row, :] = False
    tgt_bn = tgt.copy(); tgt_bn[:neck_row, :] = False
    prim = iou(sil_bn, tgt_bn)
    return {"iou_full": full, "iou_primary": prim, "neck_row": neck_row,
            "silhouette": sil}

# ---------------- stage: select heroes by azimuth ----------------

def aligned_iou(verts, faces, mask256, azim, device):
    """Render the mean body at azim, then iteratively align (scale +
    camera shift) by matching the rendered silhouette bbox to the
    mask bbox — outcomes are MEASURED each round, so camera sign
    conventions cannot silently poison the score the way V2's fixed
    camera did. Returns (best_iou, cam dict)."""
    import torch, numpy as np
    tgt = torch.tensor(mask256.astype(np.float32), device=device)
    rows = np.nonzero(mask256.any(axis=1))[0]
    cols = np.nonzero(mask256.any(axis=0))[0]
    mh = rows.max() - rows.min() + 1
    mcx = (cols.min() + cols.max()) / 2.0
    mcy = (rows.min() + rows.max()) / 2.0
    scale, dtx, dty = 1.0, 0.0, 0.0
    best = (-1.0, None)
    world_per_px = (2 * 3.2 * math.tan(math.radians(19.0))) / 256.0
    sx, sy = 1.0, -1.0  # shift sign conventions, self-corrected below
    prev_errx = prev_erry = None
    for it in range(4):
        with torch.no_grad():
            alpha = render_alpha(verts, faces, azim, device, size=256,
                                 dtx=dtx, dty=dty, scale=scale, sigma=1e-3)
        sil = (alpha > 0.5).cpu().numpy()
        v = float(soft_iou(alpha, tgt))
        if v > best[0]:
            best = (v, {"scale": scale, "dtx": dtx, "dty": dty})
        r2 = np.nonzero(sil.any(axis=1))[0]
        c2 = np.nonzero(sil.any(axis=0))[0]
        if r2.size == 0:
            scale *= 1.3
            continue
        rh = r2.max() - r2.min() + 1
        rcx = (c2.min() + c2.max()) / 2.0
        rcy = (r2.min() + r2.max()) / 2.0
        scale *= float(mh) / float(rh)
        errx, erry = mcx - rcx, mcy - rcy
        if prev_errx is not None and abs(errx) > abs(prev_errx):
            sx = -sx
        if prev_erry is not None and abs(erry) > abs(prev_erry):
            sy = -sy
        dtx += sx * errx * world_per_px
        dty += sy * erry * world_per_px
        prev_errx, prev_erry = errx, erry
    return best

def stage_select(model, faces, indir, maskdir, device):
    import torch, numpy as np
    from PIL import Image
    cands = []
    with torch.no_grad():
        mean_verts = model(betas=torch.zeros(1, 100, device=device),
                           return_verts=True).vertices[0]
    for name in CANDIDATES:
        mp = os.path.join(maskdir, os.path.splitext(name)[0] + "_mask.png")
        if not os.path.exists(mp):
            continue
        mask = np.array(Image.open(mp)) > 0
        sq, _ = prep_mask(mask)
        m256 = sq[::2, ::2]
        best = (-1.0, None, None)
        for az in range(0, 360, 15):
            v, cam = aligned_iou(mean_verts, faces, m256, float(az), device)
            if v > best[0]:
                best = (v, float(az), cam)
        cands.append({"name": name, "azim": best[1], "iou": best[0], "mask512": sq})
        print(f"SELECT {name} azim~{best[1]:.0f} aligned-IoU {best[0]:.3f}", flush=True)
    used = set()
    heroes = {}
    for slot, target in ANGLE_SLOTS.items():
        def dist(c):
            d = abs(c["azim"] - target) % 360.0
            return min(d, 360.0 - d)
        pool = sorted((c for c in cands if c["name"] not in used), key=dist)
        pick = pool[0]
        used.add(pick["name"])
        heroes[slot] = {"name": pick["name"], "azim": pick["azim"],
                        "slot_target": target, "azim_off": dist(pick),
                        "mask512": pick["mask512"], "sel_iou": pick["iou"]}
        print(f"HERO {slot}: {pick['name']} (azim {pick['azim']:.0f}, off {dist(pick):.0f}, aligned-IoU {pick['iou']:.3f})", flush=True)
    side = heroes["side"]
    if side["azim_off"] > 30:
        print(f"GATE-FAIL select: no true side view among candidates (best side pick {side['name']} is {side['azim_off']:.0f} deg off side) — the bust-projection check cannot run")
        sys.exit(1)
    worst = min(h["sel_iou"] for h in heroes.values())
    if worst < 0.30:
        print(f"GATE-FAIL select: worst hero aligned-IoU {worst:.3f} < 0.30 — selection measurement untrustworthy, fit would be built on noise")
        sys.exit(1)
    return heroes

# ---------------- stage: run everything ----------------

def stage_run(indir, workdir):
    import torch, numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    model_dir = find_smplx()
    if not model_dir:
        print("GATE-FAIL preflight: SMPLX_FEMALE.npz not found on the volume")
        sys.exit(1)
    print(f"smplx model dir: {model_dir}", flush=True)
    maskdir = os.path.join(workdir, "masks")
    os.makedirs(maskdir, exist_ok=True)
    hero_names = [h[1] for h in HEROES]
    from PIL import Image as _Im
    def _ok(nm):
        p = os.path.join(maskdir, os.path.splitext(nm)[0] + "_mask.png")
        if not os.path.exists(p):
            return False
        try:
            _Im.open(p).verify()
            return True
        except Exception:
            return False
    if all(_ok(nm) for nm in hero_names):
        print("MASKS-REUSED (all heroes verify from a previous run)", flush=True)
    else:
        stage_segment(indir, maskdir)
    model = load_model(model_dir, device)
    faces = model.faces_tensor.to(device)
    from PIL import Image as _Im2
    import numpy as _np
    heroes_map = {}
    for slot, nm, az in HEROES:
        mp = os.path.join(maskdir, os.path.splitext(nm)[0] + "_mask.png")
        m = _np.array(_Im2.open(mp)) > 0
        sq, _ = prep_mask(m)
        heroes_map[slot] = {"name": nm, "azim": az, "mask512": sq}
        print(f"HERO {slot}: {nm} (azim {az:.0f}, DAVID-LABELED)", flush=True)
    views = []
    for i, (slot, h) in enumerate(heroes_map.items()):
        views.append({"idx": i, "slot": slot, "name": h["name"],
                      "azim": h["azim"], "mask512": h["mask512"]})
    print("FIT joint (beta + offsets) on 4 heroes", flush=True)
    state = fit_shape(model, faces, views, device)
    metrics = {"heroes": {}}
    sils = {}
    for v in views:
        m = view_metrics(model, faces, state, v, device)
        metrics["heroes"][v["slot"]] = {"name": v["name"], "azim": v["azim"],
            "iou_primary": m["iou_primary"], "iou_full": m["iou_full"]}
        sils[v["slot"]] = (m["silhouette"], v["mask512"])
        print(f"METRIC {v['slot']} ({v['name']}): IoU below-neck {m['iou_primary']:.3f} full {m['iou_full']:.3f}", flush=True)
    side_sil, side_tgt = sils["side"]
    bd = bust_delta(side_sil, side_tgt)
    if bd is None:
        print("GATE-FAIL metrics: bust band empty on the side view")
        sys.exit(1)
    metrics["bust"] = bd
    print(f"METRIC bust: {bd}", flush=True)
    # leave-one-out on the side view
    print("FIT leave-one-out (no side view)", flush=True)
    loo_views = [v for v in views if v["slot"] != "side"]
    for i, v in enumerate(loo_views):
        v = dict(v); v["idx"] = i; loo_views[i] = v
    loo_state = fit_shape(model, faces, loo_views, device)
    side_view = [v for v in views if v["slot"] == "side"][0]
    # fit only the side view's pose/camera against the LOO shape
    import torch as _t
    pose = _t.zeros(1, 63, device=device, requires_grad=True)
    orient = _t.zeros(1, 3, device=device, requires_grad=True)
    cam = _t.tensor([0.0, 0.0, 1.0], device=device, requires_grad=True)
    opt = _t.optim.Adam([pose, orient, cam], lr=0.02)
    tgt = _t.tensor(side_view["mask512"].astype(np.float32), device=device)
    with _t.no_grad():
        base = model(betas=loo_state["betas"], return_verts=True).vertices + loo_state["offsets"]
    for step in range(200):
        opt.zero_grad()
        out = model(betas=loo_state["betas"], global_orient=orient, body_pose=pose,
                    return_verts=True)
        verts = out.vertices + loo_state["offsets"]
        alpha = render_alpha(verts[0], faces, side_view["azim"], device,
                             dtx=cam[0], dty=cam[1], scale=cam[2])
        loss = (1 - soft_iou(alpha, tgt)) + 0.02 * (pose ** 2).mean()
        loss.backward(); opt.step()
    loo_state2 = {"betas": loo_state["betas"], "offsets": loo_state["offsets"],
                  "poses": [pose.detach()], "orients": [orient.detach()], "cams": [cam.detach()]}
    sv = dict(side_view); sv["idx"] = 0
    lm = view_metrics(model, faces, loo_state2, sv, device)
    metrics["loo"] = {"iou_primary": lm["iou_primary"], "iou_full": lm["iou_full"]}
    print(f"METRIC loo side: IoU below-neck {lm['iou_primary']:.3f} full {lm['iou_full']:.3f}", flush=True)
    lab, reasons = verdict(metrics)
    metrics["verdict"] = lab; metrics["reasons"] = reasons
    json.dump(metrics, open(os.path.join(workdir, "gate_metrics.json"), "w"), indent=1)
    print(f"GATE-VERDICT {lab}: {'; '.join(reasons)}", flush=True)
    print("GATE-RUN-DONE", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "segment", "run"])
    ap.add_argument("--input", default="/workspace/ComfyUI/input/gate_candidates")
    ap.add_argument("--work", default="/workspace/gate_test")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "segment":
        stage_segment(a.input, os.path.join(a.work, "masks"))
    else:
        os.makedirs(a.work, exist_ok=True)
        stage_run(a.input, a.work)

if __name__ == "__main__":
    main()
