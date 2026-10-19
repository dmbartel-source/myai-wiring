#!/usr/bin/env python3
# STAGE1-FIT-V5 — MYAI measured-body STAGE-1 scaffold fit.
# Implements STAGE1_FITTER_DESIGN.md (S0-S5) on top of the GATE-FIT-V8
# machinery (gate_fit.py is imported lazily and NEVER modified):
#   S0 ingest: manifest-driven views, gate-proven mask prep, keypoints
#      from a sidecar keypoints.json (loader only; extraction is a
#      separate step with its own preflight).
#   S1 cameras: per-sequence focal measured at init (gate's fixed
#      fov=38 generalized), smooth monotone azimuth trajectory anchored
#      to David's hero labels (front 0 / side 105 / back 180 / TQ 215).
#   S2 init: regressor predictions read from a sidecar init.json when
#      present (robust uncertainty-weighted beta fusion = the shape
#      ANCHOR); without it, gate-proven measured-IoU cameras + zero
#      betas, printed loudly as INIT-FALLBACK. Nothing is faked.
#   S3 joint fit: shared betas (anchor trust region), per-view axis-
#      angle pose, trajectory-constrained cameras, GMoF keypoint term
#      (rho=100) when keypoints exist, silhouette at low weight,
#      ONE global scale solved so neutral height = 1.6256 m then FIXED,
#      outlier-view rejection (>2.5 sigma) with the dropped list saved.
#   S4 offsets: CANONICAL scalar field along surface normals (never
#      free 3-vectors — the gate-export failure mode), hard clamp
#      0.15 m, Laplacian + magnitude regularization, per-vertex
#      confidence from view coverage; unobserved vertices stay at 0.
#      Optional normals / pointmap evidence from sidecar dirs, each
#      with a loud MISSING behavior when absent.
#   S5 export + validation: scaffold.npz (betas, scalar offsets,
#      confidence, global scale), neutral-pose PLY, in-process
#      MESH-VALIDATE geometry checks, virtual tape measurements.
#      Validation FAIL prints STAGE1-VALIDATION-FAIL and exits 2 —
#      this script never prints success over a failed mesh.
# Torch-free helpers carry the selftest: `python3 stage1_fit.py
# selftest` runs on plain CPU python (no CUDA, no smplx, no torch).
import argparse, json, math, os, sys

MARKER = "STAGE1-FIT-V6"
HEIGHT_M_DEFAULT = 1.6256          # David-supplied (5 ft 4 in), Oct 18, 2026
GMOF_RHO = 100.0                   # SMPLify-X lineage (design: Values)
OFFSET_CLAMP_M = 0.15              # design hard clamp on scalar offsets
BETA_CLAMP = 2.5                   # physical bound on shape betas (hard clamp)
OUTLIER_SIGMA = 2.5                # design outlier-view rejection rule
SIGMA_FLOOR = 0.02                 # documented floor so a tight MAD can't mass-drop
RENDER_SIZE = 512
RENDER_DIST = 3.2                  # gate-proven camera distance
FOV_DEFAULT = 38.0                 # gate-proven field of view (deg)
SILHOUETTE_SIGMA = 1e-7              # calibrated Oct 19, 2026 (pod probes
                                     # probe_sigma/2/3): at sigma=1e-4 the
                                     # soft-silhouette interior alpha caps
                                     # at ~0.51 against binary targets,
                                     # biasing every fit outward; at 1e-7
                                     # soft_iou is 0.911 at the true body
                                     # and DROPS for perturbed (0.892),
                                     # +3% (0.867) and -3% (0.854) bodies:
                                     # the truth is the strict maximum.
FOV_GRID = (30.0, 38.0, 46.0)      # measured per-sequence focal candidates
NUM_VERTS = 10475                  # SMPL-X topology
# COCO-17 -> SMPL-X body-joint indices (first 22 = SMPL kinematic order;
# gate_fit.py already relies on joints[12] == neck). Eyes/ears unused;
# wrists/ankles get low trust per the design (hands/feet low-trust).
COCO_TO_SMPLX = [(0, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20),
                 (10, 21), (11, 1), (12, 2), (13, 4), (14, 5), (15, 7),
                 (16, 8)]
KP_TRUST = {15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
            20: 0.5, 21: 0.5, 1: 1.0, 2: 1.0, 4: 1.0, 5: 1.0,
            7: 0.5, 8: 0.5}        # keyed by SMPL-X joint index

_G = None
def gate():
    """Lazy gate_fit import (torch-free at module level in gate_fit,
    but lazy keeps `selftest` bulletproof on any machine)."""
    global _G
    if _G is None:
        import gate_fit as g
        _G = g
    return _G

# ---------------- torch-free helpers ----------------

def gmof_np(residuals, rho=GMOF_RHO):
    """Geman-McClure robust loss (SMPLify-X GMof): mean of
    rho^2 * r^2 / (rho^2 + r^2). Quadratic near 0, saturates at rho^2."""
    import numpy as np
    r2 = np.asarray(residuals, dtype=np.float64) ** 2
    return float(np.mean(rho * rho * r2 / (rho * rho + r2)))

def fuse_betas(beta_list, weights=None):
    """Robust per-coefficient weighted median of per-view beta vectors
    (Sengupta-pattern fusion) -> (anchor, spread). Spread = median
    per-coefficient MAD*1.4826 across coefficients."""
    import numpy as np
    B = np.asarray(beta_list, dtype=np.float64)
    assert B.ndim == 2 and len(B) >= 1
    n, d = B.shape
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64).copy()
        w[w < 0] = 0
        if w.sum() <= 0:
            w = np.ones(n)
    anchor = np.zeros(d)
    for j in range(d):
        order = np.argsort(B[:, j])
        vals, ws = B[order, j], w[order]
        cw = np.cumsum(ws)
        anchor[j] = vals[min(int(np.searchsorted(cw, 0.5 * cw[-1])),
                             n - 1)]
    mad = np.median(np.abs(B - anchor), axis=0) * 1.4826
    return anchor, float(np.median(mad))

def clamp_offsets_np(s, limit=OFFSET_CLAMP_M):
    import numpy as np
    return np.clip(np.asarray(s, dtype=np.float64), -limit, limit)

def clamp_betas_np(x, limit=BETA_CLAMP):
    import numpy as np
    return np.clip(np.asarray(x, dtype=np.float64), -limit, limit)

def mesh_height_np(verts):
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    return float(v[:, 1].max() - v[:, 1].min())

def solve_height_scale(current_height, target=HEIGHT_M_DEFAULT):
    """The ONE global scale: target / current (design S3)."""
    assert current_height > 0
    return float(target) / float(current_height)

def vertex_normals_np(verts, faces):
    """Face-normal accumulation, normalized (pattern from
    export_scaffold.py, which shipped the gate body PLY)."""
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    fa = np.asarray(faces, dtype=np.int64)
    fn = np.cross(v[fa[:, 1]] - v[fa[:, 0]], v[fa[:, 2]] - v[fa[:, 0]])
    nrm = np.zeros_like(v)
    for k in range(3):
        np.add.at(nrm, fa[:, k], fn)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return nrm / ln

def apply_scalar_offsets(verts, normals, s):
    """verts + s_i * n_i — the ONLY way offsets ever move a vertex."""
    import numpy as np
    return (np.asarray(verts, dtype=np.float64)
            + np.asarray(s, dtype=np.float64)[:, None]
            * np.asarray(normals, dtype=np.float64))

def outlier_drop(residuals, sigma_mult=OUTLIER_SIGMA, sigma_floor=SIGMA_FLOOR):
    """Design rule: drop views whose residual exceeds median + k*sigma,
    sigma = 1.4826*MAD floored at sigma_floor. Returns
    (keep_mask, dropped_indices, median, sigma)."""
    import numpy as np
    r = np.asarray(residuals, dtype=np.float64)
    med = float(np.median(r))
    sigma = max(1.4826 * float(np.median(np.abs(r - med))), sigma_floor)
    thr = med + sigma_mult * sigma
    dropped = [int(i) for i in np.nonzero(r > thr)[0]]
    keep = np.ones(len(r), dtype=bool)
    keep[dropped] = False
    return keep, dropped, med, sigma

def map_keypoints_to_crop(kps_xy, cropbox, size=RENDER_SIZE):
    """Image-pixel keypoints -> the prep_mask square-crop space.
    cropbox = (t, b, l, r) inclusive, as returned by gate prep_mask;
    the crop is placed top-left in a square of side max(h, w) and
    resized to size — the same transform the mask undergoes."""
    import numpy as np
    t, b, l, r = cropbox
    side = float(max(b - t + 1, r - l + 1))
    k = np.asarray(kps_xy, dtype=np.float64).copy()
    k[:, 0] = (k[:, 0] - l) * size / side
    k[:, 1] = (k[:, 1] - t) * size / side
    return k

# ---------------- trajectory (S1, torch-free core) ----------------

def _unwrap360(azims):
    """Continuous unwrap of degree azimuths (manual, version-proof)."""
    import numpy as np
    a = np.asarray(azims, dtype=np.float64).copy()
    if len(a) < 2:
        return a
    d = np.diff(a)
    d = (d + 180.0) % 360.0 - 180.0
    return np.concatenate([[a[0]], a[0] + np.cumsum(d)])

def monotone_project(azims):
    """Unwrap then cumulative-max: enforces a non-decreasing (smooth
    orbit) azimuth sequence in frame order."""
    import numpy as np
    u = _unwrap360(azims)
    return np.maximum.accumulate(u)

def monotone_project_with_fixed(azims, fixed_idx, fixed_val):
    """Monotone projection holding azims[fixed_idx] == fixed_val:
    cummax forward from the fixed point, cummin backward from it."""
    import numpy as np
    u = _unwrap360(azims)
    u = u - u[fixed_idx] + fixed_val
    out = u.copy()
    if fixed_idx < len(u) - 1:
        out[fixed_idx + 1:] = fixed_val + np.maximum.accumulate(
            np.maximum(u[fixed_idx + 1:] - fixed_val, 0.0))
    if fixed_idx > 0:
        rev = fixed_val - np.maximum.accumulate(
            np.maximum(fixed_val - u[:fixed_idx][::-1], 0.0))
        out[:fixed_idx] = rev[::-1]
    out[fixed_idx] = fixed_val
    return out

def trajectory_line(anchor_frames, anchor_azims):
    """Least-squares azimuth = a + rate*frame through >= 2 anchors."""
    import numpy as np
    f = np.asarray(anchor_frames, dtype=np.float64)
    a = _unwrap360(np.asarray(anchor_azims, dtype=np.float64))
    A = np.stack([np.ones_like(f), f], axis=1)
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    return float(coef[0]), float(coef[1])

def trajectory_base_azimuths(views):
    """Per-view base azimuth from the manifest anchors alone.
    Returns {file: (azim_deg or None, source)} where source is
    'anchor' (David-labeled), 'line' (>= 2 anchors in the sequence,
    linear in frame_index), or 'measure' (no anchor information —
    azimuth must be measured at fit start, gate-proven method)."""
    from collections import defaultdict
    seq_anchors = defaultdict(list)
    for v in views:
        if v.get("azimuth_anchor_deg") is not None and v.get("sequence"):
            seq_anchors[v["sequence"]].append(
                (v["frame_index"], float(v["azimuth_anchor_deg"])))
    lines = {}
    for seq, pts in seq_anchors.items():
        if len(pts) >= 2:
            lines[seq] = trajectory_line([p[0] for p in pts],
                                         [p[1] for p in pts])
    out = {}
    for v in views:
        f = v["file"]
        if v.get("azimuth_anchor_deg") is not None:
            out[f] = (float(v["azimuth_anchor_deg"]), "anchor")
        elif v.get("sequence") in lines:
            a0, rate = lines[v["sequence"]]
            out[f] = (a0 + rate * float(v["frame_index"]), "line")
        else:
            out[f] = (None, "measure")
    return out

# ---------------- mesh slicing / tape (S5c, S5d; torch-free) ----------------

def slice_points_np(verts, faces, y):
    """Intersection points of the mesh with the horizontal plane y."""
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    fa = np.asarray(faces, dtype=np.int64)
    pts = []
    for tri in fa:
        p = v[tri]
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            a, b = p[e0], p[e1]
            if (a[1] - y) * (b[1] - y) < 0:
                t = (y - a[1]) / (b[1] - a[1])
                pts.append(a + t * (b - a))
    return np.array(pts) if pts else np.zeros((0, 3))

def hull_perimeter_xz(points):
    """Convex-hull perimeter of XZ points (monotone chain, numpy-only).
    Convention (stated per design S5d): hull perimeter of the plane-
    slice points — near-convex slices (chest/waist/hip) only."""
    import numpy as np
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 3:
        return 0.0
    P = np.unique(P, axis=0)
    if len(P) < 3:
        return 0.0
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    closed = np.concatenate([hull, hull[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

def tape_measurements(verts, faces):
    """Virtual tape measurements (design S5d). CONVENTION: horizontal
    plane slices at fixed fractions of the robust height (validator's
    0.1/99.9-percentile band): chest 0.78, waist 0.62, hip 0.52;
    circumference = convex-hull perimeter of the slice points.
    Bust conventions differ ~10 cm between protocols — this one is
    stated, fixed, and used consistently for synth-vs-real compares."""
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    y0, y1 = np.percentile(v[:, 1], [0.1, 99.9])
    h = float(y1 - y0)
    out = {"convention": "plane-slice hull perimeter at fixed robust-"
                         "height fractions (chest .78 / waist .62 / hip .52)",
           "height_m": h}
    for name, frac in (("chest", 0.78), ("waist", 0.62), ("hip", 0.52)):
        yy = y0 + frac * h
        pts = slice_points_np(v, faces, yy)
        if len(pts) >= 3:
            w = float(pts[:, 0].max() - pts[:, 0].min())
            d = float(pts[:, 2].max() - pts[:, 2].min())
            out[name] = {"width_m": w, "depth_m": d,
                         "circumference_m": hull_perimeter_xz(pts[:, [0, 2]])}
        else:
            out[name] = None
    return out

# ---------------- validator loading (S5c) ----------------

def validator_candidates(cli_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [cli_path, os.environ.get("STAGE1_VALIDATOR"),
             os.path.join(here, "..", "..", "goals", "myai", "fitter",
                          "mesh_validate.py"),
             "/workspace/mesh_validate.py",
             os.path.join(here, "mesh_validate.py")]
    return [os.path.normpath(c) for c in cands if c]

def load_validator(cli_path=None):
    """Import MESH-VALIDATE from the first existing candidate path.
    Returns (validate_fn, source). Falls back to the vendored copy
    below (verbatim MESH-VALIDATE-V1 logic) with source 'VENDORED' —
    the caller prints the source loudly; a vendored run is a flagged
    run, never a silent one."""
    import importlib.util
    for c in validator_candidates(cli_path):
        if os.path.exists(c):
            spec = importlib.util.spec_from_file_location("mesh_validate", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.validate, c
    return _vendored_validate, "VENDORED"

def _vendored_read_ply(path):
    # Vendored verbatim from mesh_validate.py (MESH-VALIDATE-V1).
    import struct
    import numpy as np
    with open(path, 'rb') as f:
        data = f.read()
    head_end = data.index(b'end_header\n') + len(b'end_header\n')
    head = data[:head_end].decode('ascii', 'replace')
    fmt = 'ascii'
    nv = nf = 0
    vprops, fprops = [], []
    cur = None
    for line in head.splitlines():
        p = line.split()
        if not p: continue
        if p[0] == 'format': fmt = p[1]
        elif p[0] == 'element':
            cur = p[1]
            if cur == 'vertex': nv = int(p[2])
            elif cur == 'face': nf = int(p[2])
        elif p[0] == 'property':
            (vprops if cur == 'vertex' else fprops).append(p)
    if nf == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    body = data[head_end:]
    if fmt == 'ascii':
        toks = body.decode('ascii', 'replace').split()
        it = iter(toks)
        nvp = len([p for p in vprops if p[1] != 'list'])
        V = np.array([[float(next(it)) for _ in range(nvp)] for _ in range(nv)])[:, :3]
        F = []
        for _ in range(nf):
            n = int(next(it)); F.append([int(next(it)) for _ in range(n)])
        return V, np.array(F)
    vsize = 0; types = {'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8),
                        'uchar': ('B', 1), 'uint8': ('B', 1), 'int': ('i', 4), 'int32': ('i', 4)}
    vfmt = ''
    for p in vprops:
        if p[1] == 'list': continue
        vfmt += types[p[1]][0]; vsize += types[p[1]][1]
    V = np.zeros((nv, 3))
    off = 0
    vstruct = struct.Struct('<' + vfmt)
    for i in range(nv):
        vals = vstruct.unpack_from(body, off); off += vsize
        V[i] = vals[:3]
    F = []
    for _ in range(nf):
        n = body[off]; off += 1
        F.append(list(struct.unpack_from('<%di' % n, body, off))); off += 4 * n
    return V, np.array(F)

def _vendored_validate(path):
    # Vendored verbatim from mesh_validate.py (MESH-VALIDATE-V1).
    import numpy as np
    from collections import defaultdict
    V, F = _vendored_read_ply(path)
    if F is None or len(F) == 0:
        return {'verdict': 'FAIL', 'fails': ['no faces: not a mesh (splat/point PLY out of scope)'],
                'verts': len(V), 'faces': 0}
    nbrs = defaultdict(set)
    for a, b, c in F:
        nbrs[int(a)].update((int(b), int(c))); nbrs[int(b)].update((int(a), int(c))); nbrs[int(c)].update((int(a), int(b)))
    runaway = []
    for i, ns in nbrs.items():
        nd = np.linalg.norm(V[list(ns)] - V[i], axis=1)
        if nd.min() > 0.15: runaway.append(i)
    ok_mask = np.ones(len(V), bool); ok_mask[runaway] = False
    W = V[ok_mask]
    out = {'verts': len(V), 'faces': len(F), 'runaway': len(runaway)}
    y0, y1 = np.percentile(W[:, 1], [0.1, 99.9])
    out['height'] = float(y1 - y0)
    ratios = {}
    for name, frac in (('chest', 0.78), ('waist', 0.62)):
        yy = y0 + frac * (y1 - y0)
        band = W[np.abs(W[:, 1] - yy) < 0.02 * (y1 - y0)]
        if len(band) > 20:
            w = band[:, 0].max() - band[:, 0].min(); d = band[:, 2].max() - band[:, 2].min()
            ratios[name] = float(d / w) if w > 0 else 0.0
            out[f'{name}_width'] = float(w); out[f'{name}_depth'] = float(d)
    out['ratios'] = ratios
    fails = []
    if out['runaway'] > 0: fails.append(f"H1 runaway vertices: {out['runaway']}")
    for name, r in ratios.items():
        if not (0.40 <= r <= 0.80): fails.append(f'H2 {name} depth/width {r:.2f} outside 0.40-0.80')
    out['verdict'] = 'FAIL' if fails else 'PASS'
    out['fails'] = fails
    return out

# ---------------- manifest / sidecar loading (torch-free) ----------------

def manifest_candidates(cli_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [cli_path, os.environ.get("STAGE1_MANIFEST"),
             os.path.join(here, "..", "..", "goals", "myai", "fitter",
                          "view_manifest.json"),
             "/workspace/view_manifest.json"]
    return [os.path.normpath(c) for c in cands if c]

def load_manifest(cli_path=None):
    for c in manifest_candidates(cli_path):
        if os.path.exists(c):
            m = json.load(open(c))
            assert m["view_count"] == len(m["views"]), \
                f"manifest view_count {m['view_count']} != {len(m['views'])} views"
            return m, c
    raise FileNotFoundError(f"view manifest not found in {manifest_candidates(cli_path)}")

def load_sidecar(path, label):
    """Sidecar JSON loader with the project's MISSING behavior:
    absent is reported, never fatal, never faked."""
    if path and os.path.exists(path):
        print(f"{label}-SIDECAR-FOUND {path}", flush=True)
        return json.load(open(path))
    print(f"{label}-SIDECAR-MISSING {path or '(no path)'}", flush=True)
    return None

def write_ply(path, verts, faces, normals):
    """ASCII PLY with vertex normals (pattern from export_scaffold.py)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, n in zip(verts, normals):
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} "
                    f"{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for fa in faces:
            f.write(f"3 {fa[0]} {fa[1]} {fa[2]}\n")

# ---------------- selftest (torch-free; runs anywhere) ----------------

def _lathe_mesh(zscale=0.6, runaway=False, height=1.7, rings=60, segs=48):
    """Synthetic person-like surface of revolution with elliptical
    cross-sections (z scaled) for validator tests. Returns
    (verts, faces) as numpy arrays."""
    import numpy as np
    prof = [(0.00, 0.10), (0.30, 0.085), (0.50, 0.13), (0.52, 0.19),
            (0.62, 0.155), (0.78, 0.20), (0.85, 0.17), (0.92, 0.06),
            (0.97, 0.09), (1.00, 0.02)]
    fr = np.array([p[0] for p in prof]); rr = np.array([p[1] for p in prof])
    verts = []
    for i in range(rings + 1):
        f = i / rings
        y = f * height
        r = float(np.interp(f, fr, rr))
        for j in range(segs):
            th = 2 * math.pi * j / segs
            verts.append([r * math.cos(th), y, zscale * r * math.sin(th)])
    verts = np.array(verts)
    faces = []
    for i in range(rings):
        for j in range(segs):
            a = i * segs + j; b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + j; d = (i + 1) * segs + (j + 1) % segs
            faces.append([a, c, b]); faces.append([b, c, d])
    if runaway:
        verts[len(verts) // 2] = verts[len(verts) // 2] + np.array([1.2, 0, 0])
    return verts, np.array(faces)

def _write_test_ply(path, verts, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for fa in faces:
            f.write(f"3 {fa[0]} {fa[1]} {fa[2]}\n")

def selftest():
    import numpy as np
    import tempfile
    # 1. manifest loading (the frozen 39-view set)
    m, mpath = load_manifest()
    assert len(m["views"]) == 39, len(m["views"])
    assert abs(m["subject_height_m"] - 1.6256) < 1e-9
    heroes = {v["file"]: v["azimuth_anchor_deg"] for v in m["views"]
              if v.get("hero_role")}
    assert heroes == {"lib_5765_f_0001.jpg": 0, "lib_5765_f_0012.jpg": 105,
                      "5766.png": 180, "5769.png": 215}, heroes
    print(f"SELFTEST manifest OK ({mpath})")
    # 2. GMoF shape: zero at 0, ~r^2 for small r, saturates at rho^2
    assert gmof_np([0.0]) == 0.0
    small = gmof_np([1.0])
    assert abs(small - 1.0) / 1.0 < 1e-3, small
    big = gmof_np([1e6])
    assert abs(big - GMOF_RHO ** 2) / GMOF_RHO ** 2 < 1e-6, big
    seq = [gmof_np([r]) for r in (1, 2, 5, 10, 50, 200)]
    assert all(b > a for a, b in zip(seq, seq[1:])), seq
    print("SELFTEST gmof OK")
    # 3. scalar offsets: exact displacement along normals + clamp
    rng = np.random.RandomState(0)
    V = rng.randn(50, 3)
    N = V / np.linalg.norm(V, axis=1, keepdims=True)
    out = apply_scalar_offsets(V, N, np.full(50, 0.05))
    disp = np.linalg.norm(out - V, axis=1)
    assert np.allclose(disp, 0.05, atol=1e-9), disp[:3]
    c = clamp_offsets_np(np.array([-0.3, 0.05, 0.42]))
    assert c[0] == -0.15 and c[1] == 0.05 and c[2] == 0.15, c
    cb = clamp_betas_np(np.array([-3.5, 0.7, 2.5]))
    assert cb[0] == -2.5 and cb[1] == 0.7 and cb[2] == 2.5, cb
    print("SELFTEST scalar-offsets OK")
    print("SELFTEST betas-clamp OK")
    # 4. height-scale solve on synthetic verts
    Vh = np.array([[0, 0, 0], [0.2, 2.0, 0], [0.1, 1.0, 0.3]])
    s = solve_height_scale(mesh_height_np(Vh), 1.6256)
    assert abs(s - 0.8128) < 1e-9, s
    assert abs(mesh_height_np(Vh * s) - 1.6256) < 1e-9
    print("SELFTEST height-scale OK")
    # 5. trajectory anchors: 5765 line through (1, 0) and (12, 105)
    base = trajectory_base_azimuths(m["views"])
    a5, src5 = base["lib_5765_f_0005.jpg"]
    assert src5 == "line" and abs(a5 - (105.0 / 11.0) * 4) < 1e-6, (a5, src5)
    a1, src1 = base["lib_5765_f_0001.jpg"]
    assert src1 == "anchor" and a1 == 0.0
    a19, src19 = base["lib_5784-3_f_0019.jpg"]
    assert src19 == "measure" and a19 is None
    line = [base[f"lib_5765_f_{i:04d}.jpg"][0]
            for i in (1, 5, 6, 7, 12, 13)]
    assert all(b > a for a, b in zip(line, line[1:])), line
    mp_ = monotone_project([350.0, 355.0, 5.0, 20.0])
    assert np.allclose(mp_, [350.0, 355.0, 365.0, 380.0]), mp_
    mf = monotone_project_with_fixed([10.0, 30.0, 20.0, 60.0], 2, 25.0)
    assert mf[2] == 25.0 and np.all(np.diff(mf) >= 0), mf
    print("SELFTEST trajectory OK")
    # 6. beta fusion: weighted median shrugs off one outlier view
    inl = np.random.RandomState(1).randn(8, 10) * 0.5
    B = np.concatenate([inl, np.full((1, 10), 8.0)])
    anchor, spread = fuse_betas(B)
    assert np.abs(anchor).max() < 1.0, anchor
    assert spread > 0
    print("SELFTEST beta-fusion OK")
    # 7. outlier rule: one gross residual dropped, inliers kept
    r = np.array([0.05, 0.045, 0.055, 0.05, 0.048, 0.5])
    keep, dropped, med, sig = outlier_drop(r)
    assert dropped == [5] and keep.sum() == 5, (dropped, med, sig)
    print("SELFTEST outlier-rule OK")
    # 8. keypoint crop mapping: center of crop stays center-ish
    kp = map_keypoints_to_crop(np.array([[150.0, 250.0]]), (100, 299, 50, 249))
    assert np.allclose(kp, [[256.0, 384.0]]), kp
    print("SELFTEST keypoint-mapping OK")
    # 9. validator: synthetic BAD mesh must FAIL, SANE mesh must PASS
    validate, vsrc = load_validator()
    print(f"SELFTEST validator source: {vsrc}")
    with tempfile.TemporaryDirectory() as td:
        bv, bf = _lathe_mesh(zscale=1.05, runaway=True)
        bp = os.path.join(td, "bad.ply")
        _write_test_ply(bp, bv, bf)
        vb = validate(bp)
        assert vb["verdict"] == "FAIL", vb
        assert any("H1" in x for x in vb["fails"]), vb["fails"]
        assert any("H2" in x for x in vb["fails"]), vb["fails"]
        sv_, sf_ = _lathe_mesh(zscale=0.6, runaway=False)
        sp = os.path.join(td, "sane.ply")
        _write_test_ply(sp, sv_, sf_)
        vs = validate(sp)
        assert vs["verdict"] == "PASS", vs
    print("SELFTEST validator OK (bad mesh FAILs, sane mesh PASSes)")
    # 10. tape measurements on the sane lathe: chest wider than waist
    tape = tape_measurements(sv_, sf_)
    assert tape["chest"] and tape["waist"], tape
    assert tape["chest"]["circumference_m"] > tape["waist"]["circumference_m"]
    assert abs(tape["height_m"] - 1.7) < 0.05, tape["height_m"]
    print("SELFTEST tape OK")
    print(f"SELFTEST-OK {MARKER} manifest/gmof/offsets/betas-clamp/"
          f"height/trajectory/fusion/outlier/keypoints/validator/tape "
          f"verified torch-free")

# ---------------- torch rendering (gate-derived, fov-parameterized) ----------------
# render_alpha / project_points / aligned_iou / init_cam_measured below
# are gate_fit.py (GATE-FIT-V8) functions copied and extended with a
# per-sequence fov parameter — gate's renderer hardcodes fov=38.
# gate_fit.py itself is imported for soft_iou / prep_mask / load_model /
# find_smplx and is never modified.

def render_alpha_fov(verts, faces, azim, device, size=RENDER_SIZE,
                     dist=RENDER_DIST, dtx=0.0, dty=0.0, scale=1.0,
                     fov=FOV_DEFAULT, sigma=SILHOUETTE_SIGMA):
    import torch
    from pytorch3d.renderer import (FoVPerspectiveCameras, MeshRasterizer,
        MeshRenderer, RasterizationSettings, SoftSilhouetteShader,
        BlendParams, look_at_view_transform)
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
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov))
    rast = RasterizationSettings(image_size=size, blur_radius=0.0,
                                 faces_per_pixel=1, bin_size=0)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=rast),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4)))
    out = renderer(mesh)
    return out[0, ..., 3]

def project_points_fov(verts, azim, device, size=RENDER_SIZE,
                       dist=RENDER_DIST, dtx=0.0, dty=0.0, scale=1.0,
                       fov=FOV_DEFAULT):
    import torch
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
    R, T = look_at_view_transform(dist=dist, elev=0.0, azim=azim, device=device)
    T = T + torch.tensor([[float(dtx), float(dty), 0.0]], device=device, dtype=T.dtype)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov))
    pts = (verts * float(scale)).reshape(1, -1, 3)
    return cam.transform_points_screen(pts, image_size=(size, size))[0]

def _wpp(fov, size):
    """World units per pixel at RENDER_DIST for a given fov."""
    return (2 * RENDER_DIST * math.tan(math.radians(float(fov) / 2.0))) / size

def aligned_iou_fov(verts, faces, mask, azim, device, size=256,
                    fov=FOV_DEFAULT):
    """Gate's aligned_iou with fov: iteratively align (scale + shift)
    by bbox matching; every candidate is MEASURED, so conventions
    cannot silently poison the score. Returns (best_iou, cam dict)."""
    import torch, numpy as np
    G = gate()
    tgt = torch.tensor(mask.astype(np.float32), device=device)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    mh = rows.max() - rows.min() + 1
    mcx = (cols.min() + cols.max()) / 2.0
    mcy = (rows.min() + rows.max()) / 2.0
    scale, dtx, dty = 1.0, 0.0, 0.0
    best = (-1.0, None)
    wpp = _wpp(fov, size)
    sx, sy = 1.0, -1.0
    prev_errx = prev_erry = None
    for _ in range(4):
        with torch.no_grad():
            alpha = render_alpha_fov(verts, faces, azim, device, size=size,
                                     dtx=dtx, dty=dty, scale=scale,
                                     fov=fov, sigma=SILHOUETTE_SIGMA)
        sil = (alpha > 0.5).cpu().numpy()
        v = float(G.soft_iou(alpha, tgt))
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
        dtx += sx * errx * wpp
        dty += sy * erry * wpp
        prev_errx, prev_erry = errx, erry
    return best

def init_cam_measured_fov(model, faces, view, device, rounds=20):
    """Gate's init_cam_measured with fov: monotone IoU-gated camera
    init against the real renderer. view needs mask512/azim/fov/name."""
    import torch, numpy as np
    G = gate()
    tgt = view["mask512"]
    fov = view.get("fov", FOV_DEFAULT)
    tys, txs = np.nonzero(tgt)
    th = float(tys.max() - tys.min() + 1)
    tc_r, tc_c = float(tys.mean()), float(txs.mean())
    with torch.no_grad():
        out = model(betas=torch.zeros(1, 100, device=device), return_verts=True)
        verts0 = out.vertices[0]
    wpp = _wpp(fov, RENDER_SIZE)
    def hard_iou(dtx, dty, scale):
        a = render_alpha_fov(verts0, faces, view["azim"], device,
                             dtx=dtx, dty=dty, scale=scale, fov=fov,
                             sigma=1e-5)
        sil = (a > 0.5).cpu().numpy()
        return G.iou(sil, tgt), sil
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
        cand = (dtx - (tc_c - rc_c) * wpp * 0.8,
                dty + (tc_r - rc_r) * wpp * 0.8,
                scale * (th / rh) ** 0.5)
        val, _ = hard_iou(*cand)
        if val >= best:
            dtx, dty, scale = cand
            best = val
        else:
            break
    print(f"INIT {view['name']}: dtx {dtx:.2f} dty {dty:.2f} "
          f"scale {scale:.2f} IoU {best:.3f} (fov {fov:.0f})", flush=True)
    return [dtx, dty, scale]

# ---------------- S0: ingest ----------------

def stage_segment_manifest(indir, maskdir, files):
    """Manifest-driven variant of gate's stage_segment: rembg ISNet,
    largest connected component, temp-write + PIL-verify + atomic
    replace. Verified-existing masks are reused (gate pattern)."""
    import numpy as np
    from PIL import Image
    os.makedirs(maskdir, exist_ok=True)
    session = None
    report = {}
    for name in files:
        stem = os.path.splitext(name)[0]
        dst = os.path.join(maskdir, stem + "_mask.png")
        if os.path.exists(dst):
            try:
                Image.open(dst).verify()
                report[name] = {"reused": True}
                print(f"SEGMENT-REUSED {name}", flush=True)
                continue
            except Exception:
                pass
        src = os.path.join(indir, name)
        if not os.path.exists(src):
            print(f"SEGMENT-MISSING {name}", flush=True)
            continue
        if session is None:
            from rembg import new_session
            session = new_session("isnet-general-use")
        from rembg import remove
        out = remove(Image.open(src).convert("RGB"), session=session)
        alpha = np.array(out.split()[-1])
        mask = alpha > 127
        try:
            from scipy import ndimage
            lab, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(mask, lab, range(1, n + 1))
                mask = lab == (int(np.argmax(sizes)) + 1)
        except ImportError:
            pass
        tmp = dst + ".tmp.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(tmp)
        Image.open(tmp).verify()
        os.replace(tmp, dst)
        t, b = gate().person_rows(mask)
        report[name] = {"mask_frac": float(mask.mean()),
                        "person_px_h": int(b - t)}
        print(f"SEGMENT {name} frac={mask.mean():.3f} person_h={b - t}px",
              flush=True)
    json.dump(report, open(os.path.join(maskdir, "segment_report.json"), "w"),
              indent=1)
    return report

def load_keypoints_for_view(kp_sidecar, view, img_shape):
    """Map a view's sidecar keypoints into 512 crop space.
    Returns an (K, 3) array [x512, y512, conf] over COCO_TO_SMPLX
    pairs, or None with a printed reason. Never raises on bad data."""
    import numpy as np
    if kp_sidecar is None:
        return None
    entry = kp_sidecar.get(view["file"])
    if entry is None:
        print(f"KEYPOINTS-MISSING {view['file']} (not in sidecar)", flush=True)
        return None
    try:
        kps = np.asarray(entry["keypoints"], dtype=np.float64)
        fmt = str(entry.get("format", "coco17"))
        if fmt not in ("coco17", "dwpose-body17"):
            print(f"KEYPOINTS-SKIP {view['file']}: unsupported format {fmt}",
                  flush=True)
            return None
        assert kps.ndim == 2 and kps.shape[0] >= 17 and kps.shape[1] >= 2
        xy = kps[:, :2]
        conf = kps[:, 2] if kps.shape[1] >= 3 else np.ones(len(kps))
        if img_shape is not None and entry.get("image_size"):
            iw, ih = entry["image_size"]
            if (ih, iw) != tuple(img_shape[:2]):
                print(f"KEYPOINTS-SKIP {view['file']}: sidecar image_size "
                      f"{entry['image_size']} != image {img_shape}", flush=True)
                return None
        pairs = COCO_TO_SMPLX
        sel_xy = np.stack([xy[c] for c, _ in pairs])
        sel_cf = np.array([conf[c] for c, _ in pairs])
        mapped = map_keypoints_to_crop(sel_xy, view["cropbox"])
        return np.concatenate([mapped, sel_cf[:, None]], axis=1)
    except Exception as e:
        print(f"KEYPOINTS-SKIP {view['file']}: {e}", flush=True)
        return None

def stage_ingest(indir, workdir, manifest, kp_sidecar):
    """S0: build the enriched view list. Hard-fails only when a hero
    image/mask is missing or > 20% of views are unusable."""
    import numpy as np
    from PIL import Image
    G = gate()
    maskdir = os.path.join(workdir, "masks")
    files = [v["file"] for v in manifest["views"]]
    stage_segment_manifest(indir, maskdir, files)
    base_az = trajectory_base_azimuths(manifest["views"])
    views, missing = [], []
    kp_found = 0
    for i, mv in enumerate(manifest["views"]):
        name = mv["file"]
        mp = os.path.join(maskdir, os.path.splitext(name)[0] + "_mask.png")
        ip = os.path.join(indir, name)
        if not os.path.exists(mp) or not os.path.exists(ip):
            missing.append(name)
            print(f"INGEST-MISSING {name} (hero={mv.get('hero_role')})",
                  flush=True)
            continue
        mask = np.array(Image.open(mp)) > 0
        sq, cropbox = G.prep_mask(mask)
        img = Image.open(ip)
        v = {"idx": len(views), "file": name, "name": name,
             "sequence": mv.get("sequence"),
             "frame_index": mv.get("frame_index"),
             "hero_role": mv.get("hero_role"),
             "azimuth_anchor_deg": mv.get("azimuth_anchor_deg"),
             "mask512": sq, "cropbox": cropbox,
             "img_size": img.size}
        az, src = base_az[name]
        v["azim"], v["azim_source"] = az, src
        v["keypoints"] = load_keypoints_for_view(kp_sidecar, v, mask.shape)
        if v["keypoints"] is not None:
            kp_found += 1
        views.append(v)
    heroes_missing = [mv["file"] for mv in manifest["views"]
                      if mv.get("hero_role") and mv["file"] in missing]
    if heroes_missing:
        print(f"STAGE1-FAIL ingest: hero views missing: {heroes_missing}")
        sys.exit(1)
    if len(missing) > 0.2 * len(files):
        print(f"STAGE1-FAIL ingest: {len(missing)}/{len(files)} views missing")
        sys.exit(1)
    print(f"INGEST-DONE views={len(views)} missing={len(missing)} "
          f"keypoints={kp_found}/{len(views)}", flush=True)
    if kp_sidecar is not None and kp_found == 0:
        print("KEYPOINTS-MISSING-ALL (keypoint term disabled everywhere)",
              flush=True)
    return views

# ---------------- S1: measured azimuths + per-sequence focals ----------------

def measure_azimuths(model, faces, views, device):
    """Fill 'measure' azimuths with the gate-proven grid search over
    the zero-beta body (256px), then enforce per-sequence smooth
    monotone trajectories with anchors held fixed."""
    import torch, numpy as np
    G = gate()
    with torch.no_grad():
        mean_verts = model(betas=torch.zeros(1, 100, device=device),
                           return_verts=True).vertices[0]
    for v in views:
        if v["azim"] is not None:
            continue
        m256 = v["mask512"][::2, ::2]
        best = (-1.0, None)
        for az in range(0, 360, 15):
            val, _ = aligned_iou_fov(mean_verts, faces, m256, float(az),
                                     device, size=256,
                                     fov=v.get("fov", FOV_DEFAULT))
            if val > best[0]:
                best = (val, float(az))
        v["azim"] = best[1]
        v["azim_source"] = "measured"
        print(f"TRAJ-MEASURED {v['name']}: azim {best[1]:.0f} "
              f"aligned-IoU {best[0]:.3f}", flush=True)
    from collections import defaultdict
    seqs = defaultdict(list)
    for v in views:
        if v.get("sequence"):
            seqs[v["sequence"]].append(v)
    for seq, vs in seqs.items():
        vs.sort(key=lambda x: x["frame_index"])
        azs = [v["azim"] for v in vs]
        anchored = [i for i, v in enumerate(vs)
                    if v.get("azimuth_anchor_deg") is not None]
        if anchored:
            i0 = anchored[0]
            proj = monotone_project_with_fixed(
                azs, i0, float(vs[i0]["azimuth_anchor_deg"]))
        else:
            proj = monotone_project(azs)
        for v, p in zip(vs, proj):
            if v.get("azimuth_anchor_deg") is not None:
                v["azim"] = float(v["azimuth_anchor_deg"])
            else:
                v["azim"] = float(p)
    for v in views:
        print(f"TRAJ {v['name']}: azim {v['azim']:.1f} "
              f"({v['azim_source']})", flush=True)

def measure_focals(model, faces, views, device):
    """One focal per sequence (stills: per file), chosen by measured
    aligned IoU over FOV_GRID on the sequence's first view, then FIXED
    for all gradient stages. Prints FOCAL lines."""
    import torch, numpy as np
    with torch.no_grad():
        mean_verts = model(betas=torch.zeros(1, 100, device=device),
                           return_verts=True).vertices[0]
    from collections import defaultdict
    groups = defaultdict(list)
    for v in views:
        groups[v.get("sequence") or v["file"]].append(v)
    for key, vs in groups.items():
        rep = vs[0]
        m256 = rep["mask512"][::2, ::2]
        scores = {}
        for fov in FOV_GRID:
            val, _ = aligned_iou_fov(mean_verts, faces, m256, rep["azim"],
                                     device, size=256, fov=fov)
            scores[fov] = val
        best_fov = max(scores, key=scores.get)
        for v in vs:
            v["fov"] = float(best_fov)
        print(f"FOCAL {key}: fov {best_fov:.0f} "
              f"(scores { {int(k): round(s, 3) for k, s in scores.items()} })",
              flush=True)

# ---------------- optional surface-evidence sidecars (S4) ----------------

def load_evidence_sidecars(workdir, views):
    """Normals / pointmap evidence dirs (per-view .npy in the ORIGINAL
    image space, mapped through the view's crop like keypoints).
    Absence is reported per type, never fatal."""
    import numpy as np
    status = {}
    for kind in ("normals", "pointmaps"):
        d = os.path.join(workdir, kind)
        found = 0
        if os.path.isdir(d):
            for v in views:
                p = os.path.join(d, os.path.splitext(v["file"])[0] + ".npy")
                if os.path.exists(p):
                    try:
                        arr = np.load(p)
                        sq = _map_evidence_to_crop(arr, v["cropbox"])
                        v[kind] = sq
                        found += 1
                    except Exception as e:
                        print(f"{kind.upper()}-SKIP {v['file']}: {e}",
                              flush=True)
        status[kind] = found
        if found == 0:
            print(f"{kind.upper()}-MISSING (no {kind} evidence; term off)",
                  flush=True)
        else:
            print(f"{kind.upper()}-FOUND {found}/{len(views)} views",
                  flush=True)
    return status

def _map_evidence_to_crop(arr, cropbox, size=RENDER_SIZE):
    """Crop an HxWx3 evidence map with the mask's cropbox and resize
    (nearest) to size x size — the same square the mask lives in."""
    import numpy as np
    from PIL import Image
    t, b, l, r = cropbox
    crop = arr[t:b + 1, l:r + 1]
    side = max(crop.shape[0], crop.shape[1])
    sq = np.zeros((side, side, crop.shape[2]), dtype=arr.dtype)
    sq[:crop.shape[0], :crop.shape[1]] = crop
    out = np.zeros((size, size, crop.shape[2]), dtype=np.float32)
    for ch in range(crop.shape[2]):
        im = Image.fromarray(sq[:, :, ch].astype(np.float32), mode="F")
        out[:, :, ch] = np.array(im.resize((size, size), Image.NEAREST))
    return out

def normals_term(verts, faces, view, device):
    """Sign-agnostic cosine agreement between the mesh's vertex
    normals (camera space) and the sidecar normal map, sampled at the
    vertices' projected pixels inside the silhouette. Convention-
    robust by construction (|cos|): normal-map z-sign conventions
    differ across predictors and are not worth a silent failure."""
    import torch, numpy as np
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import look_at_view_transform
    mesh = Meshes(verts=[verts], faces=[faces])
    vn = mesh.verts_normals_packed()
    R, T = look_at_view_transform(dist=RENDER_DIST, elev=0.0,
                                  azim=view["azim"], device=device)
    from pytorch3d.renderer import FoVPerspectiveCameras
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view["fov"])
    vn_cam = cam.get_world_to_view_transform().transform_normals(vn)
    cam_s = float(view["_cam"][2])  # verts arrive metric (g applied)
    scr = project_points_fov(verts, view["azim"], device,
                             dtx=float(view["_cam"][0]),
                             dty=float(view["_cam"][1]), scale=cam_s,
                             fov=view["fov"])
    tgt = torch.tensor(view["normals"], device=device)
    xs = scr[:, 0].round().long()
    ys = scr[:, 1].round().long()
    ok = (xs >= 0) & (xs < RENDER_SIZE) & (ys >= 0) & (ys < RENDER_SIZE)
    sil = torch.tensor(view["mask512"], device=device)
    ok = ok & (sil[ys.clamp(0, RENDER_SIZE - 1),
                   xs.clamp(0, RENDER_SIZE - 1)] > 0.5)
    if int(ok.sum()) < 50:
        return None
    tn = tgt[ys[ok], xs[ok]]
    tn = tn / tn.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos = (vn_cam[ok] * tn).sum(dim=1).abs().clamp(max=1.0)
    return float(ok.sum()), 1.0 - cos.mean()

def pointmap_term(verts, faces, view, device):
    """Point-to-pointmap agreement via projected correspondences:
    each vertex is projected to its pixel, the sidecar pointmap's 3D
    point at that pixel is its target, and the loss is the robust
    mean distance in CAMERA space after a per-view scale/shift
    alignment (closed form, detached) — the design's answer to
    monocular pointmaps carrying unknown per-view scale/shift.
    Differentiable: the camera transform of the model vertices
    carries gradients; the sampling indices do not need to."""
    import torch, numpy as np
    from pytorch3d.renderer import (FoVPerspectiveCameras,
        look_at_view_transform)
    R, T0 = look_at_view_transform(dist=RENDER_DIST, elev=0.0,
                                   azim=view["azim"], device=device)
    cam_s = float(view["_cam"][2])  # verts arrive metric (g applied)
    T = T0 + torch.tensor([[float(view["_cam"][0]), float(view["_cam"][1]),
                            0.0]], device=device, dtype=T0.dtype)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, fov=view["fov"])
    pts = (verts * cam_s).reshape(1, -1, 3)
    pts_cam = cam.transform_points(pts)[0]
    scr = cam.transform_points_screen(pts, image_size=(RENDER_SIZE,
                                                       RENDER_SIZE))[0]
    tgt = torch.tensor(view["pointmaps"], device=device)
    xs = scr[:, 0].round().long()
    ys = scr[:, 1].round().long()
    ok = (xs >= 0) & (xs < RENDER_SIZE) & (ys >= 0) & (ys < RENDER_SIZE)
    sil = torch.tensor(view["mask512"], device=device)
    ok = ok & (sil[ys.clamp(0, RENDER_SIZE - 1),
                   xs.clamp(0, RENDER_SIZE - 1)] > 0.5)
    ok = ok & (tgt[ys.clamp(0, RENDER_SIZE - 1),
                   xs.clamp(0, RENDER_SIZE - 1)].abs().sum(dim=1) > 0)
    if int(ok.sum()) < 50:
        return None
    M = pts_cam[ok]
    Tp = tgt[ys[ok], xs[ok]]
    with torch.no_grad():
        mm, mt = M.mean(dim=0), Tp.mean(dim=0)
        Mc, Tc = M - mm, Tp - mt
        a = float((Mc * Tc).sum() / (Mc * Mc).sum().clamp(min=1e-9))
        b = mt - a * mm
    resid = torch.linalg.norm(a * M + b - Tp, dim=1)
    return float(ok.sum()), torch.sqrt(resid ** 2 + 1e-6).mean()

# ---------------- S2/S3: joint fit ----------------

def keypoint_term(model_out_joints, view, device, g_scale):
    """GMoF keypoint term for one view. Returns (n_points, loss) or
    None when the view has no usable keypoints. Loss is the GMoF mean
    normalized by 100 so a ~10 px mean error contributes ~1.0 —
    comparable in scale to the silhouette term."""
    import torch, numpy as np
    kp = view.get("keypoints")
    if kp is None:
        return None
    joints = model_out_joints[0]
    sel = torch.tensor([s for _, s in COCO_TO_SMPLX], device=device)
    pts3d = joints[sel]
    cam_s = g_scale * float(view["_cam"][2])
    scr = project_points_fov(pts3d, view["azim"], device,
                             dtx=float(view["_cam"][0]),
                             dty=float(view["_cam"][1]), scale=cam_s,
                             fov=view["fov"])[:, :2]
    tgt = torch.tensor(kp[:, :2], device=device, dtype=scr.dtype)
    conf = torch.tensor(kp[:, 2], device=device, dtype=scr.dtype)
    trust = torch.tensor([KP_TRUST[s] for _, s in COCO_TO_SMPLX],
                         device=device, dtype=scr.dtype)
    w = conf * trust
    ok = w > 0.05
    if int(ok.sum()) < 4:
        return None
    r2 = ((scr[ok] - tgt[ok]) ** 2).sum(dim=1)
    rho2 = GMOF_RHO * GMOF_RHO
    per = rho2 * r2 / (rho2 + r2)
    loss = (per * w[ok]).sum() / w[ok].sum() / 100.0
    return int(ok.sum()), loss

def fit_joint(model, faces, views, device, steps_a=250, steps_b=150,
              anchor=None, height_m=HEIGHT_M_DEFAULT):
    """S3: stage A silhouette fit (gate-proven recipe) -> dead-view
    guard -> measured azimuth refinement -> global scale solve
    (height anchor, then FIXED) -> stage B polish with the GMoF
    keypoint term + anchor trust region. Returns the state dict."""
    import torch, numpy as np
    G = gate()
    n = len(views)
    for v in views:
        c0 = init_cam_measured_fov(model, faces, v, device)
        v["_cam0"] = c0
    betas = torch.zeros(1, 100, device=device, requires_grad=True)
    if anchor is not None:
        with torch.no_grad():
            betas.copy_(torch.tensor(anchor, device=device,
                                     dtype=betas.dtype).reshape(1, -1))
    def clamp_betas():
        with torch.no_grad():
            betas.clamp_(-BETA_CLAMP, BETA_CLAMP)
    clamp_betas()  # stage entry: anchor or zero betas start in bounds
    poses = []
    for v in views:
        p0 = torch.zeros(1, 63, device=device)
        if "init_pose" in v:
            pv = torch.tensor(v["init_pose"], device=device,
                              dtype=p0.dtype).reshape(-1)
            if len(pv) == 63:
                p0 = pv.reshape(1, 63).clone()
                print(f"INIT-POSE {v['name']}: body_pose from sidecar "
                      f"(body-space axis-angle; global_orient is "
                      f"camera-relative in regressor output and is "
                      f"NOT transferred)", flush=True)
        poses.append(p0.requires_grad_(True))
    orients = [torch.zeros(1, 3, device=device, requires_grad=True)
               for _ in range(n)]
    cams = [torch.tensor(v["_cam0"], device=device, requires_grad=True)
            for v in views]
    targets = [torch.tensor(v["mask512"].astype(np.float32), device=device)
               for v in views]
    anchor_t = None
    if anchor is not None:
        anchor_t = torch.tensor(anchor, device=device,
                                dtype=betas.dtype).reshape(1, -1)
    def build_opt(lr, include_b=True):
        ps = [poses, orients, cams]
        ps = [p for grp in ps for p in grp]
        if include_b:
            ps = [betas] + ps
        return torch.optim.Adam(ps, lr=lr)
    def clamp_cams():
        with torch.no_grad():
            for c in cams:
                c[0] = c[0].clamp(-1.0, 1.0)
                c[1] = c[1].clamp(-1.0, 1.0)
                c[2] = c[2].clamp(0.65, 1.6)
    def stage_loss(i, out, g_scale, use_kp):
        v = views[i]
        vv = dict(v)
        vv["_cam"] = cams[i]
        vv["_g"] = g_scale
        alpha = render_alpha_fov(out.vertices[0], faces, v["azim"], device,
                                 dtx=cams[i][0], dty=cams[i][1],
                                 scale=g_scale * cams[i][2], fov=v["fov"])
        l = (1 - G.soft_iou(alpha, targets[i]))
        l = l + 0.02 * (poses[i] ** 2).mean() + 0.001 * (betas ** 2).mean()
        if anchor_t is not None:
            l = l + 0.05 * ((betas - anchor_t) ** 2).mean()
        if use_kp:
            kt = keypoint_term(out.joints, vv, device, g_scale)
            if kt is not None:
                l = l + 1.0 * kt[1]
        return l
    opt = build_opt(0.02)
    for step in range(steps_a):
        opt.zero_grad()
        loss = 0.0
        for i in range(n):
            out = model(betas=betas, global_orient=orients[i],
                        body_pose=poses[i], return_verts=True)
            loss = loss + stage_loss(i, out, 1.0, False)
        loss.backward(); opt.step(); clamp_betas(); clamp_cams()
        if step % 50 == 0:
            print(f"  S3-A step {step} loss {float(loss):.4f}", flush=True)
    # dead-view guard (gate pattern): solo refit for collapsed views
    with torch.no_grad():
        for i, v in enumerate(views):
            out = model(betas=betas, global_orient=orients[i],
                        body_pose=poses[i], return_verts=True)
            a = render_alpha_fov(out.vertices[0], faces, v["azim"], device,
                                 dtx=cams[i][0], dty=cams[i][1],
                                 scale=cams[i][2], fov=v["fov"], sigma=1e-5)
            vi = G.iou((a > 0.5).cpu().numpy(), v["mask512"])
            print(f"  BETA-IOU {v['name']}: {vi:.3f}", flush=True)
            if vi < 0.10:
                print(f"  VIEW-DEAD {v['name']} — solo cam/pose refit",
                      flush=True)
                optd = torch.optim.Adam([poses[i], orients[i], cams[i]],
                                        lr=0.02)
                for _ in range(150):
                    optd.zero_grad()
                    o2 = model(betas=betas, global_orient=orients[i],
                               body_pose=poses[i], return_verts=True)
                    vv = dict(v); vv["_cam"] = cams[i]; vv["_g"] = 1.0
                    a2 = render_alpha_fov(o2.vertices[0], faces, v["azim"],
                                          device, dtx=cams[i][0],
                                          dty=cams[i][1], scale=cams[i][2],
                                          fov=v["fov"])
                    l2 = (1 - G.soft_iou(a2, targets[i])) \
                        + 0.02 * (poses[i] ** 2).mean()
                    l2.backward(); optd.step(); clamp_cams()
    refine_azimuths(model, faces, views, device,
                    {"betas": betas, "poses": poses, "orients": orients,
                     "cams": cams}, g_scale=1.0)
    # global scale: solve once from the neutral body height, then FIX
    with torch.no_grad():
        out = model(betas=betas, return_verts=True)
        h0 = mesh_height_np(out.vertices[0].cpu().numpy())
    g_scale = solve_height_scale(h0, height_m)
    print(f"HEIGHT-SCALE neutral height {h0:.4f} m (native units) -> "
          f"global scale {g_scale:.4f} for {height_m:.4f} m; FIXED",
          flush=True)
    with torch.no_grad():
        for c in cams:
            c[2] = (c[2] / g_scale).clamp(0.5, 2.0)
    def clamp_cams_b():
        with torch.no_grad():
            for c in cams:
                c[0] = c[0].clamp(-1.0, 1.0)
                c[1] = c[1].clamp(-1.0, 1.0)
                c[2] = c[2].clamp(0.5, 2.0)
    opt = build_opt(0.01)
    for step in range(steps_b):
        opt.zero_grad()
        loss = 0.0
        for i in range(n):
            out = model(betas=betas, global_orient=orients[i],
                        body_pose=poses[i], return_verts=True)
            loss = loss + stage_loss(i, out, g_scale, True)
        loss.backward(); opt.step(); clamp_betas(); clamp_cams_b()
        if step % 50 == 0:
            print(f"  S3-B step {step} loss {float(loss):.4f}", flush=True)
    state = {"betas": betas.detach(),
             "poses": [p.detach() for p in poses],
             "orients": [o.detach() for o in orients],
             "cams": [c.detach() for c in cams],
             "g_scale": g_scale}
    for i, v in enumerate(views):
        v["_cam"] = state["cams"][i]
        v["_g"] = g_scale
    return state

def refine_azimuths(model, faces, views, device, params, g_scale=1.0):
    """Measured azimuth refinement (no gradients through the camera):
    +/-12 deg grid by hard IoU with the current shape; anchored views
    locked; per-sequence monotone re-projection afterwards."""
    import torch, numpy as np
    G = gate()
    betas, poses, orients, cams = (params["betas"], params["poses"],
                                   params["orients"], params["cams"])
    with torch.no_grad():
        for i, v in enumerate(views):
            if v.get("azimuth_anchor_deg") is not None:
                print(f"  TRAJ-REFINE-SKIP {v['name']} (anchored)", flush=True)
                continue
            out = model(betas=betas, global_orient=orients[i],
                        body_pose=poses[i], return_verts=True)
            verts = out.vertices[0]
            base = v["azim"]
            best = (-1.0, base)
            for d in range(-12, 13, 3):
                a = render_alpha_fov(verts, faces, base + d, device,
                                     dtx=cams[i][0], dty=cams[i][1],
                                     scale=g_scale * cams[i][2],
                                     fov=v["fov"], sigma=1e-5)
                val = G.iou((a > 0.5).cpu().numpy(), v["mask512"])
                if val > best[0]:
                    best = (val, base + d)
            if best[1] != base:
                print(f"  TRAJ-REFINE {v['name']}: {base:.1f} -> "
                      f"{best[1]:.1f} (IoU {best[0]:.3f})", flush=True)
                v["azim"] = float(best[1])
    from collections import defaultdict
    seqs = defaultdict(list)
    for v in views:
        if v.get("sequence"):
            seqs[v["sequence"]].append(v)
    for seq, vs in seqs.items():
        vs.sort(key=lambda x: x["frame_index"])
        azs = [v["azim"] for v in vs]
        anchored = [i for i, v in enumerate(vs)
                    if v.get("azimuth_anchor_deg") is not None]
        if anchored:
            i0 = anchored[0]
            proj = monotone_project_with_fixed(
                azs, i0, float(vs[i0]["azimuth_anchor_deg"]))
        else:
            proj = monotone_project(azs)
        for v, p in zip(vs, proj):
            if v.get("azimuth_anchor_deg") is not None:
                v["azim"] = float(v["azimuth_anchor_deg"])
            else:
                v["azim"] = float(p)

# ---------------- S4: canonical scalar offsets ----------------

def compute_coverage(model, faces, state, views, device):
    """Per-vertex confidence from view coverage: for each kept view,
    a vertex earns cos-weighted credit when its posed surface normal
    faces the camera. Azimuth-only camera direction (the fitted
    global_orient is small); documented approximation."""
    import torch, numpy as np
    from pytorch3d.structures import Meshes
    acc = np.zeros(NUM_VERTS, dtype=np.float64)
    used = [v for v in views if not v.get("dropped")]
    with torch.no_grad():
        for v in used:
            i = v["idx"]
            out = model(betas=state["betas"],
                        global_orient=state["orients"][i],
                        body_pose=state["poses"][i], return_verts=True)
            verts = out.vertices[0]
            mesh = Meshes(verts=[verts], faces=[faces])
            vn = mesh.verts_normals_packed().cpu().numpy()
            vv = verts.cpu().numpy()
            az = math.radians(v["azim"])
            campos = np.array([RENDER_DIST * math.sin(az), 0.0,
                               RENDER_DIST * math.cos(az)])
            to_cam = campos - vv
            to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
            cos = np.einsum("ij,ij->i", vn, to_cam)
            acc += np.clip(cos - 0.15, 0.0, None)
    conf = np.clip(acc / max(1.0, 0.5 * len(used)), 0.0, 1.0)
    print(f"COVERAGE views={len(used)} conf mean {conf.mean():.3f} "
          f"zero {float((conf <= 0.05).mean()):.3f}", flush=True)
    return conf

def fit_offsets(model, faces, state, views, device, conf, steps=600):
    """S4: one canonical scalar field s (meters), applied along the
    posed surface normal each view; exported along neutral normals.
    Evidence: silhouette (low weight) + optional normals/pointmap
    terms. Regularizers: annealed Laplacian (gate pattern), magnitude
    weighted against low-confidence vertices; unobserved vertices are
    frozen at 0 and the field is hard-clamped to +/-OFFSET_CLAMP_M
    after every step."""
    import torch, numpy as np
    G = gate()
    from pytorch3d.loss import mesh_laplacian_smoothing
    from pytorch3d.structures import Meshes
    kept = [v for v in views if not v.get("dropped")]
    if not kept:
        print("STAGE1-FAIL S4: every view was dropped as an outlier")
        sys.exit(1)
    s = torch.zeros(1, NUM_VERTS, 1, device=device, requires_grad=True)
    poses = [p.clone().detach().requires_grad_(True) for p in state["poses"]]
    orients = [o.clone().detach().requires_grad_(True)
               for o in state["orients"]]
    cams = [c.clone().detach().requires_grad_(True) for c in state["cams"]]
    g = state["g_scale"]
    conf_t = torch.tensor(conf, device=device, dtype=torch.float32)
    frozen = conf_t <= 0.05
    mag_w = (1.5 - conf_t).reshape(1, NUM_VERTS, 1)
    targets = {v["idx"]: torch.tensor(v["mask512"].astype(np.float32),
                                      device=device) for v in kept}
    with torch.no_grad():
        base0 = model(betas=state["betas"], return_verts=True).vertices[0]
        n0 = vertex_normals_np(base0.cpu().numpy(),
                               faces.cpu().numpy())
    n0_t = torch.tensor(n0, device=device, dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [s], "lr": 0.005},
                            {"params": poses + orients, "lr": 0.003},
                            {"params": cams, "lr": 0.003}])
    for step in range(steps):
        opt.zero_grad()
        loss = 0.0
        first_verts = None
        for v in kept:
            i = v["idx"]
            out = model(betas=state["betas"], global_orient=orients[i],
                        body_pose=poses[i], return_verts=True)
            posed = out.vertices * g
            mesh = Meshes(verts=[posed[0]], faces=[faces])
            vn = mesh.verts_normals_packed().detach()
            verts = posed + s[0, :, 0].reshape(-1, 1) * vn.reshape(-1, 3)
            if first_verts is None:
                first_verts = (base0 * g
                               + s[0, :, 0].reshape(-1, 1) * n0_t)
            alpha = render_alpha_fov(verts[0], faces, v["azim"], device,
                                     dtx=cams[i][0], dty=cams[i][1],
                                     scale=cams[i][2], fov=v["fov"])
            loss = loss + 0.5 * (1 - G.soft_iou(alpha, targets[i]))
            vv = dict(v); vv["_cam"] = cams[i]; vv["_g"] = g
            if "normals" in v:
                nt = normals_term(verts[0], faces, vv, device)
                if nt is not None:
                    loss = loss + 1.0 * nt[1]
            if "pointmaps" in v:
                pt = pointmap_term(verts[0], faces, vv, device)
                if pt is not None:
                    loss = loss + 1.0 * pt[1]
        mesh0 = Meshes(verts=[first_verts], faces=[faces])
        lap_w = 2.0 if step < int(steps * 0.6) else 0.5
        loss = loss + lap_w * mesh_laplacian_smoothing(mesh0) \
            + 0.5 * (mag_w * s ** 2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            s.clamp_(-OFFSET_CLAMP_M, OFFSET_CLAMP_M)
            s[0, frozen, 0] = 0.0
            for c in cams:
                c[0] = c[0].clamp(-1.0, 1.0)
                c[1] = c[1].clamp(-1.0, 1.0)
                c[2] = c[2].clamp(0.5, 2.0)
        if step % 100 == 0:
            sa = s.detach().abs()
            print(f"  S4 step {step} loss {float(loss):.4f} "
                  f"|s| mean {float(sa.mean()):.4f} max {float(sa.max()):.4f}",
                  flush=True)
    state = dict(state)
    state["scalar_offsets"] = s.detach()
    state["poses"] = [p.detach() for p in poses]
    state["orients"] = [o.detach() for o in orients]
    state["cams"] = [c.detach() for c in cams]
    for v in views:
        v["_cam"] = state["cams"][v["idx"]]
    return state

# ---------------- per-view metrics (stage-1 variant of gate's) ----------------

def view_metrics_s1(model, faces, state, view, device):
    """Full + below-neck IoU for one view under the fitted state
    (metric scale + scalar offsets applied along posed normals)."""
    import torch, numpy as np
    from pytorch3d.structures import Meshes
    G = gate()
    i = view["idx"]
    g = state["g_scale"]
    cam = state["cams"][i]
    with torch.no_grad():
        out = model(betas=state["betas"],
                    global_orient=state["orients"][i],
                    body_pose=state["poses"][i], return_verts=True)
        posed = out.vertices * g
        if "scalar_offsets" in state:
            mesh = Meshes(verts=[posed[0]], faces=[faces])
            vn = mesh.verts_normals_packed()
            s = state["scalar_offsets"][0, :, 0].reshape(-1, 1)
            posed = posed + s * vn.reshape(1, -1, 3)
        alpha = render_alpha_fov(posed[0], faces, view["azim"], device,
                                 dtx=float(cam[0]), dty=float(cam[1]),
                                 scale=float(cam[2]), fov=view["fov"],
                                 sigma=1e-5)
        sil = (alpha > 0.5).cpu().numpy()
        neck = out.joints[0][12]
        scr = project_points_fov(neck.reshape(1, 3), view["azim"], device,
                                 dtx=float(cam[0]), dty=float(cam[1]),
                                 scale=g * float(cam[2]), fov=view["fov"])
        neck_row = int(round(float(scr[0, 1])))
    tgt = view["mask512"]
    sil_bn = sil.copy(); sil_bn[:neck_row, :] = False
    tgt_bn = tgt.copy(); tgt_bn[:neck_row, :] = False
    return {"iou_full": G.iou(sil, tgt), "iou_primary": G.iou(sil_bn, tgt_bn),
            "neck_row": neck_row, "silhouette": sil}

# ---------------- S5: export + validation ----------------

def stage_export(model, faces, state, views, workdir, conf,
                 validator_path=None):
    """Build the neutral metric mesh, write scaffold.npz + PLY, run
    the geometry validator IN-PROCESS, compute tape measurements and
    per-view metrics, write stage1_metrics.json. Returns
    (verdict_dict, metrics_dict). A FAIL verdict is data here — the
    caller decides the exit; nothing prints success in this stage."""
    import torch, numpy as np
    g = state["g_scale"]
    with torch.no_grad():
        base = model(betas=state["betas"], return_verts=True).vertices[0]
    base_np = base.cpu().numpy() * g
    faces_np = faces.cpu().numpy()
    nrm = vertex_normals_np(base_np, faces_np)
    s_np = state["scalar_offsets"][0, :, 0].cpu().numpy()
    assert np.abs(s_np).max() <= OFFSET_CLAMP_M + 1e-6, \
        f"offset clamp violated at export: {np.abs(s_np).max():.4f}"
    verts = apply_scalar_offsets(base_np, nrm, s_np)
    npz_path = os.path.join(workdir, "scaffold.npz")
    np.savez(npz_path,
             betas=state["betas"].cpu().numpy(),
             scalar_offsets=s_np.reshape(1, NUM_VERTS, 1),
             confidence=conf, global_scale=np.float64(g),
             height_m=np.float64(HEIGHT_M_DEFAULT))
    ply_path = os.path.join(workdir, "scaffold_body.ply")
    write_ply(ply_path, verts.astype(np.float32), faces_np,
              vertex_normals_np(verts, faces_np).astype(np.float32))
    print(f"EXPORT-DONE {ply_path} verts {len(verts)} faces {len(faces_np)}",
          flush=True)
    validate, vsrc = load_validator(validator_path)
    print(f"VALIDATOR-SOURCE {vsrc}", flush=True)
    vres = validate(ply_path)
    print(f"VALIDATOR-VERDICT {vres['verdict']} {vres.get('fails')}",
          flush=True)
    tape = tape_measurements(verts, faces_np)
    print(f"TAPE {json.dumps(tape)}", flush=True)
    per_view = []
    for v in views:
        m = view_metrics_s1(model, faces, state, v, device_current())
        per_view.append({"file": v["file"],
                         "iou_full": m["iou_full"],
                         "iou_primary": m["iou_primary"],
                         "residual": 1.0 - m["iou_primary"],
                         "dropped": bool(v.get("dropped", False)),
                         "azim": v["azim"], "fov": v["fov"]})
        print(f"METRIC {v['file']}: IoU below-neck {m['iou_primary']:.3f} "
              f"full {m['iou_full']:.3f}", flush=True)
    metrics = {"marker": MARKER, "height_m": HEIGHT_M_DEFAULT,
               "global_scale": g, "views": per_view,
               "dropped_views": [v["file"] for v in views
                                 if v.get("dropped")],
               "validator": vres, "validator_source": vsrc,
               "tape": tape,
               "offset_stats": {"max_abs_m": float(np.abs(s_np).max()),
                                "mean_abs_m": float(np.abs(s_np).mean())},
               "confidence": {"mean": float(conf.mean()),
                              "frac_zero": float((conf <= 0.05).mean())}}
    json.dump(metrics, open(os.path.join(workdir, "stage1_metrics.json"),
                            "w"), indent=1)
    return vres, metrics

_DEVICE = None
def device_current():
    return _DEVICE

# ---------------- S5b: leave-one-profile-out ----------------

def stage_lovo(model, faces, views, workdir, device, height_m, steps=150):
    """Hold out the side hero (the profile the gate could not
    predict), refit S3 briefly without it, then fit ONLY its
    pose/camera against the held-out shape and score the prediction —
    the bust-depth transfer test. Prints LOVO lines; result saved."""
    import torch, numpy as np
    side = [v for v in views if v.get("hero_role") == "side"]
    if not side:
        print("LOVO-SKIPPED (no side hero in the manifest set)", flush=True)
        return None
    sv = side[0]
    rest = [dict(v) for v in views if v["file"] != sv["file"]]
    for i, v in enumerate(rest):
        v["idx"] = i
    print(f"LOVO hold out {sv['file']} (profile)", flush=True)
    state = fit_joint(model, faces, rest, device, steps_a=steps,
                      steps_b=max(50, steps // 2), anchor=None,
                      height_m=height_m)
    import torch as _t
    pose = _t.zeros(1, 63, device=device, requires_grad=True)
    orient = _t.zeros(1, 3, device=device, requires_grad=True)
    cam0 = init_cam_measured_fov(model, faces, {**sv, "azim": sv["azim"]},
                                 device)
    cam = _t.tensor(cam0, device=device, requires_grad=True)
    with _t.no_grad():
        cam[2] = cam[2] / state["g_scale"]
    opt = _t.optim.Adam([pose, orient, cam], lr=0.02)
    tgt = _t.tensor(sv["mask512"].astype(np.float32), device=device)
    G = gate()
    for _ in range(200):
        opt.zero_grad()
        out = model(betas=state["betas"], global_orient=orient,
                    body_pose=pose, return_verts=True)
        alpha = render_alpha_fov(out.vertices[0], faces, sv["azim"], device,
                                 dtx=cam[0], dty=cam[1],
                                 scale=state["g_scale"] * cam[2],
                                 fov=sv["fov"])
        loss = (1 - G.soft_iou(alpha, tgt)) + 0.02 * (pose ** 2).mean()
        loss.backward(); opt.step()
    st2 = {"betas": state["betas"], "poses": [pose.detach()],
           "orients": [orient.detach()], "cams": [cam.detach()],
           "g_scale": state["g_scale"]}
    svv = dict(sv); svv["idx"] = 0
    m = view_metrics_s1(model, faces, st2, svv, device)
    bd = gate().bust_delta(m["silhouette"], sv["mask512"])
    res = {"held_out": sv["file"], "iou_primary": m["iou_primary"],
           "iou_full": m["iou_full"], "bust": bd}
    print(f"LOVO-RESULT {json.dumps(res)}", flush=True)
    json.dump(res, open(os.path.join(workdir, "lovo_result.json"), "w"),
              indent=1)
    return res

# ---------------- run orchestration ----------------

def print_guards():
    print(f"{MARKER} guards: manifest=on gmof(rho={GMOF_RHO:.0f})=on "
          f"trust-region=on height-anchor={HEIGHT_M_DEFAULT}m "
          f"traj-monotone=on outlier(>{OUTLIER_SIGMA}sigma)=on "
          f"offset=canonical-scalar clamp={OFFSET_CLAMP_M}m "
          f"beta-clamp={BETA_CLAMP} silhouette-sigma={SILHOUETTE_SIGMA:g} "
          f"validator=on", flush=True)

def stage_run(args):
    import torch, numpy as np
    global _DEVICE
    print_guards()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device
    print(f"device {device}", flush=True)
    manifest, mpath = load_manifest(args.manifest)
    print(f"MANIFEST {mpath} views={manifest['view_count']} "
          f"height={manifest['subject_height_m']}", flush=True)
    height_m = float(manifest.get("subject_height_m", HEIGHT_M_DEFAULT))
    workdir = args.work
    os.makedirs(workdir, exist_ok=True)
    G = gate()
    model_dir = G.find_smplx()
    if not model_dir:
        print("STAGE1-FAIL preflight: SMPLX_FEMALE.npz not found")
        sys.exit(1)
    print(f"smplx model dir: {model_dir}", flush=True)
    kp_path = args.keypoints or _first_existing(
        [os.path.join(workdir, "keypoints.json"),
         os.path.join(args.input, "keypoints.json")])
    kp_sidecar = load_sidecar(kp_path, "KEYPOINTS")
    views = stage_ingest(args.input, workdir, manifest, kp_sidecar)
    model = G.load_model(model_dir, device)
    faces = model.faces_tensor.to(device)
    measure_focals(model, faces, views, device)
    measure_azimuths(model, faces, views, device)
    # S2: regressor init sidecar -> beta anchor (never faked)
    init_path = args.init or _first_existing(
        [os.path.join(workdir, "init.json"),
         os.path.join(args.input, "init.json")])
    init_sidecar = load_sidecar(init_path, "INIT")
    anchor = None
    if init_sidecar:
        blist, wlist = [], []
        for v in views:
            e = init_sidecar.get(v["file"])
            if e and "betas" in e:
                b = np.asarray(e["betas"], dtype=np.float64)
                if len(b) < 100:
                    b = np.concatenate([b, np.zeros(100 - len(b))])
                blist.append(b[:100])
                wlist.append(float(e.get("weight", 1.0)))
                if "body_pose" in e:
                    v["init_pose"] = np.asarray(e["body_pose"],
                                                 dtype=np.float64)
                print(f"INIT {v['file']}: betas from "
                      f"{e.get('source', 'sidecar')}", flush=True)
        if blist:
            anchor, spread = fuse_betas(blist, wlist)
            print(f"INIT-ANCHOR fused {len(blist)} views, "
                  f"beta spread {spread:.3f}", flush=True)
    if anchor is None:
        print("INIT-FALLBACK no usable init sidecar: measured-IoU "
              "cameras + zero betas (gate-proven path); the pod "
              "preflight tests the real regressors separately",
              flush=True)
    print("S3 joint fit", flush=True)
    state = fit_joint(model, faces, views, device,
                      steps_a=args.steps_beta, steps_b=args.steps_beta // 2,
                      anchor=anchor, height_m=height_m)
    if any("init_pose" in v for v in views):
        print("INIT-POSE-NOTE sidecar body_pose values initialized "
              "the S3 poses (body-space, camera-independent); "
              "global_orient was not transferred (camera-relative)",
              flush=True)
    # outlier-view rejection on S3 residuals
    resid = []
    for v in views:
        m = view_metrics_s1(model, faces, state, v, device)
        v["s3_iou_primary"] = m["iou_primary"]
        resid.append(1.0 - m["iou_primary"])
    keep, dropped, med, sig = outlier_drop(np.array(resid))
    for i in dropped:
        views[i]["dropped"] = True
        print(f"OUTLIER-DROPPED {views[i]['file']} "
              f"residual {resid[i]:.3f} (median {med:.3f}, sigma {sig:.3f})",
              flush=True)
    print(f"OUTLIER-SUMMARY dropped={len(dropped)} kept={int(keep.sum())}",
          flush=True)
    ev_status = load_evidence_sidecars(workdir, views)
    conf = compute_coverage(model, faces, state, views, device)
    print("S4 canonical scalar offsets", flush=True)
    state = fit_offsets(model, faces, state, views, device, conf,
                        steps=args.steps_off)
    if args.lovo_steps > 0:
        stage_lovo(model, faces, views, workdir, device, height_m,
                   steps=args.lovo_steps)
    vres, metrics = stage_export(model, faces, state, views, workdir,
                                 conf, validator_path=args.validator)
    metrics["sidecars"] = {"keypoints": kp_path if kp_sidecar else None,
                           "init": init_path if init_sidecar else None,
                           "evidence": ev_status}
    metrics["outlier"] = {"median_residual": med, "sigma": sig,
                          "dropped": [views[i]["file"] for i in dropped]}
    json.dump(metrics, open(os.path.join(workdir, "stage1_metrics.json"),
                            "w"), indent=1)
    if vres["verdict"] != "PASS":
        print(f"STAGE1-VALIDATION-FAIL {vres.get('fails')}", flush=True)
        sys.exit(2)
    print("STAGE1-VALIDATION-PASS", flush=True)
    print("STAGE1-DONE", flush=True)

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else None

# ---------------- S5a: synthetic recovery test (pod, GPU) ----------------

def stage_synth(args):
    """Render a KNOWN body (seeded betas + a known smooth scalar
    offset field) through a hero-like view distribution, run the
    S2-S4 machinery on the synthetic masks, and report recovery
    errors in mm. This is the design's mandatory self-test gate —
    run it before the real set is trusted."""
    import torch, numpy as np
    global _DEVICE
    print_guards()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _DEVICE = device
    G = gate()
    model_dir = G.find_smplx()
    if not model_dir:
        print("STAGE1-FAIL preflight: SMPLX_FEMALE.npz not found")
        sys.exit(1)
    model = G.load_model(model_dir, device)
    faces = model.faces_tensor.to(device)
    rng = np.random.RandomState(7)
    # Decaying beta spectrum, like real SMPL-X fits: shape variance
    # concentrates in the leading components. MEASURED on the V4
    # synth's saved PLYs: the previous flat spectrum
    # (randn(100)*0.7, clip +-1.5) produced a NON-PERSON — chest
    # width 1.217 m, chest circumference 2.487 m against waist
    # 0.656 m at 1.626 m height — so the printed chest recovery
    # error (117.72 cm) was the target's own defect, not the
    # fitter's. std_j = 0.8 * 0.82**j for j < 30, 0.03 tail.
    beta_std = np.array([0.8 * (0.82 ** j) if j < 30 else 0.03
                         for j in range(100)])
    betas_true = np.clip(rng.randn(100) * beta_std, -2.0, 2.0)
    with torch.no_grad():
        out_true = model(betas=torch.tensor(betas_true, device=device,
                                            dtype=torch.float32).reshape(1, -1),
                         return_verts=True)
        base = out_true.vertices[0]
        joints_true = out_true.joints[0].cpu().numpy()
    base_np = base.cpu().numpy()
    nrm_np = vertex_normals_np(base_np, faces.cpu().numpy())
    s_true = 0.035 * np.sin(3.0 * base_np[:, 1]) * np.cos(2.0 * base_np[:, 0])
    s_true = clamp_offsets_np(s_true)
    verts_true = apply_scalar_offsets(base_np, nrm_np, s_true)
    print(f"SYNTH-TARGET ||betas_true|| "
          f"{float(np.linalg.norm(betas_true)):.3f} | native height "
          f"{mesh_height_np(verts_true):.4f} m (offset-applied, "
          f"pre-anchor)", flush=True)
    azims = [0.0, 45.0, 90.0, 105.0, 135.0, 180.0, 215.0, 270.0]
    views = []
    vt = torch.tensor(verts_true, device=device, dtype=torch.float32)
    for i, az in enumerate(azims):
        with torch.no_grad():
            a = render_alpha_fov(vt, faces, az, device, sigma=1e-5)
        views.append({"idx": i, "file": f"synth_{int(az):03d}.png",
                      "name": f"synth_{int(az):03d}", "sequence": None,
                      "frame_index": None, "hero_role": None,
                      "azimuth_anchor_deg": az, "azim": float(az),
                      "azim_source": "anchor", "fov": FOV_DEFAULT,
                      "mask512": (a > 0.5).cpu().numpy(),
                      "keypoints": None})
        print(f"SYNTH-VIEW azim {az:.0f} mask ready", flush=True)
    # DWPose-like keypoints from the TRUE body, in the exact format
    # keypoint_term consumes: (K,3) [x512, y512, conf] over
    # COCO_TO_SMPLX pairs (what load_keypoints_for_view returns).
    # The synth masks were rendered with the identity cam
    # (dtx=dty=0, scale=1.0, fov=FOV_DEFAULT), so the true joints are
    # projected with that same camera; seeded 2.0px gaussian noise
    # keeps it deterministic. _cam/_g/fov/azim are set by fit_joint
    # at fit time exactly as in the real ingest path.
    smplx_idx = [sd for _, sd in COCO_TO_SMPLX]
    kp_rng = np.random.RandomState(1234)
    kp_conf = np.array([0.45 if sd in (20, 21, 7, 8) else 0.9
                        for _, sd in COCO_TO_SMPLX])
    for v in views:
        with torch.no_grad():
            scr = project_points_fov(
                torch.tensor(joints_true[smplx_idx], device=device,
                             dtype=torch.float32),
                v["azim"], device, dtx=0.0, dty=0.0, scale=1.0,
                fov=FOV_DEFAULT)
        xy = scr[:, :2].cpu().numpy()
        xy = xy + kp_rng.randn(len(smplx_idx), 2) * 2.0
        v["keypoints"] = np.concatenate([xy, kp_conf[:, None]], axis=1)
    print(f"SYNTH-KEYPOINTS {len(views)} views: true-joint projections + "
          f"2.0px seeded noise, conf 0.9 (0.45 wrists/ankles)",
          flush=True)
    state = fit_joint(model, faces, views, device, steps_a=args.synth_steps,
                      steps_b=args.synth_steps // 2, anchor=None,
                      height_m=HEIGHT_M_DEFAULT)
    conf = compute_coverage(model, faces, state, views, device)
    state = fit_offsets(model, faces, state, views, device, conf,
                        steps=args.synth_steps)
    beta_errs = np.abs(state["betas"].cpu().numpy().reshape(-1)
                       - betas_true)
    b_err = float(beta_errs.max())
    b_mean30 = float(beta_errs[:30].mean())
    s_hat = state["scalar_offsets"][0, :, 0].cpu().numpy()
    # recovery is judged at the SAME metric height: the fit anchors
    # height to 1.6256 m, so scale the known body to that height too
    k_true = HEIGHT_M_DEFAULT / mesh_height_np(verts_true)
    verts_true = verts_true * k_true
    s_rmse_mm = float(np.sqrt(np.mean((s_hat - s_true * k_true) ** 2))
                      * 1000.0)
    with torch.no_grad():
        base_hat = model(betas=state["betas"],
                         return_verts=True).vertices[0]
    bh = base_hat.cpu().numpy() * state["g_scale"]
    nh = vertex_normals_np(bh, faces.cpu().numpy())
    verts_hat = apply_scalar_offsets(bh, nh, s_hat)
    tape_t = tape_measurements(verts_true, faces.cpu().numpy())
    tape_h = tape_measurements(verts_hat, faces.cpu().numpy())
    chest_err = None
    if tape_t["chest"] and tape_h["chest"]:
        chest_err = abs(tape_t["chest"]["circumference_m"]
                        - tape_h["chest"]["circumference_m"]) * 100.0
    chest_s = "None" if chest_err is None else f"{chest_err:.2f}"
    surf_d = np.linalg.norm(verts_hat - verts_true, axis=1) * 1000.0
    surf_mean = float(surf_d.mean())
    surf_rmse = float(np.sqrt((surf_d ** 2).mean()))
    print(f"SYNTH-RECOVERY beta max|err| {b_err:.3f} | "
          f"beta mean|err|30 {b_mean30:.3f} | offset RMSE "
          f"{s_rmse_mm:.1f} mm | chest circumference err {chest_s} cm | "
          f"surface mean {surf_mean:.1f} mm | surface RMSE "
          f"{surf_rmse:.1f} mm",
          flush=True)
    os.makedirs(args.work, exist_ok=True)
    faces_np = faces.cpu().numpy()
    true_ply = os.path.join(args.work, "synth_true.ply")
    write_ply(true_ply, verts_true.astype(np.float32), faces_np,
              vertex_normals_np(verts_true, faces_np).astype(np.float32))
    rec_ply = os.path.join(args.work, "synth_recovered.ply")
    write_ply(rec_ply, verts_hat.astype(np.float32), faces_np,
              vertex_normals_np(verts_hat, faces_np).astype(np.float32))
    print(f"SYNTH-PLY true={true_ply} recovered={rec_ply}", flush=True)
    print("SYNTH-DONE", flush=True)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run", "synth"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--input", default="/workspace/ComfyUI/input/stage1_views")
    ap.add_argument("--work", default="/workspace/stage1")
    ap.add_argument("--validator", default=None)
    ap.add_argument("--keypoints", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--steps-beta", type=int, default=250)
    ap.add_argument("--steps-off", type=int, default=600)
    ap.add_argument("--lovo-steps", type=int, default=150)
    ap.add_argument("--synth-steps", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "synth":
        stage_synth(a)
    else:
        stage_run(a)

if __name__ == "__main__":
    main()
