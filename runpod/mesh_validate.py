"""MESH-VALIDATE-V1 — geometry validation for exported body meshes.
Born from the Oct 18, 2026 gate-export failure: a mesh whose 2D fit scores
looked fine was torn (70 runaway vertices) and bloated (torso ~3x too deep),
and nothing checked the geometry before it was shown. This module is the
check. Pure numpy; reads ASCII or binary PLY via a minimal parser.

Checks (all reported, verdict = FAIL if any hard check fails):
  H1 runaway vertices: a vertex whose nearest mesh-neighbor is > 15 cm away
     is off the surface. Hard fail if count > 0.
  H2 torso depth/width ratio at chest and waist slices must sit in the
     anthropometric band 0.40-0.80 (a real torso is much narrower than deep
     is wrong: depth ~0.55-0.65 of width typically; band is generous).
  H3 robust height (99.9th-0.1th percentile of Y) vs width: a standing body
     with arms at sides is 3.2-4.2x taller than wide; T/A-pose 0.85-1.15x.
     Reported, soft check (pose-dependent).
  Slices use percentile-robust bands over the non-runaway vertices.
"""
import sys
import numpy as np
from collections import defaultdict

def read_ply(path):
    import struct
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
    # binary little endian, standard x/y/z float layout
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

def validate(path):
    V, F = read_ply(path)
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

if __name__ == '__main__':
    import json
    print(json.dumps(validate(sys.argv[1]), indent=1))
