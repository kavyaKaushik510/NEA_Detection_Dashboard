# PPA-3 v2 (simple) — with 3 minimal changes:
# 1) d_min_px lowered to 0.8 px (seed small-motion)
# 2) prediction radius gets a small path-length floor: max(..., 0.04*(d12+d23))
# 3) greedy deconflict tie-breaker: sort by (J asc, total_disp desc)
#
# Everything else unchanged.

from __future__ import annotations
from dataclasses import dataclass
from math import atan2, hypot, log, pi, isfinite
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_KDTREE = True
except Exception:
    KDTree = None
    _HAVE_KDTREE = False

def _wrap_angle_rad(dth: float) -> float:
    return (dth + pi) % (2 * pi) - pi

@dataclass
class LinkParamsV2:
    # Displacement seed window (px)
    d_min_px: float = 0.8       # was 5.0 — allow slow/short-cadence movers
    d_max_px: float = 150.0
    # Prediction radius floor (px) + k-sigma multiplier
    r_pred_min_px: float = 8.0
    k_sigma_pred: float = 2.5
    # Per-folder alignment sigma (px) added in quadrature
    sigma_align_px: float = 1.0  # could be set from Gaia residuals if available
    # Score weights (physics-first; same as before)
    w_resid: float = 0.55
    w_cross: float = 0.25
    w_angle: float = 0.15
    w_vratio: float = 0.05
    # Optional soft constraints
    vel_ratio_min: float = 0.6
    vel_ratio_max: float = 1.5
    angle_soft_scale_deg: float = 15.0
    # Small-displacement micro-mode trigger (sum of segment lengths, px)
    small_disp_total_px: float = 9.0
    # Keep angle as a soft prior only
    no_hard_angle_gate: bool = True
    # Enable seed-level best p3 selection
    choose_best_p3_per_seed: bool = True

class PPA3LinkerV2:
    def __init__(self, params: Optional[LinkParamsV2] = None):
        self.p = params or LinkParamsV2()

    def _coerce(self, C: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if C is None:
            return out
        if isinstance(C, list) and (len(C) == 0 or isinstance(C[0], dict)):
            for i, c in enumerate(C):
                try:
                    x = float(c.get('x', c.get('X', c.get('col'))))
                    y = float(c.get('y', c.get('Y', c.get('row'))))
                except Exception:
                    continue
                if not (isfinite(x) and isfinite(y)):
                    continue
                sigma = float(c.get('sigma', c.get('sig', 1.0)))
                cid = c.get('id', i)
                keep = dict(c)
                keep.update({'x': x, 'y': y, 'sigma': sigma, 'id': cid})
                out.append(keep)
            return out
        # numpy-like
        try:
            import numpy as np
            A = np.asarray(C)
            if A.ndim == 2 and A.shape[1] >= 2:
                for i, (x, y) in enumerate(A[:, :2]):
                    if isfinite(float(x)) and isfinite(float(y)):
                        out.append({'x': float(x), 'y': float(y), 'sigma': 1.0, 'id': i})
                return out
        except Exception:
            pass
        return out

    def link_three_frames(
        self,
        C1: Sequence[Dict[str, Any]] | Any,
        C2: Sequence[Dict[str, Any]] | Any,
        C3: Sequence[Dict[str, Any]] | Any,
        t1: float, t2: float, t3: float,
    ) -> List[Dict[str, Any]]:
        """Return disjoint triplets scored by physics-first geometric likelihood."""
        P = self.p
        C1 = self._coerce(C1); C2 = self._coerce(C2); C3 = self._coerce(C3)
        dt12 = float(t2 - t1); dt23 = float(t3 - t2)
        if not (dt12 > 0 and dt23 > 0):
            raise ValueError("Times must satisfy t1 < t2 < t3 (seconds).")

        # KD structures
        if _HAVE_KDTREE and len(C2) and len(C3):
            import numpy as np
            xy2 = np.array([(c['x'], c['y']) for c in C2], float)
            xy3 = np.array([(c['x'], c['y']) for c in C3], float)
            k2 = KDTree(xy2); k3 = KDTree(xy3)
        else:
            k2 = None; k3 = None

        seed_r = P.d_max_px
        triplets: List[Tuple[float, int, int, int, Dict[str,float]]] = []  # (J, i1,i2,i3,diag)

        # Iterate frame-1 points
        for i1, p1 in enumerate(C1):
            x1, y1, s1 = p1['x'], p1['y'], float(p1.get('sigma', 1.0))
            # neighbors in F2
            if k2 is not None:
                idx2_list = k2.query_ball_point([x1, y1], r=seed_r)
            else:
                idx2_list = range(len(C2))

            for i2 in idx2_list:
                p2 = C2[i2]
                x2, y2, s2 = p2['x'], p2['y'], float(p2.get('sigma', 1.0))
                dx12, dy12 = x2 - x1, y2 - y1
                d12 = hypot(dx12, dy12)
                if d12 < P.d_min_px or d12 > P.d_max_px:
                    continue

                # predict to F3
                vx, vy = dx12/dt12, dy12/dt12
                x3p, y3p = x2 + vx*dt23, y2 + vy*dt23

                # prediction uncertainty
                sig1 = (s1**2 + P.sigma_align_px**2)**0.5
                sig2 = (s2**2 + P.sigma_align_px**2)**0.5
                # assume unknown sigma3 beyond alignment
                sig3 = P.sigma_align_px
                sig_pred = ((sig1*(dt23/dt12))**2 +
                            (sig2*(1.0 + dt23/dt12))**2 +
                            (sig3)**2 )**0.5
                # CHANGE #2: path-length floor in prediction radius
                r_pred = max(P.r_pred_min_px, P.k_sigma_pred*sig_pred, 0.04*(d12 + hypot(x3p - x2, y3p - y2)))

                # candidates near prediction
                if k3 is not None:
                    idx3_list = k3.query_ball_point([x3p, y3p], r=r_pred)
                else:
                    idx3_list = [j for j, p3 in enumerate(C3)
                                 if hypot(p3['x']-x3p, p3['y']-y3p) <= r_pred]

                if not idx3_list:
                    continue

                # common geometric terms
                ux = dx12 / d12
                uy = dy12 / d12
                th12 = atan2(dy12, dx12)

                # evaluate all p3 and (optionally) keep only the best per seed
                best_local: Optional[Tuple[float,int,Dict[str,float]]] = None
                for i3 in idx3_list:
                    p3 = C3[i3]; x3, y3 = p3['x'], p3['y']
                    d23 = hypot(x3 - x2, y3 - y2)
                    if d23 == 0.0:
                        continue
                    # prediction residual and cross-track
                    e = hypot(x3 - x3p, y3 - y3p)
                    cross = abs((x3-x1)*(-uy) + (y3-y1)*(ux))
                    rv = (d23/dt23) / (d12/dt12)
                    th23 = atan2(y3 - y2, x3 - x2)
                    dth_deg = abs(_wrap_angle_rad(th23 - th12)) * 180.0/pi

                    # small-displacement micro-mode
                    total_disp = d12 + d23
                    if total_disp < P.small_disp_total_px:
                        w_resid, w_cross, w_angle, w_vr = 0.70, 0.30, 0.0, 0.0
                        angle_scale = P.angle_soft_scale_deg  # unused in micro-mode
                    else:
                        w_resid, w_cross = P.w_resid, P.w_cross
                        w_angle, w_vr = P.w_angle, P.w_vratio
                        angle_scale = P.angle_soft_scale_deg

                    # SOFT priors only; normalise by characteristic scales
                    J = (w_resid * (e / r_pred) +
                         w_cross * (cross / r_pred) +
                         w_angle * (dth_deg / max(1e-6, angle_scale)) +
                         w_vr    * abs(log(max(1e-6, rv))))

                    diag = dict(residual_px=e, cross_track_px=cross,
                                angle_change_deg=dth_deg, vel_ratio=rv,
                                r_pred_px=r_pred, total_disp_px=total_disp, disp12_px=d12,          # NEW
                                disp23_px=d23 )

                    if best_local is None or J < best_local[0]:
                        best_local = (J, i3, diag)

                if best_local is None:
                    continue

                J_best, i3_best, diag_best = best_local
                triplets.append((J_best, i1, i2, i3_best, diag_best))

        # CHANGE #3: Greedy de-conflict by ascending score, tie-break by longer path
        triplets.sort(key=lambda t: (t[0], -t[4].get('total_disp_px', 0.0)))
        used1, used2, used3 = set(), set(), set()
        out: List[Dict[str, Any]] = []

        for J, i1, i2, i3, diag in triplets:
            if i1 in used1 or i2 in used2 or i3 in used3:
                continue
            used1.add(i1); used2.add(i2); used3.add(i3)
            out.append({
                'p1': C1[i1], 'p2': C2[i2], 'p3': C3[i3],
                'J': float(J), **{k: float(v) for k,v in diag.items()}
            })
        return out

# Quick self-test
if __name__ == "__main__":
    C1 = [{'x':10.0,'y':10.0}]
    C2 = [{'x':10.6,'y':10.8}]  # small hop should now seed (>=0.8 px not required here but example)
    C3 = [{'x':11.2,'y':11.6}]
    linker = PPA3LinkerV2()
    out = linker.link_three_frames(C1,C2,C3,0.0,1.0,2.0)
    print("OK", out[0]['J'], out[0]['residual_px'])
