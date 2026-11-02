"""Detect punches in a Sports2D shadowboxing track and measure weight shift and sequencing.

Usage: python punch_analysis.py <pose_m_personXX.trc> <out_dir>

- Punch = peak of horizontal wrist-to-shoulder reach (normalized by torso length), only
  while the fist is around shoulder height - so hanging arms don't count.
- Weight shift = horizontal CoM position between the ankles, oriented toward the punch:
  0 = over the rear foot, 1 = over the front foot. Horizontal only: with a low, angled
  camera the front foot sits lower in the image, which would skew a 2D projection.
- Sequencing = time of peak forward velocity of hips, shoulders and fist before impact.
  A good kinetic chain fires hips -> shoulders -> fist.
All measurements are 2D from one camera, so treat them as trends, not lab numbers.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from analyze import center_of_mass, read_trc

EXT_MIN, PROMINENCE, WINDOW_S = 0.60, 0.25, 0.4
MAX_DROP = 0.5  # fist may sit at most this many torso lengths below the shoulder


def main(trc_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t, p = read_trc(trc_path)
    fps = 1 / np.median(np.diff(t))
    win = int(WINDOW_S * fps)
    torso = np.nanmedian(np.linalg.norm(p["Neck"] - p["Hip"], axis=1))
    com = center_of_mass(p)
    shoulders = (p["LShoulder"] + p["RShoulder"]) / 2
    smooth = lambda x: savgol_filter(x, 7, 2)
    vel = lambda x: np.gradient(smooth(x), 1 / fps)

    punches = []
    ext = {}
    for side, name in (("L", "left"), ("R", "right")):
        d = p[side + "Wrist"] - p[side + "Shoulder"]
        at_shoulder_height = d[:, 1] > -MAX_DROP * torso
        ext[side] = smooth(np.where(at_shoulder_height, np.abs(d[:, 0]), 0) / torso)
        peaks, _ = find_peaks(ext[side], height=EXT_MIN, prominence=PROMINENCE, distance=int(0.25 * fps))
        for k in peaks:
            start = max(0, k - win) + int(np.argmin(ext[side][max(0, k - win):k + 1]))
            direction = np.sign(d[k, 0])
            # front foot = the ankle further along the punch direction
            rear, front = sorted((p["LAnkle"][:, 0], p["RAnkle"][:, 0]), key=lambda a: direction * a[k])
            r = lambda i: (com[i, 0] - rear[i]) / (front[i] - rear[i])
            lo = max(0, start - 3)
            seg = slice(lo, k + 1)
            fwd = lambda x: direction * vel(x[:, 0])[seg]
            t_hip, t_sho, t_fist = (t[lo + int(np.argmax(fwd(x)))] for x in (p["Hip"], shoulders, p[side + "Wrist"]))
            punches.append(dict(
                hand=name, t=t[k], ext=ext[side][k], dur_ms=(t[k] - t[start]) * 1000,
                w_start=r(start), w_peak=r(k), shift=r(k) - r(start),
                hip_lead_ms=(t_fist - t_hip) * 1000, sho_lead_ms=(t_fist - t_sho) * 1000,
                chain_ok=t_hip <= t_sho <= t_fist,
            ))
    punches.sort(key=lambda d: d["t"])

    header = f"{'#':>2} {'hand':5} {'time':>6} {'reach':>5} {'dur ms':>6} {'CoM start':>9} {'CoM peak':>8} {'shift':>6} {'hip lead':>8} {'sho lead':>8} chain"
    lines = [header]
    for i, d in enumerate(punches, 1):
        lines.append(f"{i:>2} {d['hand']:5} {d['t']:6.2f} {d['ext']:5.2f} {d['dur_ms']:6.0f} {d['w_start']:9.2f} {d['w_peak']:8.2f} "
                     f"{d['shift']:+6.2f} {d['hip_lead_ms']:8.0f} {d['sho_lead_ms']:8.0f} {'yes' if d['chain_ok'] else 'no'}")
    for hand in ("left", "right"):
        ds = [d for d in punches if d["hand"] == hand]
        if ds:
            lines.append(f"\n{hand}: {len(ds)} punches | mean CoM shift {np.mean([d['shift'] for d in ds]):+.2f} "
                         f"| CoM at impact {np.mean([d['w_peak'] for d in ds]):.2f} "
                         f"| hips lead fist {np.median([d['hip_lead_ms'] for d in ds]):.0f} ms (median) "
                         f"| hip->shoulder->fist order in {np.mean([d['chain_ok'] for d in ds]):.0%}")
    report = "\n".join(lines)
    (out / "punches.txt").write_text(report + "\n")
    print(report)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for side, color in (("L", "tab:blue"), ("R", "tab:red")):
        ax[0].plot(t, ext[side], color=color, lw=1, label=f"{'Left' if side == 'L' else 'Right'} arm extension")
    for i, d in enumerate(punches, 1):
        c = "tab:blue" if d["hand"] == "left" else "tab:red"
        ax[0].annotate(str(i), (d["t"], d["ext"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=c)
        ax[1].plot([d["t"] - d["dur_ms"] / 1000, d["t"]], [d["w_start"], d["w_peak"]], color=c, marker="o", ms=3)
    ax[0].axhline(EXT_MIN, color="gray", ls=":", lw=1)
    ax[0].set_ylabel("wrist-shoulder / torso")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("Detected punches (numbered)")
    ax[1].axhspan(0, 1, color="tab:green", alpha=0.08)
    ax[1].axhline(0.5, color="gray", ls=":", lw=1)
    ax[1].set_ylabel("CoM between feet\n0 = rear, 1 = front")
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("Weight shift during each punch (start -> impact)")
    fig.tight_layout()
    fig.savefig(out / "punches.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:3])
