"""Plot joint angles and an estimated 2D center of mass from Sports2D output.

Usage: python analyze.py <pose_m_personXX.trc> <angles_personXX.mot> <out_dir>

Center of mass uses Winter's anthropometric table: each body segment has a
fixed share of body mass and a CoM located at a fixed fraction along the
segment (proximal -> distal). Whole-body CoM is the mass-weighted average.
From a single camera this is a 2D approximation.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (mass fraction, proximal point, distal point, CoM fraction from proximal)
SEGMENTS = {
    "head":       (0.081,  "Neck",      "Head",      1.0),
    "trunk":      (0.497,  "Hip",       "Neck",      0.5),
    "R_upperarm": (0.028,  "RShoulder", "RElbow",    0.436),
    "L_upperarm": (0.028,  "LShoulder", "LElbow",    0.436),
    "R_forearm":  (0.016,  "RElbow",    "RWrist",    0.430),
    "L_forearm":  (0.016,  "LElbow",    "LWrist",    0.430),
    "R_hand":     (0.006,  "RWrist",    "RWrist",    0.0),
    "L_hand":     (0.006,  "LWrist",    "LWrist",    0.0),
    "R_thigh":    (0.100,  "RHip",      "RKnee",     0.433),
    "L_thigh":    (0.100,  "LHip",      "LKnee",     0.433),
    "R_shank":    (0.0465, "RKnee",     "RAnkle",    0.433),
    "L_shank":    (0.0465, "LKnee",     "LAnkle",    0.433),
    "R_foot":     (0.0145, "RAnkle",    "RBigToe",   0.5),
    "L_foot":     (0.0145, "LAnkle",    "LBigToe",   0.5),
}


def read_trc(path):
    lines = Path(path).read_text().splitlines()
    markers = [m for m in lines[3].split("\t")[2:] if m.strip()]
    data = np.loadtxt(lines[5:], delimiter="\t", ndmin=2)
    time = data[:, 1]
    xyz = data[:, 2:].reshape(len(data), -1, 3)
    return time, {m: xyz[:, i, :2] for i, m in enumerate(markers)}


def read_mot(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 1
    return pd.read_csv(path, sep="\t", skiprows=start)


def center_of_mass(pts):
    com = 0
    for mass, prox, dist, frac in SEGMENTS.values():
        com = com + mass * (pts[prox] + frac * (pts[dist] - pts[prox]))
    return com


def main(trc_path, mot_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    time, pts = read_trc(trc_path)
    com = center_of_mass(pts)
    feet = np.stack([pts[k][:, 0] for k in ("LHeel", "RHeel", "LBigToe", "RBigToe")], axis=1)
    support_min, support_max = np.nanmin(feet, axis=1), np.nanmax(feet, axis=1)
    inside = (com[:, 0] >= support_min) & (com[:, 0] <= support_max)

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].fill_between(time, support_min, support_max, color="tab:green", alpha=0.2, label="Base of support (feet, horizontal)")
    ax[0].plot(time, com[:, 0], color="tab:red", label="Center of mass, horizontal")
    ax[0].set_ylabel("x (m)")
    ax[0].legend(loc="upper right")
    ax[0].set_title(f"Center of mass vs. base of support - CoM over feet {inside.mean():.0%} of frames")
    ax[1].plot(time, com[:, 1], color="tab:blue")
    ax[1].set_ylabel("CoM height (m)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "center_of_mass.png", dpi=120)

    angles = read_mot(mot_path)
    # "shoulders" is the shoulder-line segment angle; it wraps around ±180° and swamps the plot
    wanted = [c for c in angles.columns
              if c != "shoulders" and any(k in c.lower() for k in ("elbow", "shoulder", "hip", "knee", "trunk"))]
    fig, axes = plt.subplots(len(wanted), 1, figsize=(11, 1.8 * len(wanted)), sharex=True)
    for a, col in zip(np.atleast_1d(axes), wanted):
        a.plot(angles["time"], angles[col])
        a.set_ylabel(col, rotation=0, ha="right", va="center")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "joint_angles.png", dpi=110)

    summary = angles[wanted].agg(["min", "max"]).T
    summary["range"] = summary["max"] - summary["min"]
    summary.round(1).to_csv(out / "angle_summary.csv")
    print(summary.round(1).to_string())
    print(f"\nCoM over base of support: {inside.mean():.0%} of frames")
    print(f"CoM height range: {np.nanmin(com[:, 1]):.2f} - {np.nanmax(com[:, 1]):.2f} m")


if __name__ == "__main__":
    main(*sys.argv[1:4])
