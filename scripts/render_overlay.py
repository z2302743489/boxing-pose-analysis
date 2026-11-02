"""Render a slow-motion video with skeleton, center of mass and trunk lean drawn on every frame.

Usage: python render_overlay.py <video> <pose_px_personXX.trc> <out.mp4> [punches.txt] [--speed 0.5] [--stance orthodox]

- Skeleton: left side blue, right side red, spine white.
- Center of mass (Winter segment table): yellow target, plumb line to the floor, short trail.
- Base of support: bar under the feet, green when the CoM is over it, red when outside.
- Trunk lean: hip->neck vs vertical, positive = leaning toward the lead foot.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze import center_of_mass, read_trc

SCALE = 2
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
BLUE, RED, WHITE, YELLOW = (255, 150, 40), (60, 60, 240), (255, 255, 255), (0, 220, 255)  # BGR
GREEN, BAD = (90, 210, 90), (60, 60, 240)
BONES = [
    ("Head", "Neck", WHITE), ("Neck", "Hip", WHITE),
    ("Neck", "LShoulder", BLUE), ("LShoulder", "LElbow", BLUE), ("LElbow", "LWrist", BLUE),
    ("Neck", "RShoulder", RED), ("RShoulder", "RElbow", RED), ("RElbow", "RWrist", RED),
    ("Hip", "LHip", BLUE), ("LHip", "LKnee", BLUE), ("LKnee", "LAnkle", BLUE),
    ("LAnkle", "LHeel", BLUE), ("LAnkle", "LBigToe", BLUE),
    ("Hip", "RHip", RED), ("RHip", "RKnee", RED), ("RKnee", "RAnkle", RED),
    ("RAnkle", "RHeel", RED), ("RAnkle", "RBigToe", RED),
]
FEET = ("LHeel", "LBigToe", "LSmallToe", "LAnkle", "RHeel", "RBigToe", "RSmallToe", "RAnkle")
PUNCH_NAMES = {"orthodox": {"left": ("刺拳 JAB", BLUE), "right": ("后手直拳 CROSS", RED)},
               "southpaw": {"right": ("刺拳 JAB", RED), "left": ("后手直拳 CROSS", BLUE)}}


def rolling_median(x, n):
    pad = np.pad(x, n // 2, mode="edge")
    return np.array([np.nanmedian(pad[i:i + n]) for i in range(len(x))])


def dashed(img, a, b, color, thick=2, dash=10):
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    for s in np.arange(0, length, 2 * dash):
        p = a + (b - a) * s / length
        q = a + (b - a) * min(s + dash, length) / length
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), color, thick, cv2.LINE_AA)


def read_punches(path):
    if not path or not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[0].isdigit():
            rows.append((parts[1], float(parts[2])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("trc"); ap.add_argument("out"); ap.add_argument("punches", nargs="?")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--stance", choices=PUNCH_NAMES, default="orthodox")
    args = ap.parse_args()

    t, p = read_trc(args.trc)
    p = {k: v * SCALE for k, v in p.items()}
    com = center_of_mass(p)
    torso = np.nanmedian(np.linalg.norm(p["Neck"] - p["Hip"], axis=1))
    lead, rear = ("LAnkle", "RAnkle") if args.stance == "orthodox" else ("RAnkle", "LAnkle")
    # which way is "forward" on screen: from rear ankle toward lead ankle, smoothed over ~0.5 s
    feet_dx = rolling_median(p[lead][:, 0] - p[rear][:, 0], 15)
    forward = np.where(np.abs(feet_dx) > 0.25 * torso, np.sign(feet_dx), 0)
    trunk = p["Neck"] - p["Hip"]
    lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))
    lean_sign = np.sign(trunk[:, 0]) * forward
    punches = read_punches(args.punches)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * SCALE, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * SCALE
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps * args.speed}", "-i", "-", "-r", f"{fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    big, mid, small = (ImageFont.truetype(FONT, s) for s in (34, 24, 19))
    trail = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(t):
            break
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
        texts = []  # (xy, text, font, BGR) drawn with PIL afterwards
        pt = lambda k: tuple(map(int, p[k][i]))
        valid = not np.isnan(com[i]).any()

        if valid:
            # base of support under the feet
            fx = np.array([p[k][i, 0] for k in FEET])
            floor_y = int(np.nanmax([p[k][i, 1] for k in FEET]) + 10)
            inside = np.nanmin(fx) <= com[i, 0] <= np.nanmax(fx)
            cv2.line(img, (int(np.nanmin(fx)), floor_y), (int(np.nanmax(fx)), floor_y), GREEN if inside else BAD, 6, cv2.LINE_AA)

            # skeleton
            for a, b, c in BONES:
                cv2.line(img, pt(a), pt(b), (0, 0, 0), 7, cv2.LINE_AA)
                cv2.line(img, pt(a), pt(b), c, 4, cv2.LINE_AA)
            for k in p:
                if k in ("Nose", "LSmallToe", "RSmallToe"):
                    continue
                c = BLUE if k.startswith("L") else RED if k.startswith("R") else WHITE
                cv2.circle(img, pt(k), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, pt(k), 4, c, -1, cv2.LINE_AA)

            # trunk lean: vertical reference, trunk line and arc at the hip
            hip, neck = pt("Hip"), pt("Neck")
            ref_len = int(0.8 * torso)
            dashed(img, hip, (hip[0], hip[1] - ref_len), WHITE, 2)
            cv2.line(img, hip, neck, YELLOW, 3, cv2.LINE_AA)
            ang_trunk = np.degrees(np.arctan2(trunk[i, 1], trunk[i, 0]))
            cv2.ellipse(img, hip, (int(0.35 * torso),) * 2, 0, -90, ang_trunk, YELLOW, 2, cv2.LINE_AA)
            word = "前倾" if lean_sign[i] > 0 else "后仰" if lean_sign[i] < 0 else "倾斜"
            side = 1 if trunk[i, 0] >= 0 else -1
            texts.append(((hip[0] + side * int(0.4 * torso) - (0 if side > 0 else 110), hip[1] - int(0.55 * torso)),
                          f"{word} {lean[i]:.0f}°", mid, YELLOW))

            # center of mass: trail, plumb line, target
            trail.append(tuple(map(int, com[i])))
            trail = trail[-12:]
            for j in range(1, len(trail)):
                cv2.line(img, trail[j - 1], trail[j], YELLOW, 2, cv2.LINE_AA)
            c = tuple(map(int, com[i]))
            dashed(img, c, (c[0], floor_y), YELLOW, 2, 8)
            cv2.circle(img, (c[0], floor_y), 6, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(img, c, 15, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, 13, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(img, c, 6, (0, 0, 0), -1, cv2.LINE_AA)
            texts.append(((c[0] + 20, c[1] - 14), "重心", small, YELLOW))

        # HUD panel at the top
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 230), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        texts.append(((20, 14), f"慢放 {args.speed:g}×    {t[i]:5.2f} s", small, WHITE))
        if valid:
            texts.append(((20, 46), f"躯干{word} {lean[i]:4.1f}°", big, YELLOW))
            span = p[lead][i, 0] - p[rear][i, 0]
            bar_x, bar_y, bar_w = 20, 140, w - 40
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (80, 80, 80), -1)
            if abs(span) > 0.25 * torso:
                r = (com[i, 0] - p[rear][i, 0]) / span
                mx = int(bar_x + np.clip(r, 0, 1) * bar_w)
                cv2.rectangle(img, (bar_x, bar_y), (mx, bar_y + 18), YELLOW, -1)
                cv2.line(img, (bar_x + bar_w // 2, bar_y - 4), (bar_x + bar_w // 2, bar_y + 22), WHITE, 2)
                texts.append(((20, 100), f"重心位置  {r * 100:3.0f}%  (0 后脚 → 100 前脚)", small, WHITE))
            else:
                texts.append(((20, 100), "重心位置  —  (双脚重叠/转身中)", small, WHITE))
            texts.append(((20, 166), "后脚", small, WHITE))
            texts.append(((w - 60, 166), "前脚", small, WHITE))
        x = 20
        for hand, tp in punches:
            if tp - 0.2 <= t[i] <= tp + 0.1:
                name, color = PUNCH_NAMES[args.stance][hand]
                texts.append(((x, 190), name, mid, color))
                x += int(mid.getlength(name)) + 24

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for xy, s, font, bgr in texts:
            draw.text(xy, s, font=font, fill=bgr[::-1], stroke_width=2, stroke_fill=(0, 0, 0))
        ff.stdin.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).tobytes())
        i += 1

    ff.stdin.close()
    ff.wait()
    print(f"{i} frames -> {args.out} ({i / (fps * args.speed):.1f} s at {args.speed:g}x)")


if __name__ == "__main__":
    main()
