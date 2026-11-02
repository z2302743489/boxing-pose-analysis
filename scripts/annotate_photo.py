"""Annotate a single boxing photo: skeleton, center of mass and trunk lean for the two main
people, plus a posture-based estimate of the punch's peak impact force.

Usage: python annotate_photo.py <photo> <out.jpg> [--mass 70] [--stance orthodox]

Peak force model (impulse-momentum with a half-sine contact pulse):
    F_peak = (pi/2) * m_eff * v / dt
- m_eff: effective striking mass. Arm-only punches deliver ~2.5% of body mass; full body
  involvement (weight onto the lead foot, rear leg driven straight, arm aligned) up to ~6%.
  Literature: Olympic boxers average ~2.9 kg (Walilko et al., 2005).
- v: fist speed, not visible in a photo -> scenarios of 6 / 7.5 / 9 m/s.
- dt: glove-on-glove contact time -> 20 / 15 / 12 ms.
This is an order-of-magnitude estimate, not a measurement.
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rtmlib import BodyWithFeet

from analyze import center_of_mass
from render_overlay import BLUE, BAD, BONES, FEET, FONT, GREEN, RED, WHITE, YELLOW, dashed

HALPE26 = ("Nose LEye REye LEar REar LShoulder RShoulder LElbow RElbow LWrist RWrist LHip RHip "
           "LKnee RKnee LAnkle RAnkle Head Neck Hip LBigToe RBigToe LSmallToe RSmallToe LHeel RHeel").split()
LEG = ("Hip", "Knee", "Ankle", "BigToe", "SmallToe", "Heel")


def angle(a, b, c):
    """Angle at b (degrees) between b->a and b->c."""
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def analyze_person(p):
    facing = np.sign(p["Nose"][0] - p["Neck"][0])  # +1 = facing image right
    front, rear = sorted(("LAnkle", "RAnkle"), key=lambda k: -facing * p[k][0])
    com = center_of_mass({k: v[None] for k, v in p.items()})[0]
    trunk = p["Neck"] - p["Hip"]
    lean = np.degrees(np.arctan2(abs(trunk[0]), -trunk[1]))
    lean_signed = lean * np.sign(trunk[0]) * facing  # + = toward the way they face
    weight = (com[0] - p[rear][0]) / (p[front][0] - p[rear][0])
    torso = np.linalg.norm(trunk)
    reach = {s: facing * (p[s + "Wrist"][0] - p[s + "Shoulder"][0]) / torso for s in "LR"}
    return dict(facing=facing, front=front, rear=rear, com=com, lean=lean, lean_signed=lean_signed,
                weight=weight, reach=reach, torso=torso)


def swap_legs(p):
    q = dict(p)
    for part in LEG[1:] + ("Hip",):
        q["L" + part], q["R" + part] = p["R" + part], p["L" + part]
    return q


def estimate_force(p, a, mass):
    hand = max(a["reach"], key=a["reach"].get)
    elbow = angle(p[hand + "Shoulder"], p[hand + "Elbow"], p[hand + "Wrist"])
    rear_side = a["rear"][0]
    knee = angle(p[rear_side + "Hip"], p[rear_side + "Knee"], p[rear_side + "Ankle"])
    scores = {
        "重心前移": np.clip((a["weight"] - 0.4) / 0.4, 0, 1),
        "后腿蹬伸": np.clip((knee - 120) / 50, 0, 1),
        "手臂伸直": np.clip((elbow - 140) / 35, 0, 1),
    }
    involvement = float(np.mean(list(scores.values())))
    m_eff = mass * (0.025 + 0.035 * involvement)
    peak = lambda v, dt: np.pi / 2 * m_eff * v / dt
    return dict(hand=hand, elbow=elbow, knee=knee, scores=scores, involvement=involvement, m_eff=m_eff,
                low=peak(6.0, 0.020), mid=peak(7.5, 0.015), high=peak(9.0, 0.012),
                punch_type="后手直拳" if hand != a["front"][0] else "刺拳")


def draw_person(img, p, a, texts, font):
    pt = lambda k: tuple(map(int, p[k]))
    fx = np.array([p[k][0] for k in FEET])
    floor_y = int(max(p[k][1] for k in FEET) + 14)
    inside = fx.min() <= a["com"][0] <= fx.max()
    cv2.line(img, (int(fx.min()), floor_y), (int(fx.max()), floor_y), GREEN if inside else BAD, 8, cv2.LINE_AA)
    for s, e, c in BONES:
        cv2.line(img, pt(s), pt(e), (0, 0, 0), 9, cv2.LINE_AA)
        cv2.line(img, pt(s), pt(e), c, 5, cv2.LINE_AA)
    for k in p:
        if k in ("Nose", "LEye", "REye", "LEar", "REar", "LSmallToe", "RSmallToe"):
            continue
        c = BLUE if k.startswith("L") else RED if k.startswith("R") else WHITE
        cv2.circle(img, pt(k), 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, pt(k), 5, c, -1, cv2.LINE_AA)

    hip, neck = pt("Hip"), pt("Neck")
    torso = a["torso"]
    dashed(img, hip, (hip[0], int(hip[1] - 0.9 * torso)), WHITE, 3, 12)
    cv2.line(img, hip, neck, YELLOW, 4, cv2.LINE_AA)
    trunk_ang = np.degrees(np.arctan2(neck[1] - hip[1], neck[0] - hip[0]))
    cv2.ellipse(img, hip, (int(0.4 * torso),) * 2, 0, -90, trunk_ang, YELLOW, 3, cv2.LINE_AA)
    word = "前倾" if a["lean_signed"] > 1 else "后仰" if a["lean_signed"] < -1 else "直立"
    side = -1 if neck[0] < hip[0] else 1
    texts.append(((hip[0] + side * int(0.45 * torso) - (150 if side < 0 else 0), hip[1] - int(0.55 * torso)),
                  f"{word} {a['lean']:.0f}°", font, YELLOW))

    c = tuple(map(int, a["com"]))
    dashed(img, c, (c[0], floor_y), YELLOW, 3, 10)
    cv2.circle(img, (c[0], floor_y), 9, YELLOW, -1, cv2.LINE_AA)
    cv2.circle(img, c, 22, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, c, 19, YELLOW, -1, cv2.LINE_AA)
    cv2.circle(img, c, 8, (0, 0, 0), -1, cv2.LINE_AA)
    texts.append(((c[0] + 28, c[1] - 18), "重心", font, YELLOW))
    return word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("out")
    ap.add_argument("--mass", type=float, default=70, help="puncher body mass in kg")
    ap.add_argument("--stance", choices=("orthodox", "southpaw"), default="orthodox", help="puncher stance")
    args = ap.parse_args()

    img = cv2.imread(args.photo)
    kps, scores = BodyWithFeet(mode="performance", backend="onnxruntime", device="cpu")(img)
    # keep the two largest people (drops small background figures and reflections)
    size = [np.ptp(k[:, 1]) for k in kps]
    people = [dict(zip(HALPE26, kps[i])) for i in np.argsort(size)[::-1][:2]]
    stats = [analyze_person(p) for p in people]
    pi = int(np.argmax([max(a["reach"].values()) for a in stats]))  # puncher = longest reach
    di = 1 - pi

    # Pose models often swap leg labels on side/back views; fix the puncher's from the known stance
    lead = "L" if args.stance == "orthodox" else "R"
    if stats[pi]["front"][0] != lead:
        people[pi] = swap_legs(people[pi])
        stats[pi] = analyze_person(people[pi])
    force = estimate_force(people[pi], stats[pi], args.mass)

    h, w = img.shape[:2]
    panel_h = 440
    canvas = np.zeros((h + panel_h, w, 3), np.uint8)
    canvas[:h] = img
    texts = []
    label, small, big = (ImageFont.truetype(FONT, s) for s in (34, 26, 40))
    words = {i: draw_person(canvas, people[i], stats[i], texts, label) for i in (pi, di)}
    # a posture-based estimate doesn't deserve more than two significant figures
    r100 = lambda x: f"{round(x, -2):,.0f}"

    # impact arrow along the punching forearm
    hand = force["hand"]
    wrist, elbow = people[pi][hand + "Wrist"], people[pi][hand + "Elbow"]
    d = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    tip = wrist + d * 40
    tail = wrist - d * 200
    cv2.arrowedLine(canvas, tuple(map(int, tail)), tuple(map(int, tip)), (0, 0, 0), 16, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, tuple(map(int, tail)), tuple(map(int, tip)), (40, 90, 255), 10, cv2.LINE_AA, tipLength=0.25)
    texts.append(((int(wrist[0]) - 40, int(wrist[1]) - 120), f"冲击力峰值 ≈ {r100(force['mid'])} N", big, (40, 90, 255)))
    texts.append(((int(wrist[0]) - 40, int(wrist[1]) - 70), f"约 {round(force['mid'] / 9.81, -1):.0f} 公斤力", label, (40, 90, 255)))

    # bottom panel: per-person numbers, then full-width rows for the force breakdown
    y0 = h + 22
    for col, (i, name) in enumerate(((di, "防守者"), (pi, "出拳者"))):
        a, x = stats[i], 30 + col * (w // 2)
        texts.append(((x, y0), name, big, WHITE))
        texts.append(((x, y0 + 56), f"躯干{words[i]} {a['lean']:.1f}°", label, YELLOW))
        texts.append(((x, y0 + 100), f"重心位置 {a['weight'] * 100:.0f}%（0 后脚 → 100 前脚）", label, WHITE))
    s = force["scores"]
    texts.append(((30, y0 + 170), f"出拳者这一拳：{force['punch_type']} · 身体参与度 {force['involvement'] * 100:.0f}%"
                  f"（重心前移 {s['重心前移'] * 100:.0f}% · 后腿蹬伸 {s['后腿蹬伸'] * 100:.0f}% · 手臂伸直 {s['手臂伸直'] * 100:.0f}%）",
                  label, WHITE))
    texts.append(((30, y0 + 220), f"冲击力峰值估计 ≈ {r100(force['mid'])} N（约 {round(force['mid'] / 9.81, -1):.0f} 公斤力），"
                  f"范围 {r100(force['low'])}–{r100(force['high'])} N", big, (40, 90, 255)))
    texts.append(((30, y0 + 300), f"估算：F峰 ≈ π/2 × 有效质量 × 拳速 ÷ 接触时间。体重按 {args.mass:.0f} kg，"
                  f"有效质量 {force['m_eff']:.1f} kg（由姿态推算），拳速 6–9 m/s，接触时间 12–20 ms。", small, (190, 190, 190)))
    texts.append(((30, y0 + 340), "照片测不出拳速，结果是量级估计，不是实测。参考：奥运拳击手直拳平均约 3,400 N（Walilko 等，2005）。",
                  small, (190, 190, 190)))

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for xy, s_, f, bgr in texts:
        draw.text(xy, s_, font=f, fill=bgr[::-1], stroke_width=3, stroke_fill=(0, 0, 0))
    pil.save(args.out, quality=92)

    for i, name in ((di, "defender"), (pi, "puncher")):
        a = stats[i]
        print(f"{name}: lean {a['lean_signed']:+.1f} deg, CoM {a['weight']:.2f} (0 rear, 1 front), front foot {a['front']}")
    print({k: v if isinstance(v, str) else round(float(v), 2) for k, v in force.items() if k != "scores"},
          {k: round(float(v), 2) for k, v in force["scores"].items()})


if __name__ == "__main__":
    main()
