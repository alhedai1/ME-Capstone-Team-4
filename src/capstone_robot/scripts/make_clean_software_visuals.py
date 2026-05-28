#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "presentation_visuals_clean"

W, H = 1920, 1080
BG = (248, 250, 252)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
LIGHT = (226, 232, 240)
CARD = (255, 255, 255)
BLUE = (37, 99, 235)
GREEN = (22, 163, 74)
YELLOW = (202, 138, 4)
RED = (220, 38, 38)
PURPLE = (126, 34, 206)
CYAN = (8, 145, 178)
ORANGE = (234, 88, 12)


def font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = font(60, True)
F_H2 = font(38, True)
F_H3 = font(30, True)
F_BODY = font(28)
F_SMALL = font(23)
F_LABEL = font(22, True)


def canvas(title, subtitle=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 58), title, fill=INK, font=F_TITLE)
    if subtitle:
        d.text((82, 132), subtitle, fill=MUTED, font=F_BODY)
    return img, d


def rounded(d, box, fill=CARD, outline=LIGHT, radius=18, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(d, box, text, fill=INK, fnt=F_BODY, spacing=8):
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        b = d.textbbox((0, 0), line, font=fnt)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = box[1] + (box[3] - box[1] - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        x = box[0] + (box[2] - box[0] - tw) / 2
        d.text((x, y), line, fill=fill, font=fnt)
        y += th + spacing


def arrow(d, start, end, color=MUTED, width=5):
    d.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 18 if x2 >= x1 else -18
        pts = [(x2, y2), (x2 - s, y2 - 11), (x2 - s, y2 + 11)]
    else:
        s = 18 if y2 >= y1 else -18
        pts = [(x2, y2), (x2 - 11, y2 - s), (x2 + 11, y2 - s)]
    d.polygon(pts, fill=color)


def pill(d, xy, text, color, text_color=(255, 255, 255)):
    x, y = xy
    b = d.textbbox((0, 0), text, font=F_LABEL)
    w = b[2] - b[0] + 28
    h = b[3] - b[1] + 18
    d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=color)
    d.text((x + 14, y + 8), text, fill=text_color, font=F_LABEL)
    return x + w, y + h


def draw_camera_frame(d, box, label):
    rounded(d, box, fill=(241, 245, 249), outline=(148, 163, 184), radius=12, width=3)
    x1, y1, x2, y2 = box
    d.text((x1 + 24, y1 + 20), label, fill=MUTED, font=F_LABEL)
    d.line((x1 + (x2 - x1) / 2, y1 + 64, x1 + (x2 - x1) / 2, y2 - 24), fill=(148, 163, 184), width=3)
    d.text((x1 + (x2 - x1) / 2 + 14, y1 + 70), "image center", fill=(100, 116, 139), font=F_SMALL)


def software_state_machine():
    img, d = canvas(
        "Software Design: State-Based Autonomy",
        "Each state owns one perception problem and sends simple motor commands.",
    )
    states = [
        ("Search", "center pole\nin frame", BLUE),
        ("Approach", "steer using\nYOLO box", CYAN),
        ("Align", "orbit until\nbell over pole", PURPLE),
        ("Climb", "track bell circle\nwhile climbing", GREEN),
        ("Strike", "confirm bell\nthen actuate", ORANGE),
    ]
    y = 350
    x = 105
    bw, bh, gap = 260, 190, 70
    for i, (name, detail, color) in enumerate(states):
        box = (x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh)
        rounded(d, box, fill=CARD, outline=color, radius=22, width=4)
        d.text((box[0] + 28, box[1] + 28), name, fill=color, font=F_H2)
        d.text((box[0] + 28, box[1] + 92), detail, fill=INK, font=F_BODY)
        if i < len(states) - 1:
            arrow(d, (box[2] + 16, y + bh / 2), (box[2] + gap - 18, y + bh / 2), color=MUTED, width=5)

    sensors = [
        ("AI Camera", "front pole detection + climbing view", BLUE),
        ("Pi Camera", "upward alignment + striking", PURPLE),
        ("Motor/Servo", "tank drive, climb, bell strike", GREEN),
    ]
    sx, sy, sw, sh = 170, 705, 470, 145
    for i, (name, detail, color) in enumerate(sensors):
        box = (sx + i * (sw + 85), sy, sx + i * (sw + 85) + sw, sy + sh)
        rounded(d, box, fill=(255, 255, 255), outline=LIGHT, radius=16)
        d.ellipse((box[0] + 28, box[1] + 42, box[0] + 76, box[1] + 90), fill=color)
        d.text((box[0] + 98, box[1] + 32), name, fill=INK, font=F_H3)
        d.text((box[0] + 98, box[1] + 82), detail, fill=MUTED, font=F_SMALL)
    img.save(OUT_DIR / "software_state_machine_clean.png")


def pole_approach():
    img, d = canvas(
        "Ground Navigation: YOLO Box to Motor Command",
        "The AI camera gives one pole bounding box; the controller uses center error and box width.",
    )
    frame = (110, 245, 1070, 880)
    draw_camera_frame(d, frame, "front AI camera frame")
    fx1, fy1, fx2, fy2 = frame
    pole = (610, 365, 735, 750)
    d.rounded_rectangle(pole, radius=8, outline=GREEN, width=8)
    pcx = (pole[0] + pole[2]) / 2
    fcx = (fx1 + fx2) / 2
    d.line((pcx, pole[1], pcx, pole[3]), fill=BLUE, width=5)
    d.ellipse((pcx - 9, 548 - 9, pcx + 9, 548 + 9), fill=BLUE)
    d.line((fcx, 820, pcx, 820), fill=YELLOW, width=7)
    arrow(d, (fcx, 820), (pcx, 820), color=YELLOW, width=7)
    d.text((145, 790), "error_x = pole_center - frame_center", fill=INK, font=F_H3)
    d.text((650, 760), "width/frame -> distance", fill=GREEN, font=F_LABEL)

    panel = (1170, 280, 1795, 825)
    rounded(d, panel, fill=CARD, outline=LIGHT, radius=18)
    d.text((1225, 330), "Control logic", fill=INK, font=F_H2)
    rows = [
        ("error_x < -20 px", "turn left", YELLOW),
        ("|error_x| <= 20 px", "drive forward", GREEN),
        ("error_x > +20 px", "turn right", YELLOW),
        ("box width >= 20%", "stop: pole reached", RED),
    ]
    y = 420
    for cond, action, color in rows:
        d.text((1230, y), cond, fill=MUTED, font=F_BODY)
        pill(d, (1530, y - 8), action, color)
        y += 88
    img.save(OUT_DIR / "pole_approach_clean.png")


def pole_bell_alignment():
    img, d = canvas(
        "OpenCV Alignment: Pole Centerline vs. Bell Center",
        "The upward camera converts a vision error into an orbit direction.",
    )
    panels = [
        ("bell left", -180, "orbit left", YELLOW),
        ("aligned", 0, "stop + attach", GREEN),
        ("bell right", 155, "orbit right", YELLOW),
    ]
    px, py, pw, ph, gap = 100, 250, 530, 650, 65
    for i, (label, err, action, color) in enumerate(panels):
        box = (px + i * (pw + gap), py, px + i * (pw + gap) + pw, py + ph)
        draw_camera_frame(d, box, "upward camera")
        x1, y1, x2, y2 = box
        pole_x_bottom = x1 + pw * 0.52
        pole_x_top = pole_x_bottom - 55
        d.line((pole_x_top, y1 + 95, pole_x_bottom, y2 - 55), fill=GREEN, width=8)
        by = y1 + 300
        line_x_at_bell = pole_x_top + (pole_x_bottom - pole_x_top) * ((by - (y1 + 95)) / ((y2 - 55) - (y1 + 95)))
        bx = line_x_at_bell + err
        d.ellipse((bx - 44, by - 44, bx + 44, by + 44), outline=BLUE, width=8)
        d.ellipse((bx - 8, by - 8, bx + 8, by + 8), fill=BLUE)
        d.line((line_x_at_bell, by, bx, by), fill=YELLOW if err else GREEN, width=7)
        d.ellipse((line_x_at_bell - 8, by - 8, line_x_at_bell + 8, by + 8), fill=GREEN)
        centered_text(d, (x1 + 22, y2 - 170, x2 - 22, y2 - 105), f"error = {err:+.0f} px", fill=INK, fnt=F_H3)
        centered_text(d, (x1 + 22, y2 - 92, x2 - 22, y2 - 38), action, fill=color, fnt=F_H3)
        pill(d, (x1 + 24, y1 + 82), label, color if label == "aligned" else MUTED)
    img.save(OUT_DIR / "pole_bell_alignment_clean.png")


def bell_circle_tracking():
    img, d = canvas(
        "Climbing Feedback: Bell Circle Tracking",
        "During climbing, the camera watches the bell circle and switches between climb, descend, wait, and strike.",
    )
    steps = [
        ("1", "circle visible", "motors climb", GREEN),
        ("2", "circle lost", "stop motors;\nrobot slips down", YELLOW),
        ("3", "circle reacquired", "hold/wait\n3 seconds", CYAN),
        ("4", "bell confirmed", "servo strikes", ORANGE),
    ]
    x, y, bw, bh, gap = 95, 255, 405, 610, 50
    for i, (num, title, action, color) in enumerate(steps):
        box = (x + i * (bw + gap), y, x + i * (bw + gap) + bw, y + bh)
        draw_camera_frame(d, box, "climb camera")
        x1, y1, x2, y2 = box
        d.ellipse((x1 + 34, y1 + 78, x1 + 86, y1 + 130), fill=color)
        centered_text(d, (x1 + 34, y1 + 78, x1 + 86, y1 + 130), num, fill=(255, 255, 255), fnt=F_H3)
        d.text((x1 + 112, y1 + 86), title, fill=INK, font=F_H3)
        d.line((x1 + bw / 2, y1 + 110, x1 + bw / 2, y2 - 80), fill=(148, 163, 184), width=3)
        bell_cx = x1 + bw / 2 + [-30, 80, 50, 0][i]
        bell_cy = y1 + 280 + [20, -15, 35, 0][i]
        radius = [48, 0, 74, 64][i]
        if radius:
            d.ellipse((bell_cx - radius, bell_cy - radius, bell_cx + radius, bell_cy + radius), outline=BLUE, width=8)
            d.ellipse((bell_cx - 9, bell_cy - 9, bell_cx + 9, bell_cy + 9), fill=GREEN)
        else:
            d.line((bell_cx - 45, bell_cy - 45, bell_cx + 45, bell_cy + 45), fill=RED, width=8)
            d.line((bell_cx + 45, bell_cy - 45, bell_cx - 45, bell_cy + 45), fill=RED, width=8)
        centered_text(d, (x1 + 34, y2 - 150, x2 - 34, y2 - 58), action, fill=color, fnt=F_H3)
        if i < len(steps) - 1:
            arrow(d, (x2 + 10, y1 + bh / 2), (x2 + gap - 14, y1 + bh / 2), color=MUTED, width=5)
    img.save(OUT_DIR / "bell_circle_tracking_clean.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    software_state_machine()
    pole_approach()
    pole_bell_alignment()
    bell_circle_tracking()
    print(f"Saved clean visuals to: {OUT_DIR}")


if __name__ == "__main__":
    main()
