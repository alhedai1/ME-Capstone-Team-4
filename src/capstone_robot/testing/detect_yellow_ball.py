from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# CHAT: FIX YELLOW BALL FALSE POSITIVES


IMG_FOLDER = Path("../data/extracted_frames/may25/may25_strike_bell5fps")
# IMG_FOLDER = Path("../data/extracted_frames/may15/bell1")
# Filter to ensure we only try to read files, skipping directories
IMG_PATHS = [path for path in IMG_FOLDER.iterdir() if path.is_file()]

def detect_yellow_ball(path):
    frame = cv2.imread(path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broad warm-color range: yellow/orange/brown-ish
    lower = np.array([0, 40, 40])
    upper = np.array([35, 255, 255])
    # lower = np.array([8, 70, 40])
    # upper = np.array([32, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Optional: ignore very dark pixels and very white/washed-out pixels
    h, s, v = cv2.split(hsv)
    mask = cv2.bitwise_and(mask, cv2.inRange(s, 50, 255))
    mask = cv2.bitwise_and(mask, cv2.inRange(v, 50, 255))

    # Clean noise
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 10000 or area > 100000:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h
        if not 0.75 <= aspect <= 1.33:
            continue
        # extent = area / (w * h)
        # if not 0.45 <= extent <= 0.9:
        #     continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        (x, y), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        fill_ratio = area / circle_area if circle_area > 0 else 0

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        if solidity < 0.75:
            continue
        score = (
            2.0 * fill_ratio
            + 1.5 * solidity
            + 1.0 * (1.0 - abs(aspect - 1.0))
            - 0.00001 * area
        )

        # Tune these based on your camera distance
        if circularity > 0.45 and fill_ratio > 0.45 and r > 20:
            candidates.append((score, area, int(x), int(y), int(r), circularity, fill_ratio))
            copy = frame.copy()
            cv2.drawContours(copy, [c], -1, (255,0,0), -1)
            cv2.imshow("copy", copy)

    if not candidates:
        return frame, None, mask

    # Choose largest valid circular warm blob
    # candidates.sort(reverse=True)
    candidates.sort()

    score, _, x, y, r, circularity, fill_ratio = candidates[0]
    print(f"circularity: {circularity}")
    print(f"fill ratio: {fill_ratio}")

    out = frame.copy()
    cv2.circle(out, (x,y), r, (255,0,0), 1)

    return out, (x, y, r), mask


for path in IMG_PATHS:
    # detect_bell_adaptive(path)
    out, circle, mask = detect_yellow_ball(path)
    if circle:
        (x,y,r) = circle
        print(f"area: {math.pi*r*r}")
    cv2.imshow("img", out)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)