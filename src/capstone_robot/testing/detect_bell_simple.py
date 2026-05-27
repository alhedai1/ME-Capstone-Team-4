from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

IMG_FOLDER = Path("../data/extracted_frames/may25/may25_strike_bell5fps")
# IMG_FOLDER = Path("../data/extracted_frames/may15/bell1")
# Filter to ensure we only try to read files, skipping directories
IMG_PATHS = [path for path in IMG_FOLDER.iterdir() if path.is_file()]

def detect_bell(path):
    img = cv2.imread(str(path)) # Convert Path object to string for cv2.imread
    if img is None:
        return
        
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Warm/gold/brown/yellow regions
    gold_mask = cv2.inRange(
        hsv,
        np.array([8, 35, 40]),
        np.array([30, 255, 255])
    )

    # Bright low-saturation reflections
    bright_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 90, 255])
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    gold_nearby = cv2.dilate(gold_mask, kernel, iterations=2)

    valid_bright = cv2.bitwise_and(bright_mask, gold_nearby)
    mask = cv2.bitwise_or(gold_mask, valid_bright)
    
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    coverage = np.count_nonzero(clean_mask) / clean_mask.size
    print(f"coverage: {coverage}")

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw contours on a copy to keep the original image clean for plotting
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        roi_area = img.shape[0] * img.shape[1]
        area_ratio = area / roi_area
        print(f"{path.name} - Area Ratio: {area_ratio:.4f}")
    
        max_contour = img.copy()
        cv2.drawContours(max_contour, largest, -1, (0, 255, 0), -1)
        cv2.imshow("max contour", max_contour)

    # --- Matplotlib Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f"Detection: {path.name}", fontsize=12)

    axes = axes.flatten()
    
    # Convert BGR to RGB for correct matplotlib colors
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    
    axes[1].imshow(gold_mask, cmap='gray')
    axes[1].set_title("Initial Mask")
    
    axes[2].imshow(clean_mask, cmap='gray')
    axes[2].set_title("Clean Mask")
    
    axes[3].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Contours")
    
    # Remove axis ticks for cleaner look
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_bell_adaptive(path):
    img = cv2.imread(str(path))
    if img is None:
        return
        
    # 1. Focus on brightness/structure, not color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Adaptive thresholding captures edges in both bright zones and shadows
    adaptive_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 4
    )

    # # 3. Morphological close to bridge gaps caused by directional highlights
    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # closed_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. WIPE OUT BACKGROUND NOISE (Size-based opening)
    # Tiny background noise dots are smaller than 5x5 pixels. This deletes them.
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_dots = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel_noise)
    
    # # Fill solid holes inside the bell shape
    # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # clean_mask = np.zeros_like(closed_mask)

    # 4. CONNECT THE BELL SHAPE (Morphological closing)
    # This bridges the gaps in the bell caused by shadows or reflections.
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed_mask = cv2.morphologyEx(clean_dots, cv2.MORPH_CLOSE, kernel_connect)

    # 5. FILL THE BELL HOLES
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(closed_mask)

    if contours:
        for i, c in enumerate(contours):
            # Draw both external and internal contours as solid blocks
            cv2.drawContours(clean_mask, contours, i, 255, -1)

    coverage = np.count_nonzero(clean_mask) / clean_mask.size
    print(f"coverage: {coverage}")

    # Re-find combined solid contours
    final_contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_contours = img.copy()
    cv2.drawContours(img_contours, final_contours, -1, (0, 255, 0), 3)
    
    if final_contours:
        largest = max(final_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        roi_area = img.shape[0] * img.shape[1]
        area_ratio = area / roi_area
        print(f"{path.name} - Area Ratio: {area_ratio:.4f}")
    
    # --- Matplotlib Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(f"Detection: {path.name}", fontsize=12)
    axes = axes.flatten()
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    
    axes[1].imshow(adaptive_mask, cmap='gray')
    axes[1].set_title("Adaptive Mask (Local Contrast)")
    
    axes[2].imshow(clean_mask, cmap='gray')
    axes[2].set_title("Filled Solid Mask")
    
    axes[3].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Detected Edges")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


# def detect_blobs(path):
#     """
#     Detects blobs in a given frame using SimpleBlobDetector.
#     """
#     # 1. Preprocess: Convert to grayscale and smooth out noise
#     frame = cv2.imread(path)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # 2. Setup Default SimpleBlobDetector parameters
#     params = cv2.SimpleBlobDetector_Params()
    
#     # Filter by Area (Adjust minArea based on your object size)
#     params.filterByArea = True
#     params.minArea = 100
#     params.maxArea = 5000000
    
#     # Filter by Circularity (1.0 is a perfect circle)
#     params.filterByCircularity = True
#     params.minCircularity = 0.5
    
#     # Filter by Convexity
#     params.filterByConvexity = False
    
#     # Filter by Inertia (How elongated the shape is)
#     params.filterByInertia = False
    
#     # 3. Create detector with parameters
#     detector = cv2.SimpleBlobDetector_create(params)
    
#     # 4. Detect blobs
#     keypoints = detector.detect(blurred)
    
#     # 5. Draw detected blobs as red circles
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures circle size matches blob size
#     frame_with_blobs = cv2.drawKeypoints(
#         frame, 
#         keypoints, 
#         np.array([]), 
#         (0, 0, 255), 
#         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
#     )
    
#     return frame_with_blobs, keypoints

def detect_yellow_ball(path):
    frame = cv2.imread(path)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broad warm-color range: yellow/orange/brown-ish
    lower = np.array([3, 40, 40])
    upper = np.array([35, 255, 255])
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

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        (x, y), r = cv2.minEnclosingCircle(c)
        circle_area = np.pi * r * r
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # Tune these based on your camera distance
        if circularity > 0.45 and fill_ratio > 0.45 and r > 20:
            candidates.append((area, int(x), int(y), int(r), circularity, fill_ratio))
            copy = frame.copy()
            cv2.drawContours(copy, [c], -1, (255,0,0), -1)
            cv2.imshow("copy", copy)

    if not candidates:
        return frame, None, mask

    # Choose largest valid circular warm blob
    candidates.sort(reverse=True)
    _, x, y, r, circularity, fill_ratio = candidates[0]
    print(circularity)

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
    # cv2.imshow("mask", mask)
    cv2.waitKey(0)
