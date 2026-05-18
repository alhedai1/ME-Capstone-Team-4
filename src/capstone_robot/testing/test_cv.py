import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from capstone_robot.utils import find_repo_root

REPO_ROOT = find_repo_root(__file__)

img_path = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim/frame_000045.jpg"
img_folder = REPO_ROOT / "src/capstone_robot/data/extracted_frames/may15/test1_trim"


def filter_lines(lines):
    output = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 70 < angle < 110:  # Matches vertical-ish lines
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output.append(line)
    return output

def get_segments(lines):
    left_pole_segments = []
    right_pole_segments = []
    # valid_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope and angle
            if x2 - x1 == 0: continue # Avoid division by zero

            slope = (y2 - y1) / (x2 - x1)
            angle = np.degrees(np.arctan(slope))
            
            # Filter for the vertical converging angles of your pole
            # Left edge leans right (/), Right edge leans left (\)
            if -90 <= angle < -60: 
                left_pole_segments.append((x1, y1, x2, y2))
            elif 60 < angle <= 90:
                right_pole_segments.append((x1, y1, x2, y2))

    #         if 45 < np.abs(angle) < 135:
    #             mid_x = (x1 + x2) / 2
    #             valid_segments.append((mid_x, x1, y1, x2, y2))
    # valid_segments.sort(key=lambda item: item[0])
    # left_pole_points = []
    # right_pole_points = []
    # if len(valid_segments) >= 2:
    #     # Find the average X position of all detected pole pieces
    #     avg_x_center = np.mean([item[0] for item in valid_segments])
    #     print(avg_x_center)
        
    #     for mid_x, x1, y1, x2, y2 in valid_segments:
    #         if mid_x < avg_x_center:
    #             left_pole_points.append([x1, y1])
    #             left_pole_points.append([x2, y2])
    #         else:
    #             right_pole_points.append([x1, y1])
    #             right_pole_points.append([x2, y2])
    

    return left_pole_segments, right_pole_segments
    # return left_pole_points, right_pole_points

def draw_fused_line(segments, img):
    if len(segments) == 0: return
    points = []
    for x1, y1, x2, y2 in segments:
        points.append([x1, y1])
        points.append([x2, y2])
    
    # Fits a single robust line through all scattered points
    [vx, vy, x0, y0] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
    
    # Calculate start and end points to draw across the screen
    y_min, y_max = 0, img.shape[0]
    x_start = int(x0 + vx / vy * (y_min - y0))
    x_end = int(x0 + vx / vy * (y_max - y0))
    cv2.line(img, (x_start, y_min), (x_end, y_max), (0, 255, 0), 3)

# def fit_and_draw(points, img, color):
#     if len(points) < 2: return
    
#     # cv2.fitLine handles any slant, tilt, or diagonal perfectly
#     [vx, vy, x0, y0] = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
    
#     # Project the line from the very bottom to the top of the image frame
#     y_min, y_max = 0, img.shape[0]
#     x_start = int(x0[0] + (vx[0] / vy[0]) * (y_min - y0[0]))
#     x_end = int(x0[0] + (vx[0] / vy[0]) * (y_max - y0[0]))
    
#     cv2.line(img, (x_start, y_min), (x_end, y_max), color, 3)

def detect_lines_circles(gray, blurred):
    # blurred = cv2.medianBlur(gray, 5)
    # cv2.imshow("blurred1", blurred)
    # _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("binary", binary)
    
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY_INV, 11, 2)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=5)

    # lines = filter_lines(lines)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=150, param2=20, minRadius=0, maxRadius=30)

    return lines, circles, edges

def draw_lines(img, lines, color):
    # draw lines
    # line_img = img.copy()
    if lines is not None:
        for line in lines:
        #     print(f"line: {line}")
        #     print(f"line[0]: {line[0]}")
            x1,y1,x2,y2 = line
            cv2.line(img, (x1,y1), (x2,y2), color, 2)

def draw_circles(img):
    # draw circles
    if circles is not None:
        for circle in circles:
            cx, cy, r = circle[0]
            cv2.circle(img, (int(cx), int(cy)), int(r), (255,0,0), 2)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
plt.show(block=False)

for img_path in img_folder.iterdir():
    img = cv2.imread(img_path)
    lines_img = img.copy()
    final_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    lines, circles, edges = detect_lines_circles(gray, blurred)

    left_segments, right_segments = get_segments(lines)
    # print(f"left segments:\n{left_segments}")
    # print(f"lines:\n{lines}")
    # print(f"right segments:\n{right_segments}")
    # left_points, right_points = get_segments(lines)
    draw_fused_line(left_segments, final_img)
    draw_fused_line(right_segments, final_img)
    # fit_and_draw(left_points, final_img, (0,255,0))
    # fit_and_draw(right_points, final_img, (0,255,0))

    left_segments = [tuple(int(val) for val in tpl) for tpl in left_segments]
    draw_lines(lines_img, left_segments, (255,0,0))
    right_segments = [tuple(int(val) for val in tpl) for tpl in right_segments]
    draw_lines(lines_img, right_segments, (0,0,255))
    # draw_circles(img_copy)

    for ax in axes.ravel():
        ax.cla()
        ax.axis("off")

    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,1].imshow(edges, cmap='gray')
    axes[1,0].imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB))
    axes[1,1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    fig.suptitle(img_path.name)
    fig.canvas.draw_idle()
    plt.waitforbuttonpress()

plt.close()

# print(type(lines))
# print(len(lines))
# print(lines)