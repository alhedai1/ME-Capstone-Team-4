from dataclasses import dataclass

import cv2
import numpy as np
import math

@dataclass
class BellCircleDetection:
    x: int
    y: int
    radius: int

    @property
    def circle(self):
        return self.x, self.y, self.radius


class BellCircle:
    def __init__(
        self,
        color_format="rgb",
        dp=1.5,
        min_dist=60,
        param1=100,
        param2=20,
        min_radius=10,
        max_radius=30,
        stability_threshold=3,  # Minimum frames to consider a detection stable
        max_distance=20,       # Maximum pixels a stable circle can move between frames
        max_radius_diff=10,
        startup_max_radius=30,
        tracking_max_radius=120,
        radius_search_margin=25,
        lost_after_frames=8,
        startup_confirm_threshold=2,
        show_debug=False,
    ):
        self.color_format = color_format
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.last_circle = None
        self.stable_frames = 0
        self.stability_threshold = stability_threshold
        self.max_distance = max_distance
        self.max_radius_diff = max_radius_diff
        self.startup_max_radius = startup_max_radius
        self.tracking_max_radius = tracking_max_radius  
        self.radius_search_margin = radius_search_margin
        self.lost_after_frames = lost_after_frames
        self.missed_frames = 0
        self.started = False
        self.last_known_radius = None
        self.min_reacquire_radius = None
        self.reacquire_candidate = None
        self.reacquire_frames = 0
        self.reacquire_threshold = 3
        self.startup_confirm_threshold = startup_confirm_threshold
        self.show_debug = show_debug

    def detect(self, frame):
        gray = self._gray(frame)
        # cv2.imshow("gray", gray)
        # mblur = cv2.medianBlur(gray, 5)
        # cv2.imshow("mblur", mblur)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        # cv2.imshow("gblur", blur)
        # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        
        ### set houghcircles min/max radius 
        if self.last_circle is not None:
            min_radius = max(self.min_radius, self.last_circle.radius - self.radius_search_margin)
            max_radius = min(self.tracking_max_radius, self.last_circle.radius + self.radius_search_margin)
        elif not self.started:
            # true startup only
            min_radius = self.min_radius
            max_radius = self.startup_max_radius
        else:
            # lost after already seeing the bell: do NOT go back to tiny startup search
            # min_radius = max(self.min_radius, int(self.last_known_radius * 0.8))
            min_radius = self.min_radius
            max_radius = self.tracking_max_radius
        
        ### get circles
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        if circles is None:
            self.missed_frames += 1
            self.stable_frames = max(0, self.stable_frames - 1)

            if self.missed_frames >= self.lost_after_frames:
                print("Lost circle for too long, resetting last_circle")
                self.last_circle = None
                self.stable_frames = 0
                self.reacquire_candidate = None
                self.reacquire_frames = 0
                return None
            print("No circle detected at all, holding last_circle")
            return self.last_circle

        circles = np.round(circles[0]).astype(int)

        if self.show_debug:
            vis = frame.copy()
            for c in circles:
                x, y, r = c
                cv2.circle(vis, (x, y), r, (0, 0, 0), 1)
            cv2.imshow("vis", vis)

        # --- MINIMAL TRACKING LOGIC START ---
        valid_circles = []
        
        for c in circles:
            x, y, r = c
            if x < 0.2*frame.shape[1] or x > 0.8*frame.shape[1]:
                continue
            # on startup, consider all circles
            if self.last_circle is None:
                valid_circles.append(c)
                continue
            # otherwise only consider circles with close center & radius
            dist = math.hypot(x - self.last_circle.x, y - self.last_circle.y)
            rad_diff = abs(r - self.last_circle.radius)

            if dist <= self.max_distance and rad_diff <= self.max_radius_diff:
                valid_circles.append(c)

        # If all candidates were rejected as false positives, use history or return None
        if not valid_circles:
            # self.stable_frames = max(0, self.stable_frames - 1)
            # # return self.last_circle # Or return None if you want to skip frames
            # return None
            self.missed_frames += 1
            self.stable_frames = max(0, self.stable_frames - 1)

            if self.missed_frames >= self.lost_after_frames:
                print("Lost circle for too long, resetting last_circle")
                self.last_circle = None
                self.stable_frames = 0
                self.reacquire_candidate = None
                self.reacquire_frames = 0
                return None
            print("No valid circles, holding last_circle")
            return self.last_circle

        # On startup/reacquire, require a consistent candidate before locking.
        if self.last_circle is None:
            x, y, radius = max(valid_circles, key=lambda c: c[2])
            candidate = BellCircleDetection(int(x), int(y), int(radius))
            confirm_threshold = (
                self.startup_confirm_threshold
                if not self.started
                else self.reacquire_threshold
            )

            if self.reacquire_candidate is not None:
                dist = math.hypot(
                    candidate.x - self.reacquire_candidate.x,
                    candidate.y - self.reacquire_candidate.y,
                )
                rad_diff = abs(candidate.radius - self.reacquire_candidate.radius)

                if dist <= self.max_distance * 2 and rad_diff <= self.max_radius_diff * 2:
                    self.reacquire_frames += 1
                else:
                    self.reacquire_frames = 1
            else:
                self.reacquire_frames = 1

            self.reacquire_candidate = candidate

            if self.reacquire_frames < confirm_threshold:
                return None
        # otherwise choose circle closest to last in center & radius
        else:
            x, y, radius = min(
                valid_circles,
                key=lambda c: (
                    math.hypot(c[0] - self.last_circle.x, c[1] - self.last_circle.y),
                    abs(c[2] - self.last_circle.radius),
                ),
            )

        detection = BellCircleDetection(int(x), int(y), int(radius))

        # Check if the new detection matches the previous frame's temporary position
        if self.last_circle is not None:
            dist = math.hypot(detection.x - self.last_circle.x, detection.y - self.last_circle.y)
            rad_diff = abs(detection.radius - self.last_circle.radius)
            if dist <= self.max_distance and rad_diff <= self.max_radius_diff:
                self.stable_frames += 1
                # print(self.stable_frames)
            else:
                self.stable_frames = 1 # Reset tracking window if it suddenly jumped
                # print(self.stable_frames)
        else:
            self.stable_frames = 1

        self.missed_frames = 0
        self.last_circle = detection
        self.started = True
        self.last_known_radius = detection.radius
        self.reacquire_candidate = None
        self.reacquire_frames = 0
        return detection
        # --- MINIMAL TRACKING LOGIC END ---
        
        return BellCircleDetection(int(x), int(y), int(radius))

    def _gray(self, frame):
        if self.color_format == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.color_format == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported color format: {self.color_format}")


def detect_bell_circle(frame, color_format="rgb"):
    return BellCircle(color_format=color_format).detect(frame)
