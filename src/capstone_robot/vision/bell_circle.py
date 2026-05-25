from dataclasses import dataclass

import cv2
import numpy as np


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
        dp=1,
        min_dist=60,
        param1=300,
        param2=20,
        min_radius=10,
        max_radius=30,
    ):
        self.color_format = color_format
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, frame):
        gray = self._gray(frame)
        blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)
        x, y, radius = max(circles, key=lambda circle: circle[2])
        return BellCircleDetection(int(x), int(y), int(radius))

    def _gray(self, frame):
        if self.color_format == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.color_format == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Unsupported color format: {self.color_format}")


def detect_bell_circle(frame, color_format="rgb"):
    return BellCircle(color_format=color_format).detect(frame)
