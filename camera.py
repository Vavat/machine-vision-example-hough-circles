import math
import os
import subprocess

import cv2 as cv
import numpy as np


class Camera:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = current_dir + "/images/"

    def __init__(self, image_path=local_path):
        self.path = image_path

    def find_circles(
        self,
        image: str,
        min_diameter: float,
        max_diameter: float,
        min_dist: float,
        param1: int,
        param2: int,
        distance_mm_from_camera: float,
        method_dp: float = 1.5,
    ) -> tuple:
        """_summary_

        Args:
            image (string): name of image to be analysed
            min_diameter (float): minimum diameter of circle to be found in mm
            min_diameter (float): maximum diameter of circle to be found in mm
            min_dist (float): minimum distance between circles in mm
            params1 (int): Hough circle detection parameter 1 (see OpenCV docs)
            params2 (int): Hough circle detection parameter 2 (see OpenCV docs)
            distance_mm_from_camera (float): distance from camera to object in mm
            method_dp (float, optional):  Defaults to 1.5.

        Returns:
            tuple: (x_dist_mm, y_dist_mm, center, is_found)
            where x_dist_mm and y_dist_mm are the distance in mm from the center of the image to the center of the nearest circle.
            Center List contains the center coordinates of all circles found.
            is_found is a boolean indicating whether any circles were found.
        """
        # read the image path + image name
        img = cv.imread(self.path + image + ".jpg")

        # check if image was read
        if img is None:
            print(f"Could not read the image. {self.path + image + '.jpg'}")
            return 0, 0, 0, False

        # Add guassian blur to the image
        # img = cv.GaussianBlur(img, (9, 9), 0)

        # Add median blur to the image
        img = cv.medianBlur(img, 9)

        # convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # find the circles in the image
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            method_dp,
            int(min_dist * (self.calculate_pixel_to_mm(distance_mm_from_camera))),
            param1=param1,
            param2=param2,
            minRadius=int(min_diameter * (self.calculate_pixel_to_mm(distance_mm_from_camera)) * 0.5),
            maxRadius=int(max_diameter * (self.calculate_pixel_to_mm(distance_mm_from_camera)) * 0.5),
        )

        # Overlay circles on image and find center pixel, highlight center pixel in cyan
        if circles is not None:
            # record center of circle in pixels and radius
            center = []
            radius = []
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for x, y, r in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv.circle(img, (x, y), r, (0, 255, 0), 4)
                cv.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                center.append((x, y))
                radius.append(r)

                # label the circle with the radius and center coordinates
                cv.putText(
                    img,
                    f"radius: {r}",
                    (x - 50, y - 50),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv.putText(
                    img,
                    f"center: {x, y}",
                    (x - 50, y - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        else:
            print("No circles found")
            return 0, 0, 0, False

        # find the center of the image
        image_center = (int(img.shape[1] / 2), int(img.shape[0] / 2))

        # draw a circle in the center of the image
        cv.circle(img, image_center, 5, (0, 0, 255), -1)
        # add label to center of image with coordinates
        cv.putText(
            img,
            f"center: {image_center}",
            (image_center[0] - 50, image_center[1] - 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # draw a straight line from the center of the image to the center of the nearest circle
        # find the nearest circle
        nearest_circle = min(
            center,
            key=lambda x: math.hypot(x[0] - image_center[0], x[1] - image_center[1]),
        )
        # draw the line
        cv.line(img, image_center, nearest_circle, (0, 0, 255), 2)

        # calculate the x and y distance between the center of the image and the center of the circle
        x_dist = nearest_circle[0] - image_center[0]
        y_dist = nearest_circle[1] - image_center[1]

        # calculate the angle between the center of the image and the center of the circle
        angle = math.atan2(y_dist, x_dist) * 180 / math.pi

        # draw the x and y distance and angle on the image
        cv.putText(
            img,
            f"x_dist: {x_dist}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv.putText(
            img,
            f"y_dist: {y_dist}",
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv.putText(img, f"angle: {angle}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        x_dist_mm = x_dist / self.calculate_pixel_to_mm(distance_mm_from_camera)
        y_dist_mm = y_dist / self.calculate_pixel_to_mm(distance_mm_from_camera)

        # draw the x and y distance in mm on the image
        cv.putText(
            img,
            f"x_dist_mm: {x_dist_mm}",
            (10, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv.putText(
            img,
            f"y_dist_mm: {y_dist_mm}",
            (10, 150),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        name = f"{self.path}/{image}_circles.jpg"
        cv.imwrite(name, img, [int(cv.IMWRITE_JPEG_QUALITY), 25])

        return x_dist_mm, y_dist_mm, center, True

    def convert_pixel_to_mm(self, x_dist: int, y_dist: int, pixel_to_mm: float) -> tuple:
        """Convert pixel distance to mm distance."""
        x_dist_mm = x_dist / pixel_to_mm
        y_dist_mm = y_dist / pixel_to_mm
        return x_dist_mm, y_dist_mm

    def calculate_pixel_to_mm(self, focus_distance_mm: float) -> float:
        # focus_distance_mm in mm is the distance from the camera to the object

        constant_pixel_mm = 21.0  # Calibrated at 173mm away from the camera using 50mm lid
        calibration_height_mm = 173  # height of the camera when calibrated
        calibration_factor = -0.147  # calibration factor

        # using Y = mx + b
        pixel_mm_converter = ((focus_distance_mm - calibration_height_mm) * calibration_factor) + constant_pixel_mm
        return pixel_mm_converter
