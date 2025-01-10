import numpy as np
import pandas as pd
import pygame
import cv2
from moviepy.editor import VideoFileClip
from moviepy import editor
import moviepy


def region_selection(image):
    """
	Determine and cut the region of interest in the input image.
	Parameters:
		image: we pass here the output from canny where we have
		identified edges in the frame
	"""
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    # if you pass an image with more then one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # our image only has one channel so it will go under "else"
    else:
        # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    rows, cols = image.shape[:2]
    crop_bottom = int(rows * 0.8)
    bottom_left = [cols * 0.1, crop_bottom]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, crop_bottom]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    """
	Determine and cut the region of interest in the input image.
	Parameter:
		image: grayscale image which should be an output from the edge detector
	"""
    # Distance resolution of the accumulator in pixels.
    rho = 2
    # Angle resolution of the accumulator in radians.
    theta = np.pi / 180
    # Only lines that are greater than threshold will be returned.
    threshold = 20
    # Line segments shorter than that are rejected.
    minLineLength = 50
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 200
    # function returns an array containing dimensions of straight lines
    # appearing in the input image
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes.
    Added robust error handling and validation.
    """
    if lines is None:
        return None, None

    left_fit = []
    right_fit = []

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip if points are identical to avoid division by zero
                if x1 == x2 and y1 == y2:
                    continue

                if x2 == x1:
                    # Vertical line case
                    slope = 1e6 if y2 > y1 else -1e6
                else:
                    slope = (y2 - y1) / (x2 - x1)

                # Filter out horizontal lines and extreme slopes
                if abs(slope) < 0.1 or abs(slope) > 10:
                    continue

                intercept = y1 - slope * x1

                # Validate intercept is within reasonable bounds
                if abs(intercept) > 1e6:
                    continue

                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

        # Require minimum number of lines for valid detection
        min_lines = 1
        if len(left_fit) < min_lines and len(right_fit) < min_lines:
            return None, None

        # Calculate average slopes and intercepts
        left_lane = np.mean(left_fit, axis=0) if len(left_fit) >= min_lines else None
        right_lane = np.mean(right_fit, axis=0) if len(right_fit) >= min_lines else None

        # Validate final results
        if left_lane is not None and (np.isnan(left_lane[0]) or np.isnan(left_lane[1])):
            left_lane = None
        if right_lane is not None and (np.isnan(right_lane[0]) or np.isnan(right_lane[1])):
            right_lane = None

        return left_lane, right_lane

    except Exception as e:
        print(f"Error in average_slope_intercept: {str(e)}")
        return None, None


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    Added robust error handling for vertical lines and edge cases.
    """
    if line is None:
        return None

    slope, intercept = line

    try:
        # Handle vertical or near-vertical lines
        if abs(slope) > 1e6 or slope == float('inf') or slope == float('-inf'):
            # For vertical lines, use intercept as x-coordinate
            x1 = int(intercept)
            x2 = int(intercept)
        else:
            # Normal case - calculate points using slope-intercept form
            x1 = int((y1 - intercept) / slope) if slope != 0 else int(intercept)
            x2 = int((y2 - intercept) / slope) if slope != 0 else int(intercept)

        # Validate points are within reasonable bounds
        if abs(x1) > 1e6 or abs(x2) > 1e6:
            return None

        return ((x1, int(y1)), (x2, int(y2)))
    except (ZeroDivisionError, OverflowError, ValueError):
        return None


def fill_lane_area(image, lines, color=[0, 255, 0], thickness=-1):
    """
    Fill the area between the left and right lane lines with improved error handling.
    """
    try:
        if lines is None or None in lines:
            return image

        left_line, right_line = lines
        if left_line is None or right_line is None:
            return image

        line_image = np.zeros_like(image)

        # Create polygon points
        polygon_points = []
        try:
            if left_line and right_line:
                polygon_points = np.array([
                    [left_line[0], left_line[1], right_line[1], right_line[0]]
                ], dtype=np.int32)
        except Exception:
            return image

        if len(polygon_points) > 0:
            cv2.fillPoly(line_image, polygon_points, color)
            return cv2.addWeighted(image, 1.0, line_image, 0.5, 0.0)
        return image

    except Exception as e:
        print(f"Error in fill_lane_area: {str(e)}")
        return image


def temporal_smoothing(current_lines, previous_lines, smoothing_factor=0.8):
    """
    Apply temporal smoothing to reduce lane blinking.
    """
    if previous_lines is None or None in previous_lines:
        return current_lines

    if current_lines is None or None in current_lines:
        return previous_lines

    left_current, right_current = current_lines
    left_prev, right_prev = previous_lines

    if left_current and left_prev and right_current and right_prev:
        # Smooth the points
        left_smoothed = (
            (int((1 - smoothing_factor) * left_current[0][0] + smoothing_factor * left_prev[0][0]),
             int((1 - smoothing_factor) * left_current[0][1] + smoothing_factor * left_prev[0][1])),
            (int((1 - smoothing_factor) * left_current[1][0] + smoothing_factor * left_prev[1][0]),
             int((1 - smoothing_factor) * left_current[1][1] + smoothing_factor * left_prev[1][1]))
        )

        right_smoothed = (
            (int((1 - smoothing_factor) * right_current[0][0] + smoothing_factor * right_prev[0][0]),
             int((1 - smoothing_factor) * right_current[0][1] + smoothing_factor * right_prev[0][1])),
            (int((1 - smoothing_factor) * right_current[1][0] + smoothing_factor * right_prev[1][0]),
             int((1 - smoothing_factor) * right_current[1][1] + smoothing_factor * right_prev[1][1]))
        )

        return left_smoothed, right_smoothed

    return current_lines

def lane_lines(image, lines):
    """
    Modified lane_lines function to include prediction
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    if left_line and right_line:  # Only process if both lines are detected
        return left_line, right_line
    return None, None


def frame_processor(image):
    """
    Process the input frame with improved error handling.
    """
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = 5
        blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
        low_t = 50
        high_t = 150
        edges = cv2.Canny(blur, low_t, high_t)
        region = region_selection(edges)
        hough = hough_transform(region)

        if hough is None:
            return image

        left_line, right_line = lane_lines(image, hough)

        if left_line is None or right_line is None:
            return image

        # Fill the base lane area
        result = fill_lane_area(image, lane_lines(image, hough))

        return result
    except Exception as e:
        print(f"Error in frame processing: {str(e)}")
        return image  # Return original frame if processing fails

# driver function
def process_video(test_video, output_video):
    previous_lines = None  # Add this line

    def process_frame_with_smoothing(frame):
        nonlocal previous_lines  # Add this line

        result = frame_processor(frame)
        current_lines = lane_lines(frame, hough_transform(
            region_selection(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150))))

        # Apply temporal smoothing
        smoothed_lines = temporal_smoothing(current_lines, previous_lines)
        previous_lines = smoothed_lines

        # Draw smoothed lines
        if smoothed_lines[0] and smoothed_lines[1]:
            result = fill_lane_area(frame, smoothed_lines)

        return result

    input_video = editor.VideoFileClip(test_video, audio=False)
    processed = input_video.fl_image(process_frame_with_smoothing)
    processed.preview(fps=input_video.fps, audio=False)


# calling driver function
process_video(r"C:\Users\User\Desktop\Computer vision\CV-lane-detection\CV_video.mp4", 'output.mp4')