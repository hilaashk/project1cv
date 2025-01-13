import numpy as np
import pandas as pd
import pygame
import cv2
from moviepy.editor import VideoFileClip
from moviepy import editor
import moviepy


def region_selection(image):
    """
    Define and mask the region of interest (ROI) in the input image.
    The ROI is a trapezoidal area focusing on lane regions and ignoring irrelevant areas.

    Args:
        image (ndarray): Input image.

    Returns:
        ndarray: Masked image with only the region of interest visible.
    """
    # Create a mask with the same dimensions as the input image
    mask = np.zeros_like(image)

    # Determine the color for the mask based on the number of channels in the image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # Multi-channel image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255  # Single-channel grayscale image

    # Define the vertices of the trapezoidal ROI
    rows, cols = image.shape[:2]
    crop_bottom = int(rows * 0.8)  # Bottom crop height
    bottom_left = [cols * 0.1, crop_bottom]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, crop_bottom]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Fill the ROI on the mask and apply it to the image
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    """
    Apply the Hough Line Transform to detect lines in an edge-detected image.

    Args:
        image (ndarray): Edge-detected input image.

    Returns:
        ndarray: Array of detected lines in the format [[x1, y1, x2, y2], ...].
    """
    # Define Hough Transform parameters
    rho = 2  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians
    threshold = 20  # Minimum number of votes to consider a line
    minLineLength = 50  # Minimum line segment length
    maxLineGap = 200  # Maximum gap between segments to treat them as one line

    # Apply the Hough Transform
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines):
    """
    Calculate the average slope and intercept for left and right lane lines.

    Args:
        lines (list): List of detected lines from the Hough Transform,
                      each represented as [[x1, y1, x2, y2], ...].

    Returns:
        tuple: (left_lane, right_lane), where each lane is represented by (slope, intercept).
    """
    if lines is None:
        return None, None

    left_fit = []  # Store slope and intercept of left lane lines
    right_fit = []  # Store slope and intercept of right lane lines

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Ignore lines with the same start and end points
                if x1 == x2 and y1 == y2:
                    continue

                # Calculate slope and intercept
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 1e6
                if abs(slope) < 0.1 or abs(slope) > 10:  # Filter out unrealistic slopes
                    continue

                intercept = y1 - slope * x1
                if abs(intercept) > 1e6:  # Filter out unrealistic intercepts
                    continue

                # Classify lines as left or right based on slope
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))

        # Require at least one valid line for each lane
        min_lines = 1
        if len(left_fit) < min_lines and len(right_fit) < min_lines:
            return None, None

        # Calculate average slope and intercept for each lane
        left_lane = np.mean(left_fit, axis=0) if len(left_fit) >= min_lines else None
        right_lane = np.mean(right_fit, axis=0) if len(right_fit) >= min_lines else None

        # Validate lanes for NaN values
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
    Convert the slope and intercept of a line into pixel coordinates.

    Args:
        y1 (float): Bottom y-coordinate of the line.
        y2 (float): Top y-coordinate of the line.
        line (tuple): Line parameters (slope, intercept).

    Returns:
        tuple or None: Pixel coordinates of the line ((x1, y1), (x2, y2))
                       or None if the line is invalid.
    """
    if line is None:
        return None

    slope, intercept = line

    try:
        # Handle cases with extremely large slopes (vertical lines)
        if abs(slope) > 1e6 or slope == float('inf') or slope == float('-inf'):
            x1 = int(intercept)
            x2 = int(intercept)
        else:
            # Calculate x-coordinates using the line equation y = mx + b
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

        # Filter out unrealistic x-coordinates
        if abs(x1) > 1e6 or abs(x2) > 1e6:
            return None

        return ((x1, int(y1)), (x2, int(y2)))

    except Exception as e:
        print(f"Error in pixel_points: {str(e)}")
        return None

def fill_lane_area(image, lines, color=[0, 255, 0], thickness=-1):
    """
    Fill the area between the detected left and right lane lines.

    Args:
        image (ndarray): Input image on which the lane area will be filled.
        lines (tuple): Detected lane lines as (left_line, right_line), where each line is
                       represented by two points ((x1, y1), (x2, y2)).
        color (list): Color to fill the lane area. Default is green ([0, 255, 0]).
        thickness (int): Thickness of the filled area. Default is -1 (fill completely).

    Returns:
        ndarray: Image with the filled lane area overlayed.
    """
    try:
        # Check if valid lane lines are provided
        if lines is None or None in lines:
            return image

        left_line, right_line = lines
        if left_line is None or right_line is None:
            return image

        # Create an empty overlay image
        line_image = np.zeros_like(image)
        polygon_points = []

        try:
            # Define points to create the polygon connecting both lane lines
            if left_line and right_line:
                polygon_points = np.array([
                    [left_line[0], left_line[1], right_line[1], right_line[0]]
                ], dtype=np.int32)
        except Exception:
            return image

        # Fill the polygon area if valid points are available
        if len(polygon_points) > 0:
            cv2.fillPoly(line_image, polygon_points, color)
            # Blend the filled polygon with the original image
            return cv2.addWeighted(image, 1.0, line_image, 0.5, 0.0)
        return image

    except Exception as e:
        # Handle exceptions gracefully and print the error message
        print(f"Error in fill_lane_area: {str(e)}")
        return image


def temporal_smoothing(current_lines, previous_lines, smoothing_factor=0.8):
    """
    Smooth lane lines over time to reduce jitter and blinking.

    Args:
        current_lines (tuple): Current detected lane lines as (left_line, right_line).
        previous_lines (tuple): Previously detected lane lines.
        smoothing_factor (float): Weight factor for smoothing,
                                  where higher values prioritize previous positions.

    Returns:
        tuple: Smoothed lane lines as (left_line, right_line).
    """
    # Handle cases where previous or current lines are not available
    if previous_lines is None or None in previous_lines:
        return current_lines
    if current_lines is None or None in current_lines:
        return previous_lines

    left_current, right_current = current_lines
    left_prev, right_prev = previous_lines

    if left_current and left_prev and right_current and right_prev:
        # Smooth the left lane
        left_smoothed = (
            (int((1 - smoothing_factor) * left_current[0][0] + smoothing_factor * left_prev[0][0]),
             int((1 - smoothing_factor) * left_current[0][1] + smoothing_factor * left_prev[0][1])),
            (int((1 - smoothing_factor) * left_current[1][0] + smoothing_factor * left_prev[1][0]),
             int((1 - smoothing_factor) * left_current[1][1] + smoothing_factor * left_prev[1][1]))
        )

        # Smooth the right lane
        right_smoothed = (
            (int((1 - smoothing_factor) * right_current[0][0] + smoothing_factor * right_prev[0][0]),
             int((1 - smoothing_factor) * right_current[0][1] + smoothing_factor * right_prev[0][1])),
            (int((1 - smoothing_factor) * right_current[1][0] + smoothing_factor * right_prev[1][0]),
             int((1 - smoothing_factor) * right_current[1][1] + smoothing_factor * right_prev[1][1]))
        )

        return left_smoothed, right_smoothed

    # Return current lines if smoothing is not applicable
    return current_lines


def detect_lane_departure(left_line, right_line, frame_width, prev_status=None):
    """
    Detect if the vehicle is departing from its lane based on the lane center.

    Args:
        left_line (tuple): Detected left lane line as ((x1, y1), (x2, y2)).
        right_line (tuple): Detected right lane line as ((x1, y1), (x2, y2)).
        frame_width (int): Width of the video frame.
        prev_status (str, optional): Previous detection status. Default is None.

    Returns:
        str or None: "LANE CHANGE DETECTED" if a significant departure is detected, otherwise None.
    """
    # Ensure both lane lines are detected
    if left_line is None or right_line is None:
        return None

    # Calculate the center point of the lane at the bottom of the frame
    left_bottom_x = left_line[0][0]
    right_bottom_x = right_line[0][0]
    lane_center = (left_bottom_x + right_bottom_x) // 2

    # Determine the frame center
    frame_center = frame_width // 2

    # Define a threshold for detecting significant deviation
    threshold = frame_width * 0.18  # 18% of the frame width

    # Calculate the deviation from the lane center to the frame center
    deviation = lane_center - frame_center

    # Detect lane change if the deviation exceeds the threshold
    current_status = None
    if deviation > threshold or deviation < -threshold:
        current_status = "LANE CHANGE DETECTED"

    return current_status


def lane_lines(image, lines):
    """
    Generate full-length lane lines from detected Hough lines.

    Args:
        image (ndarray): Input image for determining line bounds.
        lines (list): Detected Hough lines.

    Returns:
        tuple: (left_line, right_line) where each line is represented as ((x1, y1), (x2, y2)).
    """
    # Calculate the average slope and intercept for left and right lanes
    left_lane, right_lane = average_slope_intercept(lines)

    # Define the y-coordinates for the lines
    y1 = image.shape[0]  # Bottom of the frame
    y2 = y1 * 0.6  # 60% of the frame height

    # Convert the slope and intercept to pixel points
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line

def frame_processor(image):
    """
    Process a single video frame for lane detection.

    This function applies several preprocessing steps, including grayscale conversion,
    Gaussian blurring, Canny edge detection, and region masking, followed by Hough
    line transformation to detect lanes. It overlays detected lane areas on the input frame.

    Args:
        image (ndarray): Input video frame (BGR format).

    Returns:
        ndarray: Processed video frame with detected lanes overlayed.
    """
    try:
        # Convert the image to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and smooth edges
        kernel_size = 5
        blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

        # Perform Canny edge detection to extract edges
        low_t = 50  # Lower threshold for edge detection
        high_t = 150  # Upper threshold for edge detection
        edges = cv2.Canny(blur, low_t, high_t)

        # Define and apply region of interest mask
        region = region_selection(edges)

        # Apply Hough transform to detect lines
        hough = hough_transform(region)

        # If no lines are detected, return the original image
        if hough is None:
            return image

        # Extract left and right lane lines from detected lines
        left_line, right_line = lane_lines(image, hough)

        # If either lane line is not detected, return the original image
        if left_line is None or right_line is None:
            return image

        # Fill the lane area between detected lines and overlay it on the original frame
        result = fill_lane_area(image, lane_lines(image, hough))
        return result

    except Exception as e:
        # Handle any errors gracefully and return the original image
        print(f"Error in frame processing: {str(e)}")
        return image


def process_video(test_video, output_video):
    """
    Process a video file for lane detection and lane change warnings.

    This function processes each frame of the input video to detect lanes and
    lane changes. It applies temporal smoothing for stability and overlays
    warning messages on the output video when lane departures are detected.

    Args:
        test_video (str): Path to the input video file.
        output_video (str): Path to save the processed video file.

    Returns:
        None
    """
    previous_lines = None  # Store smoothed lines from the previous frame

    def process_frame_with_smoothing(frame):
        """
        Process a single frame with temporal smoothing and lane departure detection.

        Args:
            frame (ndarray): Input video frame.

        Returns:
            ndarray: Processed frame with detected lanes and warnings overlayed.
        """
        nonlocal previous_lines

        # Process the frame for lane detection
        result = frame_processor(frame)

        # Detect current lane lines
        current_lines = lane_lines(frame, hough_transform(
            region_selection(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150))
        ))

        # Apply temporal smoothing to stabilize lane line detection
        smoothed_lines = temporal_smoothing(current_lines, previous_lines)
        previous_lines = smoothed_lines  # Update previous lines for the next frame

        # Overlay smoothed lane areas on the frame
        if smoothed_lines[0] and smoothed_lines[1]:
            result = fill_lane_area(frame, smoothed_lines)

            # Detect lane departure and display warnings if detected
            left_line, right_line = smoothed_lines
            current_status = detect_lane_departure(left_line, right_line, frame.shape[1], None)

            if current_status:
                # Create a semi-transparent overlay for the warning
                overlay = result.copy()

                # Draw a filled rectangle as a background for the warning text
                cv2.rectangle(overlay, (20, 20), (600, 100), (0, 0, 0), -1)

                # Add warning text to the overlay
                cv2.putText(overlay, current_status,
                            (30, 70),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.5,
                            (0, 0, 255),  # Red color for warning text
                            3)

                # Blend the overlay with the original frame
                result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

        return result

    # Open the input video file
    input_video = editor.VideoFileClip(test_video, audio=False)

    # Process each frame using the frame processing function
    processed = input_video.fl_image(process_frame_with_smoothing)

    # Write the processed video to the output file
    processed.write_videofile(output_video, fps=input_video.fps, audio=False)


# Main execution
if __name__ == "__main__":
    # Replace with the path to your input video file
    process_video("video/highwayDriveVid.mp4", 'results/highwayDriveResult.mp4')
