import numpy as np
import pandas as pd
import pygame
import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy import editor
import moviepy


def region_selection(image):
    """
    Determines and extracts the region of interest (ROI) in the input image.

    Parameters:
        image (numpy.ndarray): The input image, typically the output of a Canny edge detection
                               process, where edges in the frame have been identified.

    Returns:
        numpy.ndarray: A masked image where only the region of interest is preserved,
                       and the rest is blacked out.
    """
    # Create a mask with the same dimensions as the input image, initialized to zeros (black).
    mask = np.zeros_like(image)

    # Determine the mask color based on the number of channels in the image.
    if len(image.shape) > 2:  # For multi-channel images (e.g., RGB)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count  # White color for all channels
    else:  # For single-channel images (e.g., grayscale)
        ignore_mask_color = 255  # White color

    # Define a polygon to focus only on the road area in the image.
    # The polygon is based on the camera's position and expected road boundaries.
    rows, cols = image.shape[:2]
    crop_bottom = int(rows * 0.90)  # Bottom crop to exclude irrelevant areas
    bottom_left = [cols * 0.1, crop_bottom]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, crop_bottom]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Fill the polygon region in the mask with white color.
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Apply the mask to the input image using a bitwise AND operation
    # to retain only the edges within the ROI.
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def hough_transform(image):
    """
    Applies a resolution-aware probabilistic Hough Line Transform to detect lines in an edge-detected image.

    Parameters:
        image (numpy.ndarray): The input image, typically a binary edge-detected image (e.g., output from Canny).

    Returns:
        numpy.ndarray: An array of lines detected by the Hough Transform, where each line is represented
                       by its endpoints [x1, y1, x2, y2]. Returns None if no lines are detected.
    """
    # Get the dimensions of the input image.
    rows, cols = image.shape[:2]

    # Set Hough Transform parameters based on image resolution.
    if rows <= 720:  # For lower resolution (e.g., 720p)
        minLineLength = 30  # Minimum length of a line to be detected
        maxLineGap = 100  # Maximum allowed gap between line segments to treat them as a single line
        threshold = 15  # Minimum number of intersections in Hough space to detect a line
    else:  # For higher resolution (e.g., 1080p)
        minLineLength = 50
        maxLineGap = 200
        threshold = 20

    # Define Hough Transform parameters.
    rho = 2  # Distance resolution of the accumulator in pixels
    theta = np.pi / 180  # Angle resolution of the accumulator in radians

    # Perform the Hough Line Transform.
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines):
    """
    Calculates the average slope and intercept for the left and right lanes
    from a collection of detected lines.

    Parameters:
        lines (numpy.ndarray): An array of detected lines, where each line is
                               represented as [[x1, y1, x2, y2]]. Can be None
                               if no lines are detected.

    Returns:
        tuple: A tuple containing:
            - left_lane (tuple or None): Median slope and intercept for the left lane,
                                         or None if insufficient lines are found.
            - right_lane (tuple or None): Median slope and intercept for the right lane,
                                          or None if insufficient lines are found.
    """
    if lines is None:  # Return early if no lines are detected
        return None, None

    left_fit = []  # Store slope and intercept for left lane lines
    right_fit = []  # Store slope and intercept for right lane lines

    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Skip identical points or vertical lines
                if x1 == x2 or (x1 == x2 and y1 == y2):
                    continue

                # Calculate the slope and intercept of the line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Classify the line based on its slope
                # Tighter constraints to prevent invalid or crossing lanes
                if -0.85 < slope < -0.5:  # Left lane criteria
                    left_fit.append((slope, intercept))
                elif 0.5 < slope < 0.85:  # Right lane criteria
                    right_fit.append((slope, intercept))

        # Require a minimum number of valid lines for each lane
        min_lines = 1
        if len(left_fit) < min_lines or len(right_fit) < min_lines:
            return None, None

        # Calculate the median slope and intercept for each lane
        left_lane = np.median(left_fit, axis=0) if len(left_fit) >= min_lines else None
        right_lane = np.median(right_fit, axis=0) if len(right_fit) >= min_lines else None

        return left_lane, right_lane

    except Exception as e:
        # Log any errors encountered during processing
        print(f"Error in average_slope_intercept: {str(e)}")
        return None, None


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of a line into pixel coordinates for drawing.

    Parameters:
        y1 (int): The y-coordinate of the first point.
        y2 (int): The y-coordinate of the second point.
        line (tuple): A tuple containing the slope and intercept of the line
                      (slope, intercept). Can be None if the line is invalid.

    Returns:
        tuple: A tuple of two points ((x1, y1), (x2, y2)) representing the pixel
               coordinates of the line. Returns None if the line is invalid or
               cannot be processed.
    """
    if line is None:  # Return early if the line is invalid
        return None

    slope, intercept = line

    try:
        # Handle vertical or near-vertical lines
        if abs(slope) > 1e6 or slope in [float('inf'), float('-inf')]:
            # For vertical lines, use the intercept as the x-coordinate
            x1 = int(intercept)
            x2 = int(intercept)
        else:
            # Standard case - calculate x-coordinates using the slope-intercept formula
            x1 = int((y1 - intercept) / slope) if slope != 0 else int(intercept)
            x2 = int((y2 - intercept) / slope) if slope != 0 else int(intercept)

        # Validate points to ensure they are within reasonable bounds
        if abs(x1) > 1e6 or abs(x2) > 1e6:
            return None

        # Return the calculated pixel points
        return ((x1, int(y1)), (x2, int(y2)))

    except (ZeroDivisionError, OverflowError, ValueError):
        # Handle exceptions such as division by zero, overflow, or invalid values
        return None


def fill_lane_area(image, lines, color=[0, 255, 0], thickness=-1):
    """
    Fills the area between two lane lines on an image.

    Parameters:
        image (numpy.ndarray): The input image where the lane area will be filled.
        lines (tuple): A tuple containing two lines (left_line, right_line), where
                       each line is represented as a pair of points [(x1, y1), (x2, y2)].
        color (list, optional): The color to fill the lane area, specified as a BGR list.
                                Defaults to green [0, 255, 0].
        thickness (int, optional): Thickness of the fill. A value of -1 fills the polygon.
                                   Defaults to -1.

    Returns:
        numpy.ndarray: The input image with the filled lane area. If lines are invalid or
                       an error occurs, the original image is returned.
    """
    try:
        # Validate input lines
        if lines is None or None in lines:
            return image

        left_line, right_line = lines
        if left_line is None or right_line is None:
            return image

        # Create an empty image for drawing the polygon
        line_image = np.zeros_like(image)

        # Define polygon points based on the lane lines
        polygon_points = []
        try:
            if left_line and right_line:
                polygon_points = np.array([
                    [left_line[0], left_line[1], right_line[1], right_line[0]]
                ], dtype=np.int32)
        except Exception:
            return image

        # Fill the polygon if points are defined
        if len(polygon_points) > 0:
            cv2.fillPoly(line_image, polygon_points, color)
            # Blend the filled polygon with the original image
            return cv2.addWeighted(image, 1.0, line_image, 0.5, 0.0)

        # Return the original image if no valid polygon points
        return image

    except Exception as e:
        # Log any unexpected errors
        print(f"Error in fill_lane_area: {str(e)}")
        return image


def lane_lines(image, lines):
    """
    Calculates the lane lines in an image and predicts their pixel points.

    Parameters:
        image (numpy.ndarray): The input image where lane lines are detected.
        lines (list or np.ndarray): Detected lines represented as slopes and intercepts
                                    from edge detection and Hough transform.

    Returns:
        tuple: A tuple containing:
            - left_line (tuple or None): Pixel points representing the left lane line
                                         as ((x1, y1), (x2, y2)). Returns None if not detected.
            - right_line (tuple or None): Pixel points representing the right lane line
                                          as ((x1, y1), (x2, y2)). Returns None if not detected.
    """
    # Get the average slope and intercept for left and right lane lines
    left_lane, right_lane = average_slope_intercept(lines)

    # Define the start (y1) and end (y2) y-coordinates for the lane lines
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * 0.68)  # A point 68% up the image height

    # Calculate pixel points for the left and right lane lines
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    # Only return valid lines if both are detected
    if left_line and right_line:
        return left_line, right_line

    # Return None for both if either line is not detected
    return None, None


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Applies a bilateral filter to the input image for noise reduction while preserving edges.

    Parameters:
        image (numpy.ndarray): The input image to which the filter will be applied.
        d (int, optional): Diameter of each pixel neighborhood used in the filter.
                           Defaults to 9.
        sigma_color (float, optional): Filter sigma in the color space. A larger value
                                       means more distant colors will be mixed together,
                                       resulting in stronger smoothing. Defaults to 75.
        sigma_space (float, optional): Filter sigma in the coordinate space. A larger value
                                       means that pixels farther apart will influence each
                                       other as long as their colors are similar. Defaults to 75.

    Returns:
        numpy.ndarray: The filtered image with reduced noise and preserved edges.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def temporal_smoothing(current_lines, previous_lines, smoothing_factor=0.8):
    """
    Applies temporal smoothing to lane lines to reduce flickering in detections
    across consecutive frames.

    Parameters:
        current_lines (tuple): The current frame's detected lane lines, represented
                               as a tuple of two lines (left_line, right_line). Each
                               line is defined by its endpoints ((x1, y1), (x2, y2)).
        previous_lines (tuple): The previous frame's detected lane lines, represented
                                in the same format as `current_lines`.
        smoothing_factor (float, optional): A factor controlling the influence of the
                                            previous lines on the smoothed result.
                                            Must be in the range [0, 1]. Defaults to 0.8.

    Returns:
        tuple: A tuple containing:
            - left_smoothed (tuple): Smoothed pixel points for the left lane line
                                     as ((x1, y1), (x2, y2)).
            - right_smoothed (tuple): Smoothed pixel points for the right lane line
                                      as ((x1, y1), (x2, y2)).
            If no smoothing is applied due to missing lines, returns the most recent
            valid lines (either `current_lines` or `previous_lines`).
    """
    # Handle cases where no previous lines exist
    if previous_lines is None or None in previous_lines:
        return current_lines

    # Handle cases where no current lines are detected
    if current_lines is None or None in current_lines:
        return previous_lines

    # Unpack the current and previous lane lines
    left_current, right_current = current_lines
    left_prev, right_prev = previous_lines

    # Ensure all lines are valid before applying smoothing
    if left_current and left_prev and right_current and right_prev:
        # Smooth the left lane line
        left_smoothed = (
            (int((1 - smoothing_factor) * left_current[0][0] + smoothing_factor * left_prev[0][0]),
             int((1 - smoothing_factor) * left_current[0][1] + smoothing_factor * left_prev[0][1])),
            (int((1 - smoothing_factor) * left_current[1][0] + smoothing_factor * left_prev[1][0]),
             int((1 - smoothing_factor) * left_current[1][1] + smoothing_factor * left_prev[1][1]))
        )

        # Smooth the right lane line
        right_smoothed = (
            (int((1 - smoothing_factor) * right_current[0][0] + smoothing_factor * right_prev[0][0]),
             int((1 - smoothing_factor) * right_current[0][1] + smoothing_factor * right_prev[0][1])),
            (int((1 - smoothing_factor) * right_current[1][0] + smoothing_factor * right_prev[1][0]),
             int((1 - smoothing_factor) * right_current[1][1] + smoothing_factor * right_prev[1][1]))
        )

        return left_smoothed, right_smoothed

    # Return current lines if smoothing can't be applied
    return current_lines


def process_video(test_video, output_video):
    """
    Process video with temporal smoothing to reduce flickering
    """
    # Store previous frame data
    previous_lines = None

    def process_frame_with_smoothing(frame):
        """
        Processes a single video frame with temporal smoothing applied to lane detection.

        Parameters:
            frame (numpy.ndarray): The current video frame in RGB format (MoviePy format).

        Returns:
            numpy.ndarray: The processed frame with lane detection and smoothing applied,
                           converted back to RGB format for MoviePy.
        """
        nonlocal previous_lines  # Retain lane line information across frames

        # Create a copy of the frame to avoid modifying the original
        result = frame.copy()

        # Convert frame from RGB (MoviePy) to BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            # Convert the frame to grayscale for edge detection
            grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filtering to reduce noise while preserving edges
            blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

            # Apply Canny edge detection with specified thresholds
            edges = cv2.Canny(blur, 50, 150)

            # Extract the region of interest for lane detection
            region = region_selection(edges)

            # Detect lines using the Hough Transform
            hough = hough_transform(region)

            if hough is not None:
                # Get lane lines for the current frame
                current_lines = lane_lines(frame_bgr, hough)

                # Smooth the detected lines using temporal smoothing
                smoothed_lines = temporal_smoothing(current_lines, previous_lines)

                # Update the previous lines for use in the next frame
                if smoothed_lines[0] is not None and smoothed_lines[1] is not None:
                    previous_lines = smoothed_lines

                # If valid lines are detected, fill the lane area
                if smoothed_lines[0] and smoothed_lines[1]:
                    result = fill_lane_area(frame_bgr, smoothed_lines)

            # Convert the processed frame back to RGB format for MoviePy
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        except Exception as e:
            # Log any errors encountered while processing the frame
            print(f"Error processing frame: {str(e)}")
            return frame

    try:
        # Load the input video using MoviePy
        input_video = editor.VideoFileClip(test_video, audio=False)

        # Process each frame with the `process_frame_with_smoothing` function
        processed = input_video.fl_image(process_frame_with_smoothing)

        # Preview the processed video
        processed.preview(fps=input_video.fps, audio=False)

    except Exception as e:
        # Log any errors encountered while processing the video
        print(f"Error processing video: {str(e)}")

    finally:
        # Clean up resources
        try:
            input_video.close()
        except:
            pass

# calling driver function
process_video("video/NightDrive.mp4",'results/nightDriveResult.mp4')