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
    crop_bottom = int(rows * 0.90)
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
    Resolution-aware Hough transform
    """
    rows, cols = image.shape[:2]

    # Adjust parameters based on resolution
    if rows <= 720:  # For 720p
        minLineLength = 30  # Shorter minimum line length for 720p
        maxLineGap = 100  # Smaller gap for 720p
        threshold = 15  # Lower threshold for 720p
    else:  # For 1080p - your original parameters
        minLineLength = 50
        maxLineGap = 200
        threshold = 20

    rho = 2
    theta = np.pi / 180

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
                # Skip if points are identical
                if x1 == x2 and y1 == y2:
                    continue

                if x2 == x1:
                    # Handle vertical lines
                    continue  # Skip vertical lines as they're usually noise
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1

                # Tighter slope constraints to prevent crossing
                if -0.85 < slope < -0.5:  # Left lane
                    left_fit.append((slope, intercept))
                elif 0.5 < slope < 0.85:  # Right lane
                    right_fit.append((slope, intercept))

        # Require minimum number of lines
        min_lines = 1
        if len(left_fit) < min_lines or len(right_fit) < min_lines:
            return None, None

        # Calculate average slopes and intercepts
        left_lane = np.median(left_fit, axis=0) if len(left_fit) >= min_lines else None
        right_lane = np.median(right_fit, axis=0) if len(right_fit) >= min_lines else None

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


def lane_lines(image, lines):
    """
    Modified lane_lines function to include prediction
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.68
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    if left_line and right_line:  # Only process if both lines are detected
        return left_line, right_line
    return None, None

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter for noise reduction.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def temporal_smoothing(current_lines, previous_lines, smoothing_factor=0.8):
    """
    Apply temporal smoothing to lane lines to reduce flickering
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


def frame_processor(image):
    """
    Process the input frame with improved error handling and night-time adjustments.
    """
    try:
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply brightness adjustment for night conditions
        grayscale = cv2.convertScaleAbs(grayscale, alpha=2.5, beta=50)

        # Apply bilateral filter to reduce noise while preserving edges
        blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

        # Apply adaptive thresholding for better edge detection in varying lighting
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # Apply Canny edge detection with adjusted thresholds for night
        edges = cv2.Canny(edges, 50, 150)

        # Get region of interest - using same parameters as day
        region = region_selection(edges)

        # Detect lines using Hough transform - same as day
        hough = hough_transform(region)

        if hough is None:
            return image

        left_line, right_line = lane_lines(image, hough)

        if left_line is None or right_line is None:
            return image

        # Fill the lane area - same as day
        result = fill_lane_area(image, lane_lines(image, hough))

        return result
    except Exception as e:
        print(f"Error in frame processing: {str(e)}")
        return image

# driver function
def process_video(test_video, output_video):
    """
    Process video with temporal smoothing to reduce flickering
    """
    # Store previous frame data
    previous_lines = None

    def process_frame_with_smoothing(frame):
        nonlocal previous_lines

        # Process the current frame
        result = frame.copy()

        # Convert frame from RGB (MoviePy) to BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            # Get the grayscale image
            grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise while preserving edges
            blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

            # Edge detection with more robust parameters
            edges = cv2.Canny(blur, 50, 150)

            # Get region of interest
            region = region_selection(edges)

            # Detect lines
            hough = hough_transform(region)

            if hough is not None:
                # Get current frame's lines
                current_lines = lane_lines(frame_bgr, hough)

                # Apply temporal smoothing
                smoothed_lines = temporal_smoothing(current_lines, previous_lines)

                # Update previous lines for next frame
                if smoothed_lines[0] is not None and smoothed_lines[1] is not None:
                    previous_lines = smoothed_lines

                # Draw the lanes if we have valid lines
                if smoothed_lines[0] and smoothed_lines[1]:
                    result = fill_lane_area(frame_bgr, smoothed_lines)

            # Convert back to RGB for MoviePy
            return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

    try:
        # Load and process the video
        input_video = editor.VideoFileClip(test_video, audio=False)
        processed = input_video.fl_image(process_frame_with_smoothing)
        processed.write_videofile(output_video, fps=input_video.fps, audio=False)


    except Exception as e:
        print(f"Error processing video: {str(e)}")

    finally:
        # Clean up
        try:
            input_video.close()
        except:
            pass

# calling driver function
process_video("video/NightDrive.mp4",'results/nightDriveResult.mp4')