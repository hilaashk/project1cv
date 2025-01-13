import numpy as np
import cv2
from collections import deque
import os

# Buffers to smooth detected lane positions
LEFT_LINE_BUFFER = deque(maxlen=20)
RIGHT_LINE_BUFFER = deque(maxlen=20)


class VehicleDetector:
    """
    A class for detecting vehicles in a video frame using Haar Cascade
    with additional filters to reduce false positives.
    """
    def __init__(self):
        # Load the Haar Cascade classifier
        cascade_path = "haarcascade_car.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")

        self.car_cascade = cv2.CascadeClassifier(cascade_path)

        # Detection parameters
        self.scale_factor = 1.02  # How much the image size is reduced at each image scale
        self.min_neighbors = 4  # How many neighbors each candidate rectangle should have
        self.min_size = (300, 300)  # Minimum size of detected objects
        self.max_size = (1500, 1500)  # Maximum size of detected objects

        # Filtering parameters
        self.min_aspect_ratio = 0.3  # Minimum width/height ratio for valid detection
        self.max_aspect_ratio = 2.8  # Maximum width/height ratio for valid detection
        self.min_area = 1000  # Minimum area (in pixels) for valid detection

        # Parameters for the reflection zone (bottom of the frame)
        self.reflection_zone_height = 0.15  # Bottom 15% of frame
        self.reflection_zone_min_area = 2000  # Larger area threshold for reflection zone

    def is_valid_detection(self, x, y, w, h, frame_height, frame_width):
        """
        Validate a detected object based on position, size, and other criteria.

        Args:
            x, y: Top-left corner of the bounding box.
            w, h: Width and height of the bounding box.
            frame_height, frame_width: Dimensions of the video frame.

        Returns:
            (bool, str): Whether the detection is valid and the reason if invalid.
        """
        aspect_ratio = w / h
        area = w * h
        in_reflection_zone = y > frame_height * (1 - self.reflection_zone_height)

        # Aspect ratio checks
        if aspect_ratio < self.min_aspect_ratio:
            return False, "aspect_ratio_too_small"
        if aspect_ratio > self.max_aspect_ratio:
            return False, "aspect_ratio_too_large"

        # Area checks
        if in_reflection_zone:
            if area < self.reflection_zone_min_area:
                return False, "too_small_in_reflection_zone"
        else:
            if area < self.min_area:
                return False, "too_small"

        # Position checks
        if y + h > frame_height:  # Extends below the frame
            return False, "extends_below_frame"
        if y < frame_height * 0.2 and y + h < frame_height * 0.3:  # Too high in the frame
            return False, "too_high_in_frame"

        return True, "valid"


    def detect_vehicles(self, frame):
        """
        Detect vehicles in the input frame using the Haar Cascade classifier.

        Args:
            frame (ndarray): Input video frame (BGR format).

        Returns:
            detections (list): Valid vehicle detections as [bounding_box, score, label].
        """
        frame_height, frame_width = frame.shape[:2]

        # Use the left 70% of the frame for detection
        crop_width = int(frame_width * 0.7)
        detection_frame = frame[:, :crop_width]

        # Convert to grayscale for the Haar Cascade classifier
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

        # Perform vehicle detection
        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            maxSize=self.max_size
        )

        detections = []
        filtered_info = []  # Store info about filtered (invalid) detections

        for (x, y, w, h) in cars:
            is_valid, reason = self.is_valid_detection(x, y, w, h, frame_height, crop_width)

            if is_valid:
                detections.append(([x, y, w, h], 1.0, 'car'))
            else:
                filtered_info.append({
                    'box': [x, y, w, h],
                    'reason': reason,
                    'area': w * h,
                    'aspect_ratio': w / h
                })

        return detections


def draw_vehicles(frame, detections):
    """
    Draw bounding boxes and labels for detected vehicles on the frame.

    Args:
        frame (ndarray): The input video frame (BGR format).
        detections (list): List of detected vehicles, each represented as:
            [bounding_box, confidence, class_name],
            where bounding_box is [x, y, w, h].

    Returns:
        ndarray: The video frame with bounding boxes and labels drawn.
    """
    for box, confidence, class_name in detections:
        x, y, w, h = box
        color = (0, 255, 255)  # Yellow color for vehicles

        # Draw a bounding box around the detected vehicle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Calculate additional details for debugging
        area = w * h  # Area of the bounding box in pixels
        aspect_ratio = w / h  # Width-to-height ratio of the bounding box

        # Label format: class name, area, and aspect ratio
        label = f'{class_name} ({area}px, {aspect_ratio:.1f})'
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def region_selection(image):
    """
    Determine and isolate the region of interest (ROI) in the input image.

    The ROI is a trapezoidal area in the lower part of the image, focusing on the lane area.
    This approach avoids interference from adjacent lanes or irrelevant image regions.

    Args:
        image (ndarray): Input image (grayscale or color).

    Returns:
        ndarray: Image with only the region of interest visible, others masked out.
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        # Multichannel image
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # Single-channel grayscale image
        ignore_mask_color = 255

    # Define ROI vertices as a trapezoid
    rows, cols = image.shape[:2]
    crop_bottom = int(rows * 0.8)  # Bottom crop height (80% of frame)
    bottom_left = [cols * 0.15, crop_bottom]
    top_left = [cols * 0.35, rows * 0.65]
    bottom_right = [cols * 0.9, crop_bottom]
    top_right = [cols * 0.6, rows * 0.6]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # Apply mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Apply the Hough Transform to detect lines in a binary image.

    The Hough Transform identifies straight lines by detecting points in polar coordinates.

    Args:
        image (ndarray): Binary edge-detected image.

    Returns:
        ndarray: Array of detected lines in the format [[x1, y1, x2, y2], ...].
    """
    return cv2.HoughLinesP(
        image,
        rho=1,  # Pixel resolution of the accumulator (1 pixel)
        theta=np.pi / 180,  # Angular resolution of 1 degree in radians
        threshold=30,  # Minimum votes needed to accept a line
        minLineLength=40,  # Minimum line segment length
        maxLineGap=50  # Maximum allowable gap between segments to consider them as one line
    )

def average_slope_intercept(lines, frame_width):
    """
    Calculate the average slope and intercept for left and right lane lines.

    Filters and averages detected lines based on their slope, position, and proximity
    to other lines. This method ensures stability by rejecting outliers and
    requiring a minimum number of valid lines.

    Args:
        lines (ndarray): Detected lines from the Hough Transform,
                         each represented as [[x1, y1, x2, y2], ...].
        frame_width (int): Width of the video frame, used for filtering lines.

    Returns:
        tuple:
            - left_lane (tuple or None): Average (slope, intercept) of the left lane.
            - right_lane (tuple or None): Average (slope, intercept) of the right lane.
    """
    if lines is None:
        return None, None

    left_fit = []
    right_fit = []

    try:
        left_xs = []
        right_xs = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:  # Skip vertical lines (undefined slope)
                    continue

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Filter lines by slope and position
                if 0.4 < abs(slope) < 0.85:  # Filter for reasonable slopes
                    if slope < 0 and x1 < frame_width * 0.5:  # Likely left lane
                        left_xs.append(x1)
                        left_fit.append((slope, intercept))
                    elif slope > 0 and x1 > frame_width * 0.45:  # Likely right lane
                        right_xs.append(x1)
                        right_fit.append((slope, intercept))

        # Calculate median x-positions to refine left/right lane separation
        left_median_x = np.median(left_xs) if left_xs else None
        right_median_x = np.median(right_xs) if right_xs else None

        # Filter lines deviating too far from the median x-position
        if left_median_x is not None:
            left_fit = [fit for fit, x in zip(left_fit, left_xs)
                        if abs(x - left_median_x) < frame_width * 0.1]
        if right_median_x is not None:
            right_fit = [fit for fit, x in zip(right_fit, right_xs)
                         if abs(x - right_median_x) < frame_width * 0.1]

        # Require a minimum number of valid lines to form a lane
        min_lines = 2
        if len(left_fit) < min_lines and len(right_fit) < min_lines:
            return None, None

        # Calculate the average slope and intercept for each lane
        left_lane = np.mean(left_fit, axis=0) if len(left_fit) >= min_lines else None
        right_lane = np.mean(right_fit, axis=0) if len(right_fit) >= min_lines else None

        return left_lane, right_lane

    except Exception as e:
        print(f"Error in average_slope_intercept: {str(e)}")
        return None, None

def pixel_points(y1, y2, line):
    """
    Convert the slope and intercept of a line into pixel coordinates.

    Given the slope and intercept of a line (y = mx + b), this function calculates
    the corresponding x-coordinates for two given y-values, effectively transforming
    the line equation into pixel points for rendering or further calculations.

    Args:
        y1 (float): The first y-coordinate.
        y2 (float): The second y-coordinate.
        line (tuple or None): A tuple containing the slope (m) and intercept (b) of the line.
                              If None, the function returns None.

    Returns:
        tuple or None: A tuple containing two pixel points ((x1, y1), (x2, y2)), or None
                       if the line is invalid or an exception occurs.
    """
    if line is None:
        return None

    slope, intercept = line

    try:
        # Compute x-coordinates using the line equation y = mx + b
        x1 = int((y1 - intercept) / slope) if slope != 0 else int(intercept)
        x2 = int((y2 - intercept) / slope) if slope != 0 else int(intercept)
        return ((x1, int(y1)), (x2, int(y2)))
    except Exception as e:
        # Handle exceptions caused by invalid slope or intercept values
        print(f"Error in pixel_points: {e}")
        return None

# Global variables for detecting lane changes
LANE_CHANGE_THRESHOLD = 50  # Minimum pixel shift required to detect a lane change
TRANSITION_FRAMES = 15  # Number of frames over which a transition is monitored
current_transition_frame = 0  # Frame counter for the ongoing transition
is_transitioning = False  # Indicates whether a lane transition is currently happening
transition_start_lines = None  # Starting lane line positions at the beginning of the transition
transition_target_lines = None  # Target lane line positions at the end of the transition

def detect_lane_change(current_lines, previous_lines):
    """
    Detect whether a lane change has occurred by comparing current and previous lane lines.

    This function analyzes the lateral movement of lane lines and validates the consistency
    of lane width to ensure accurate detection of lane changes.

    Args:
        current_lines (tuple): Current lane lines as (left_line, right_line), where each line
                               is represented by two points ((x1, y1), (x2, y2)).
        previous_lines (tuple): Previous lane lines in the same format.

    Returns:
        bool: True if a lane change is detected, False otherwise.
    """
    # Ensure both current and previous lines are valid
    if not current_lines or not previous_lines or None in current_lines or None in previous_lines:
        return False

    # Unpack current and previous lane lines
    left_current, right_current = current_lines
    left_prev, right_prev = previous_lines

    lane_changes = []

    # Check lateral movement of the left lane
    if left_current and left_prev:
        left_movement = abs(left_current[0][0] - left_prev[0][0])
        lane_changes.append(left_movement > LANE_CHANGE_THRESHOLD)

    # Check lateral movement of the right lane
    if right_current and right_prev:
        right_movement = abs(right_current[0][0] - right_prev[0][0])
        lane_changes.append(right_movement > LANE_CHANGE_THRESHOLD)

    # Validate lane width consistency
    if left_current and right_current and left_prev and right_prev:
        current_width = abs(right_current[0][0] - left_current[0][0])  # Current lane width
        prev_width = abs(right_prev[0][0] - left_prev[0][0])  # Previous lane width
        width_change = abs(current_width - prev_width)

        # Ignore lane change detection if lane width changes significantly
        if width_change > LANE_CHANGE_THRESHOLD * 1.5:
            return False

    # Detect a lane change only if one lane moves and not both simultaneously
    return any(lane_changes) and not all(lane_changes)

def temporal_smoothing(current_lines, previous_lines, smoothing_factor=0.85):
    """
    Smooth transitions between current and previous lane line positions over time.

    This function helps reduce jitter in lane detection by blending current and previous
    lane positions. It also handles lane change transitions by ensuring a smooth
    interpolation between start and target positions.

    Args:
        current_lines (tuple): Current lane lines as (left_line, right_line), where each line
                               is represented by two points ((x1, y1), (x2, y2)).
        previous_lines (tuple): Previous lane lines in the same format.
        smoothing_factor (float): Weight factor for smoothing; higher values prioritize
                                  previous positions.

    Returns:
        tuple: Smoothed lane lines as (left_line, right_line).
    """
    global is_transitioning, current_transition_frame, transition_start_lines, transition_target_lines

    # Return current lines if no valid previous lines exist
    if previous_lines is None or None in previous_lines:
        return current_lines

    # Return previous lines if no valid current lines exist
    if current_lines is None or None in current_lines:
        return previous_lines

    # Check for a lane change and initialize transition if detected
    if not is_transitioning:
        is_lane_change = detect_lane_change(current_lines, previous_lines)
        if is_lane_change:
            is_transitioning = True
            current_transition_frame = 0
            transition_start_lines = previous_lines
            transition_target_lines = current_lines
            return previous_lines  # Keep previous lines during transition initialization

    # Handle an ongoing lane change transition
    if is_transitioning:
        progress = current_transition_frame / TRANSITION_FRAMES

        # Complete the transition when the target frame count is reached
        if current_transition_frame >= TRANSITION_FRAMES:
            is_transitioning = False
            return transition_target_lines

        # Interpolate between start and target positions for a smooth transition
        left_start, right_start = transition_start_lines
        left_target, right_target = transition_target_lines

        if left_start and left_target and right_start and right_target:
            # Validate lane width during the transition
            start_width = abs(right_start[0][0] - left_start[0][0])
            target_width = abs(right_target[0][0] - left_target[0][0])

            # Abort transition if lane width changes significantly
            if abs(target_width - start_width) > start_width * 0.3:
                is_transitioning = False
                return previous_lines

            # Smooth left lane transition
            left_smoothed = (
                (int(left_start[0][0] + (left_target[0][0] - left_start[0][0]) * progress),
                 int(left_start[0][1] + (left_target[0][1] - left_start[0][1]) * progress)),
                (int(left_start[1][0] + (left_target[1][0] - left_start[1][0]) * progress),
                 int(left_start[1][1] + (left_target[1][1] - left_start[1][1]) * progress))
            )

            # Smooth right lane transition
            right_smoothed = (
                (int(right_start[0][0] + (right_target[0][0] - right_start[0][0]) * progress),
                 int(right_start[0][1] + (right_target[0][1] - right_start[0][1]) * progress)),
                (int(right_start[1][0] + (right_target[1][0] - right_start[1][0]) * progress),
                 int(right_start[1][1] + (right_target[1][1] - right_start[1][1]) * progress))
            )

            current_transition_frame += 1
            return left_smoothed, right_smoothed

    # Apply normal temporal smoothing for stable lane detection
    left_current, right_current = current_lines
    left_prev, right_prev = previous_lines

    if left_current and left_prev and right_current and right_prev:
        current_width = abs(right_current[0][0] - left_current[0][0])  # Current lane width
        prev_width = abs(right_prev[0][0] - left_prev[0][0])  # Previous lane width

        # Reject smoothing if lane width changes significantly
        if abs(current_width - prev_width) > prev_width * 0.2:  # 20% threshold
            return previous_lines

        # Smooth left lane
        left_smoothed = (
            (int((1 - smoothing_factor) * left_current[0][0] + smoothing_factor * left_prev[0][0]),
             int((1 - smoothing_factor) * left_current[0][1] + smoothing_factor * left_prev[0][1])),
            (int((1 - smoothing_factor) * left_current[1][0] + smoothing_factor * left_prev[1][0]),
             int((1 - smoothing_factor) * left_current[1][1] + smoothing_factor * left_prev[1][1]))
        )

        # Smooth right lane
        right_smoothed = (
            (int((1 - smoothing_factor) * right_current[0][0] + smoothing_factor * right_prev[0][0]),
             int((1 - smoothing_factor) * right_current[0][1] + smoothing_factor * right_prev[0][1])),
            (int((1 - smoothing_factor) * right_current[1][0] + smoothing_factor * right_prev[1][0]),
             int((1 - smoothing_factor) * right_current[1][1] + smoothing_factor * right_prev[1][1]))
        )

        return left_smoothed, right_smoothed

    return current_lines

def fill_lane_area(image, lines, color=(0, 255, 0)):
    """
    Fill the area between detected lane lines with a semi-transparent overlay.

    This function takes the coordinates of the left and right lane lines and fills the
    area between them with a specified color. It helps visually highlight the detected
    lane area on the image.

    Args:
        image (ndarray): Input image (BGR format) where the lane area will be filled.
        lines (tuple): Detected lane lines as (left_line, right_line), where each line is
                       represented by two points ((x1, y1), (x2, y2)).
        color (tuple): BGR color for the lane area. Default is green (0, 255, 0).

    Returns:
        ndarray: Image with the lane area filled with a semi-transparent overlay.
    """
    # Check if lines are valid
    if lines is None or None in lines:
        return image

    left_line, right_line = lines

    # Ensure both left and right lines are present
    if left_line is None or right_line is None:
        return image

    # Create an overlay to draw the filled polygon
    overlay = np.zeros_like(image)

    # Define points for the polygon: bottom and top of both lane lines
    points = np.array([
        [left_line[0], left_line[1], right_line[1], right_line[0]]
    ], dtype=np.int32)

    # Fill the polygon with the specified color
    cv2.fillPoly(overlay, points, color)

    # Blend the overlay with the original image to create a semi-transparent effect
    return cv2.addWeighted(image, 1, overlay, 0.3, 0)



def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=8):
    """
    Draw the detected lane lines on the input image.

    This function takes the coordinates of the left and right lane lines and
    overlays them onto the original image, making the detected lanes clearly visible.

    Args:
        image (ndarray): Input image (BGR format) where the lane lines will be drawn.
        lines (tuple): Detected lane lines as (left_line, right_line), where each line is
                       represented by two points ((x1, y1), (x2, y2)).
        color (tuple): BGR color for the lane lines. Default is green (0, 255, 0).
        thickness (int): Thickness of the lane lines. Default is 8.

    Returns:
        ndarray: Image with the detected lane lines drawn.
    """
    # Create an empty image for drawing the lane lines
    line_image = np.zeros_like(image)

    if lines is not None:
        # Unpack the left and right lane lines
        left_line, right_line = lines

        # Draw the left lane line if it exists
        if left_line is not None:
            cv2.line(line_image, left_line[0], left_line[1], color, thickness)

        # Draw the right lane line if it exists
        if right_line is not None:
            cv2.line(line_image, right_line[0], right_line[1], color, thickness)

    # Overlay the lane lines onto the original image
    return cv2.addWeighted(image, 1, line_image, 1, 0)



# Initialize vehicle detector
vehicle_detector = VehicleDetector()


def process_frame(frame):
    """
    Process a single video frame to detect lanes and vehicles.

    This function combines vehicle detection and lane detection. It performs preprocessing
    steps like grayscale conversion, edge detection, and Hough transform to identify lanes.
    Temporal smoothing and visualization techniques are applied to stabilize and highlight
    the detected lanes and vehicles.

    Args:
        frame (ndarray): Input video frame (BGR format).

    Returns:
        ndarray: Processed video frame with detected lanes and vehicles visualized.
    """
    # Detect vehicles in the frame
    vehicle_detections = vehicle_detector.detect_vehicles(frame)

    # Convert the frame to grayscale for lane detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to extract edges
    edges = cv2.Canny(blur, 50, 150)

    # Define and select the region of interest for lane detection
    region = region_selection(edges)

    # Detect lines using Hough transform
    hough_lines = hough_transform(region)

    # Create a copy of the frame to draw the results
    result = frame.copy()

    if hough_lines is not None:
        # Process detected lines to identify lanes
        left_lane, right_lane = average_slope_intercept(hough_lines, frame.shape[1])  # Use frame width

        # Define the vertical range of the lane lines
        y1 = frame.shape[0]  # Bottom of the frame
        y2 = int(y1 * 0.67)  # Approximately 2/3 of the frame height

        # Convert the slope and intercept of the lanes into pixel points
        left_line = pixel_points(y1, y2, left_lane)
        right_line = pixel_points(y1, y2, right_lane)

        # Store current lane lines
        current_lines = (left_line, right_line)

        # Apply temporal smoothing to stabilize lane lines
        smoothed_lines = temporal_smoothing(
            current_lines,
            (LEFT_LINE_BUFFER[-1], RIGHT_LINE_BUFFER[-1]) if len(LEFT_LINE_BUFFER) > 0 else None
        )

        # Update buffers with smoothed lane lines
        if smoothed_lines[0]:
            LEFT_LINE_BUFFER.append(smoothed_lines[0])
        if smoothed_lines[1]:
            RIGHT_LINE_BUFFER.append(smoothed_lines[1])

        # Draw the filled lane area and lane lines on the frame
        if smoothed_lines[0] and smoothed_lines[1]:
            result = fill_lane_area(result, smoothed_lines)  # Fill the area between the lanes
            result = draw_lane_lines(result, smoothed_lines)  # Draw lane lines

    # Overlay detected vehicles onto the frame
    result = draw_vehicles(result, vehicle_detections)

    return result

def process_video(input_file, output_file=None, show_preview=True):
    """
    Process a video file to perform lane and vehicle detection.

    This function reads a video file frame by frame, processes each frame for lane and
    vehicle detection, and optionally writes the processed video to an output file or
    displays it in a preview window.

    Args:
        input_file (str): Path to the input video file.
        output_file (str, optional): Path to save the processed video. If None, the video
                                     will not be saved. Default is None.
        show_preview (bool): Whether to display a preview of the processed video while
                             processing. Default is True.

    Returns:
        None
    """
    # Open the video file
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if an output file is specified
    out = None
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no more frames are available

            # Process the current frame
            processed_frame = process_frame(frame)

            # Write the processed frame to the output file
            if out:
                out.write(processed_frame)

            # Show the processed frame in a preview window (optional)
            if show_preview:
                cv2.imshow('Lane and Vehicle Detection', processed_frame)
                # Press 'q' to quit the preview
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Release video capture and writer resources
        cap.release()
        if out:
            out.release()
        # Close all OpenCV windows if the preview was enabled
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage: Update these paths to match your file locations
    video_path = "video\detectCarVid.mp4"
    process_video(video_path, "results/output_with_detection.mp4", show_preview=False)
