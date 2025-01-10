import numpy as np
import cv2
from collections import deque
from moviepy.editor import VideoFileClip

# Initialize buffers for smoothing
LEFT_POLY_BUFFER = deque(maxlen=20)
RIGHT_POLY_BUFFER = deque(maxlen=10)


def region_of_interest(image):
    """Apply a region of interest mask"""
    mask = np.zeros_like(image)
    rows, cols = image.shape[:2]
    vertices = np.array([[
        (int(cols * 0.08), rows),
        (int(cols * 0.35), int(rows * 0.6)),  # Changed from 0.6 to 0.75
        (int(cols * 0.65), int(rows * 0.6)),
        (int(cols * 0.9), rows)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def detect_lane_lines(masked_edges):
    """Detect lane lines using Hough transform"""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=0.5,  # Changed from 1 to 2
        theta=np.pi / 180,
        threshold=20,
        minLineLength=30,  # Changed from 30 to 50
        maxLineGap=120  # Changed from 120 to 200
    )

    left_points = []
    right_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                if 0.5 < abs(slope) < 2:
                    # Calculate line length for weighting
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    weight = int(length / 10)  # Weight points based on line length

                    if slope < 0:  # Left lane
                        left_points.extend([(x1, y1)] * weight)
                        left_points.extend([(x2, y2)] * weight)
                    else:  # Right lane
                        right_points.extend([(x1, y1)] * weight)
                        right_points.extend([(x2, y2)] * weight)

    return left_points, right_points


def fit_poly(points):
    """Fit a polynomial to lane points"""
    if len(points) >= 3:  # Need at least 3 points for quadratic fit
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        try:
            return np.polyfit(y, x, 2)
        except:
            return None
    return None


def smooth_poly(poly, buffer):
    """Smooth polynomial coefficients using a rolling buffer"""
    if poly is not None:
        buffer.append(poly)
    if len(buffer) > 0:
        return np.mean(buffer, axis=0)
    return None


def generate_points(poly, y_start, y_end):
    """Generate points for lane visualization"""
    if poly is None:
        return None
    y = np.linspace(y_start, y_end, num=100)
    try:
        x = np.polyval(poly, y)
        return np.column_stack((x, y)).astype(np.int32)
    except:
        return None


def calculate_curvature(poly, y_eval):
    """Calculate the curvature radius of a polynomial at a point"""
    if poly is None:
        return None

    # Convert pixel space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Calculate curvature
    A = poly[0] * (xm_per_pix / (ym_per_pix ** 2))
    B = poly[1] * (xm_per_pix / ym_per_pix)

    curvature = ((1 + (2 * A * y_eval * ym_per_pix + B) ** 2) ** 1.5) / np.absolute(2 * A)
    return curvature


def enforce_lane_consistency(left_poly, right_poly, height):
    """Ensure left and right lanes don't cross"""
    if left_poly is None or right_poly is None:
        return left_poly, right_poly

    y_start, y_end = height, int(height * 0.8)  # Changed from 0.6 to 0.75
    left_points = generate_points(left_poly, y_start, y_end)
    right_points = generate_points(right_poly, y_start, y_end)

    if left_points is not None and right_points is not None:
        # Check if lanes cross
        if np.any(left_points[:, 0] >= right_points[:, 0]):
            return None, None

        # Check if lanes are too close or too far apart
        lane_width = np.mean(right_points[:, 0] - left_points[:, 0])
        if lane_width < 400 or lane_width > 1000:  # Pixel thresholds
            return None, None

    return left_poly, right_poly


def draw_lanes(frame, left_poly, right_poly):
    """Draw the lane lines"""
    overlay = np.zeros_like(frame)
    height = frame.shape[0]
    y_start, y_end = height, int(height * 0.75)

    if left_poly is not None:
        left_points = generate_points(left_poly, y_start, y_end)
        if left_points is not None:
            cv2.polylines(overlay, [left_points], False, (0, 0, 255), 10)

    if right_poly is not None:
        right_points = generate_points(right_poly, y_start, y_end)
        if right_points is not None:
            cv2.polylines(overlay, [right_points], False, (0, 0, 255), 10)

    return overlay


def draw_lane_area(frame, left_poly, right_poly):
    """Draw the lane area"""
    overlay = np.zeros_like(frame)
    height = frame.shape[0]
    y_start, y_end = height, int(height * 0.75)

    if left_poly is not None and right_poly is not None:
        left_points = generate_points(left_poly, y_start, y_end)
        right_points = generate_points(right_poly, y_start, y_end)

        if left_points is not None and right_points is not None:
            points = np.vstack((left_points, np.flipud(right_points)))
            cv2.fillPoly(overlay, [points], (255, 255, 255))

    return overlay


def process_frame(frame):
    """Process a single video frame"""
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 20, 150)

    # Apply region of interest mask
    masked_edges = region_of_interest(edges)

    # Detect lane lines
    left_points, right_points = detect_lane_lines(masked_edges)

    # Fit polynomials
    left_poly = fit_poly(left_points)
    right_poly = fit_poly(right_points)

    # Apply smoothing
    left_poly = smooth_poly(left_poly, LEFT_POLY_BUFFER)
    right_poly = smooth_poly(right_poly, RIGHT_POLY_BUFFER)

    # Check lane consistency
    left_poly, right_poly = enforce_lane_consistency(left_poly, right_poly, frame.shape[0])

    # Draw lanes and lane area
    lane_overlay = draw_lanes(frame, left_poly, right_poly)
    lane_area = draw_lane_area(frame, left_poly, right_poly)

    # Combine everything
    combined = cv2.addWeighted(lane_area, 0.3, frame, 0.7, 0)
    final_output = cv2.addWeighted(combined, 0.8, lane_overlay, 1, 1)

    # Add curvature information
    if left_poly is not None and right_poly is not None:
        left_curvature = calculate_curvature(left_poly, frame.shape[0])
        right_curvature = calculate_curvature(right_poly, frame.shape[0])
        if left_curvature and right_curvature:
            avg_curvature = (left_curvature + right_curvature) / 2
            cv2.putText(final_output,
                        f'Curve Radius: {int(avg_curvature)}m',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

    return final_output


def process_video(input_file):
    """Process video file"""
    clip = VideoFileClip(input_file)
    processed_clip = clip.fl_image(process_frame)
    processed_clip.preview()


if __name__ == "__main__":
    process_video(r"C:\Users\User\Desktop\Computer vision\CV-lane-detection\carHillDrive.mp4")