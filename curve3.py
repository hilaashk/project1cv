import numpy as np
import cv2
from collections import deque
from moviepy.editor import VideoFileClip

# Initialize buffers for smoothing
LEFT_POLY_BUFFER = deque(maxlen=10)
RIGHT_POLY_BUFFER = deque(maxlen=10)


def region_of_interest(img):
    """Apply a region of interest mask using a polygon."""
    height, width = img.shape[:2]
    polygon = np.array([[
        (int(width * 0.15), int(height * 0.94)),
        (int(width * 0.45), int(height * 0.62)),
        (int(width * 0.58), int(height * 0.62)),
        (int(width * 0.95), int(height * 0.94))
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def detect_lane_lines(masked_edges):
    """Detect lane lines using Hough transform"""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=65,  # Increased from 30
        maxLineGap=200  # Reduced from 120
    )

    left_points = []
    right_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Tightened slope threshold for better line detection
                if 0.3 < abs(slope) < 2.5:  # Changed from 2 to 1.5
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    weight = int(length / 10)

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

    y_start = int(height * 0.95)  # Updated to match ROI
    y_end = int(height * 0.6)

    left_points = generate_points(left_poly, y_start, y_end)
    right_points = generate_points(right_poly, y_start, y_end)

    if left_points is not None and right_points is not None:
        # Check if lanes cross
        if np.any(left_points[:, 0] >= right_points[:, 0]):
            return None, None

        # Check if lanes are too close or too far apart
        lane_width = np.mean(right_points[:, 0] - left_points[:, 0])
        if lane_width < 300 or lane_width > 1000:  # Pixel thresholds
            return None, None

    return left_poly, right_poly


def draw_lanes(frame, left_poly, right_poly):
    """Draw the lane lines"""
    overlay = np.zeros_like(frame)
    height = frame.shape[0]

    # Adjust the bottom drawing point to match the ROI
    y_start = int(height * 0.95)  # Match the ROI bottom crop
    y_end = int(height * 0.6)

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

    # Adjust the bottom drawing point to match the ROI
    y_start = int(height * 0.95)  # Match the ROI bottom crop
    y_end = int(height * 0.6)

    if left_poly is not None and right_poly is not None:
        left_points = generate_points(left_poly, y_start, y_end)
        right_points = generate_points(right_poly, y_start, y_end)

        if left_points is not None and right_points is not None:
            points = np.vstack((left_points, np.flipud(right_points)))
            cv2.fillPoly(overlay, [points], (255, 255, 255))

    return overlay



def process_frame(frame):
    """Process a single video frame"""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lane_region = region_of_interest(edges)
    lines = cv2.HoughLinesP(
        lane_region,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=200
    )

    left_points, right_points = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            if 0.5 < abs(slope) < 2:
                if slope < 0:
                    left_points.extend([(x1, y1), (x2, y2)])
                else:
                    right_points.extend([(x1, y1), (x2, y2)])

    # Detect lane lines
    left_points, right_points = detect_lane_lines(lane_region)

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
    combined = cv2.addWeighted(lane_area, 0.3, frame, 1, 0)
    final_output = cv2.addWeighted(combined, 0.8, lane_overlay, 0.5, 1)

    return final_output


def process_video(input_file):
    """Process the input video file"""
    clip = VideoFileClip(input_file)
    processed_clip = clip.fl_image(process_frame)
    processed_clip.preview()


if __name__ == "__main__":
    process_video("carHillDrive.mp4")