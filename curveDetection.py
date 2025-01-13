import numpy as np
import cv2
from collections import deque
from moviepy.editor import VideoFileClip

# Initialize buffers for smoothing
LEFT_POLY_BUFFER = deque(maxlen=10)
RIGHT_POLY_BUFFER = deque(maxlen=10)


def region_of_interest(img):
    """
        Apply a region of interest mask using a polygon.

        This function creates a mask to isolate a specific region of interest
        in the given image, typically focusing on the road area for lane detection.
        The polygonal region is dynamically defined based on the image dimensions.

        Parameters:
        img (numpy.ndarray): The input image to apply the region of interest mask.

        Returns:
        numpy.ndarray: The image with the region of interest applied. Areas outside
                       the polygon will be masked out (set to black).
        """

    height, width = img.shape[:2]
    polygon = np.array([[
        (int(width * 0.2), height),  # Bottom-left corner
        (int(width * 0.4), int(height * 0.6)),  # Top-left corner
        (int(width * 0.8), int(height * 0.6)),  # Top-right corner
        (int(width), height)  # Bottom-right corner
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def color_threshold(image):
    """
        Apply color thresholding to detect white and yellow lanes.

        This function isolates white and yellow colors in an image, which are
        typically used to represent lane markings on roads. A mask is created for
        each color based on predefined thresholds, and the results are combined
        to highlight the relevant regions.

        Parameters:
        image (numpy.ndarray): The input image in BGR color space.

        Returns:
        numpy.ndarray: The image with only the white and yellow regions retained,
                       masked by the combined color thresholds.
        """

    # White mask
    lower_white = np.array([160, 160, 160])
    upper_white = np.array([220, 225, 200])
    mask_white = cv2.inRange(image, lower_white, upper_white)

    # Yellow mask
    lower_yellow = np.array([70, 120, 155])
    upper_yellow = np.array([110, 160, 225])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return cv2.bitwise_and(image, image, mask=combined_mask)

def fit_poly(points):
    """
        Fit a polynomial to lane points.

        This function fits a quadratic polynomial to a set of points representing
        a lane in an image. A minimum of three points is required for a quadratic
        fit. The polynomial is fitted with the y-coordinates as the independent
        variable and the x-coordinates as the dependent variable.

        Parameters:
        points (list of tuples): A list of (x, y) coordinates representing lane points.

        Returns:
        numpy.ndarray or None: The coefficients of the quadratic polynomial in the
                               form [a, b, c] (where ax^2 + bx + c = y), or None if
                               the fit cannot be performed.
        """
    if len(points) >= 3:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        try:
            return np.polyfit(y, x, 2)
        except:
            return None
    return None


def smooth_poly(poly, buffer):
    """
       Smooth polynomial coefficients using a rolling buffer.

       This function smooths the coefficients of a polynomial by maintaining a
       rolling buffer of recent polynomials and calculating their average. This
       helps reduce noise and fluctuations in lane detection over consecutive frames.

       Parameters:
       poly (numpy.ndarray or None): The current polynomial coefficients to add
                                     to the buffer. Can be None if no valid polynomial exists.
       buffer (list): A list used as a rolling buffer to store polynomial coefficients
                      from recent frames.

       Returns:
       numpy.ndarray or None: The smoothed polynomial coefficients (averaged across
                              the buffer) or None if the buffer is empty.
       """
    if poly is not None:
        buffer.append(poly)
    if len(buffer) > 0:
        return np.mean(buffer, axis=0)
    return None


def generate_points(poly, y_start, y_end):
    """
        Generate points for lane visualization.

        This function generates a set of (x, y) points based on a given polynomial
        to represent the detected lane over a specified range of y-values. These
        points can be used to draw the lane on an image or video frame.

        Parameters:
        poly (numpy.ndarray): The polynomial coefficients (in the form [a, b, c])
                               used to calculate the x-values for each y.
        y_start (int or float): The starting y-value for generating points.
        y_end (int or float): The ending y-value for generating points.

        Returns:
        numpy.ndarray or None: A 2D array of points representing the lane (x, y)
                               coordinates. Returns None if the polynomial is invalid
                               or the calculation fails.
        """
    if poly is None:
        return None
    y = np.linspace(y_start, y_end, num=100)
    try:
        x = np.polyval(poly, y)
        return np.column_stack((x, y)).astype(np.int32)
    except:
        return None

def enforce_lane_consistency(left_poly, right_poly, height):
    """
       Ensure left and right lanes don't cross and maintain consistent width.

       This function verifies that the left and right lane boundaries do not overlap
       or cross each other. It also checks if the lane width (the distance between
       the left and right lanes) is within a reasonable range. If either of these
       conditions is violated, it returns None for both lanes to indicate an error
       in lane detection.

       Parameters:
       left_poly (numpy.ndarray or None): The polynomial coefficients for the left lane.
       right_poly (numpy.ndarray or None): The polynomial coefficients for the right lane.
       height (int): The height (number of rows) of the image, used to define the range
                     for lane point generation.

       Returns:
       tuple: A tuple containing the updated left and right polynomials. If any
              consistency checks fail, returns (None, None).
       """
    if left_poly is None or right_poly is None:
        return left_poly, right_poly

    y_start = int(height * 0.95)
    y_end = int(height * 0.6)

    left_points = generate_points(left_poly, y_start, y_end)
    right_points = generate_points(right_poly, y_start, y_end)

    if left_points is not None and right_points is not None:
        # Check if lanes cross
        if np.any(left_points[:, 0] >= right_points[:, 0]):
            return None, None

        lane_width = np.mean(right_points[:, 0] - left_points[:, 0])
        if lane_width < 300 or lane_width > 1000:
            return None, None

    return left_poly, right_poly


def draw_lanes(frame, left_poly, right_poly):
    """
       Draw the lane lines on the frame.

       This function generates the lane lines by creating points from the left
       and right polynomials and then drawing them on a blank overlay. The
       overlay is then returned, and it can be added to the original frame to
       visualize the lane markings.

       Parameters:
       frame (numpy.ndarray): The input image/frame on which the lane lines will be drawn.
       left_poly (numpy.ndarray or None): The polynomial coefficients for the left lane.
       right_poly (numpy.ndarray or None): The polynomial coefficients for the right lane.

       Returns:
       numpy.ndarray: An overlay image with the lane lines drawn on it, which can be
                      added to the original frame for visualization.
       """
    overlay = np.zeros_like(frame)
    height = frame.shape[0]

    # Adjust the bottom drawing point to match the ROI
    y_start = int(height * 0.9)  # Match the ROI bottom crop
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
    """
        Draw the lane lines on the frame.

        This function generates the lane lines by creating points from the left
        and right polynomials and then drawing them on a blank overlay. The
        overlay is then returned, and it can be added to the original frame to
        visualize the lane markings.

        Parameters:
        frame (numpy.ndarray): The input image/frame on which the lane lines will be drawn.
        left_poly (numpy.ndarray or None): The polynomial coefficients for the left lane.
        right_poly (numpy.ndarray or None): The polynomial coefficients for the right lane.

        Returns:
        numpy.ndarray: An overlay image with the lane lines drawn on it, which can be
                       added to the original frame for visualization.
        """
    overlay = np.zeros_like(frame)
    height = frame.shape[0]

    # Adjust the bottom drawing point to match the ROI
    y_start = int(height * 0.9)  # Match the ROI bottom crop
    y_end = int(height * 0.6)

    if left_poly is not None and right_poly is not None:
        left_points = generate_points(left_poly, y_start, y_end)
        right_points = generate_points(right_poly, y_start, y_end)
        points = np.vstack((left_points, np.flipud(right_points)))
        cv2.fillPoly(overlay, [points], (255, 255, 255))

    return overlay

def process_frame(frame):
    """
        Process a single video frame to detect lanes and visualize them.

        This function applies several image processing steps to a frame from the
        video. It performs color thresholding to detect lane markings, converts
        the image to grayscale, applies Gaussian blur, and detects edges using
        Canny edge detection. It then isolates the region of interest (ROI) where
        the lanes are located and uses the Hough transform to detect lane lines.
        Afterward, polynomials are fitted to the detected points, smoothing is
        applied to the polynomials, and consistency between the left and right lanes
        is checked. Finally, the detected lanes and lane area are drawn on the frame
        and returned for visualization.

        Parameters:
        frame (numpy.ndarray): The input image/frame to be processed.

        Returns:
        numpy.ndarray: The processed frame with lane lines and lane area drawn on it.
        """
    img=color_threshold(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
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


def process_video(input_path,output_path):
    """Process the input video file"""
    input_video = VideoFileClip(input_path)
    processed = input_video.fl_image(process_frame)
    processed.write_videofile(output_path, fps=input_video.fps, audio=False)


if __name__ == "__main__":
    process_video("video/curveDriveVid.mp4","results/curveDriveResult.mp4")