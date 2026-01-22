import cv2
import numpy as np

class FieldCalibrator:
    """
    Detects field lines and attempts to identify which section of the pitch
    is visible in order to compute a proper homography for the minimap.
    """
    
    def __init__(self):
        # Standard soccer field dimensions in meters
        self.field_length = 105  # meters (goal line to goal line)
        self.field_width = 68    # meters (sideline to sideline)
        
        # Key dimensions for detection
        self.penalty_area_length = 16.5  # meters from goal line
        self.penalty_area_width = 40.3   # meters
        self.goal_area_length = 5.5      # meters from goal line
        self.goal_area_width = 18.32     # meters
        self.center_circle_radius = 9.15 # meters
        self.penalty_spot_distance = 11  # meters from goal line
        
        # Store detected features
        self.detected_lines = []
        self.detected_circles = []
        self.field_mask = None
        
        # Homography matrix
        self.homography = None
        self.calibrated = False
        
    def detect_field_mask(self, frame):
        """
        Detect the green field area to isolate the pitch.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range for grass
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.field_mask = mask
        return mask
    
    def detect_white_lines(self, frame):
        """
        Detect white field lines using edge detection and color filtering.
        """
        # First, get the field mask
        field_mask = self.detect_field_mask(frame)
        
        # Convert to HSV and detect white/light colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # White line detection (high value, low saturation)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Only keep white pixels that are on or near the field
        dilated_field = cv2.dilate(field_mask, np.ones((20, 20), np.uint8))
        white_on_field = cv2.bitwise_and(white_mask, dilated_field)
        
        # Apply Canny edge detection
        edges = cv2.Canny(white_on_field, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        self.detected_lines = lines if lines is not None else []
        return self.detected_lines, white_on_field
    
    def detect_circles(self, frame):
        """
        Detect the center circle or penalty arcs.
        """
        field_mask = self.detect_field_mask(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        gray_masked = cv2.bitwise_and(gray, field_mask)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=200
        )
        
        self.detected_circles = circles[0] if circles is not None else []
        return self.detected_circles
    
    def classify_lines(self, lines, frame_shape):
        """
        Classify detected lines as horizontal or vertical based on angle.
        """
        if len(lines) == 0:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            
            # Classify based on angle
            if angle < 30:  # More horizontal
                horizontal_lines.append(line[0])
            elif angle > 60:  # More vertical
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def find_field_boundaries(self, frame):
        """
        Find the approximate boundaries of the visible field area.
        Returns the four corners of the visible field in pixel coordinates.
        """
        lines, white_mask = self.detect_white_lines(frame)
        
        if len(lines) == 0:
            return None
        
        h, w = frame.shape[:2]
        horizontal_lines, vertical_lines = self.classify_lines(lines, frame.shape)
        
        # Find the bounding lines of the field
        # Top boundary (smallest y horizontal line)
        # Bottom boundary (largest y horizontal line)
        # Left boundary (smallest x vertical line)
        # Right boundary (largest x vertical line)
        
        top_y = h
        bottom_y = 0
        left_x = w
        right_x = 0
        
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            avg_y = (y1 + y2) / 2
            if avg_y < top_y:
                top_y = int(avg_y)
            if avg_y > bottom_y:
                bottom_y = int(avg_y)
        
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            avg_x = (x1 + x2) / 2
            if avg_x < left_x:
                left_x = int(avg_x)
            if avg_x > right_x:
                right_x = int(avg_x)
        
        # If we didn't find clear boundaries, use field mask
        if top_y >= bottom_y or left_x >= right_x:
            field_mask = self.detect_field_mask(frame)
            contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest_contour)
                left_x, top_y = x, y
                right_x, bottom_y = x + cw, y + ch
        
        # Return corners: top-left, top-right, bottom-right, bottom-left
        corners = np.array([
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
            [left_x, bottom_y]
        ], dtype=np.float32)
        
        return corners
    
    def estimate_visible_field_region(self, frame):
        """
        Estimate which part of the field (in meters) is visible.
        This is a heuristic based on detected features.
        
        Returns: (x_min, y_min, x_max, y_max) in field coordinates (meters)
        """
        circles = self.detect_circles(frame)
        h, w = frame.shape[:2]
        
        # Default: assume we're looking at a section near the center
        # This covers roughly 1/4 to 1/3 of the field width
        visible_length = 35  # meters visible along the length
        visible_width = self.field_width  # full width usually visible
        
        # Check for center circle
        center_circle_detected = False
        for circle in circles:
            cx, cy, r = circle
            # If circle is roughly in center of frame, likely center circle
            if 0.3 * w < cx < 0.7 * w and 0.2 * h < cy < 0.8 * h:
                if r > 40:  # Reasonably large circle
                    center_circle_detected = True
                    break
        
        if center_circle_detected:
            # We're looking at the center of the field
            x_min = (self.field_length - visible_length) / 2
            x_max = (self.field_length + visible_length) / 2
        else:
            # Assume we're looking at one half of the field
            # Default to center-ish position
            x_min = self.field_length / 4
            x_max = x_min + visible_length
        
        y_min = 0
        y_max = self.field_width
        
        return (x_min, y_min, x_max, y_max)
    
    def calibrate(self, frame):
        """
        Perform calibration on a frame to establish the homography.
        """
        h, w = frame.shape[:2]
        
        # Find field boundaries in pixel space
        pixel_corners = self.find_field_boundaries(frame)
        
        if pixel_corners is None:
            # Fallback to frame edges
            pixel_corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)
        
        # Estimate which part of the field is visible
        x_min, y_min, x_max, y_max = self.estimate_visible_field_region(frame)
        
        # Define world corners (in meters)
        world_corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ], dtype=np.float32)
        
        # Compute homography
        self.homography, _ = cv2.findHomography(pixel_corners, world_corners)
        self.calibrated = True
        
        # Store the visible region for minimap
        self.visible_region = (x_min, y_min, x_max, y_max)
        
        return self.homography, self.visible_region
    
    def transform_point(self, pixel_point):
        """
        Transform a pixel coordinate to field coordinate using the calibrated homography.
        """
        if not self.calibrated or self.homography is None:
            return None
        
        point = np.array([[pixel_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        
        return transformed[0][0]
    
    def draw_debug(self, frame):
        """
        Draw detected lines and features for debugging.
        """
        debug_frame = frame.copy()
        
        # Draw detected lines
        if len(self.detected_lines) > 0:
            for line in self.detected_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw detected circles
        if len(self.detected_circles) > 0:
            for circle in self.detected_circles:
                cx, cy, r = circle
                cv2.circle(debug_frame, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
        
        return debug_frame
