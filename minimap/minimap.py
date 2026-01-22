import cv2
import numpy as np
from field_calibrator import FieldCalibrator

class Minimap:
    def __init__(self, minimap_size=(360, 240), padding=10, background_color=(34, 139, 34), line_color=(255, 255, 255), alpha=0.7):
        self.minimap_size = minimap_size  # width, height (120:80 yard ratio = 3:2)
        self.padding = padding
        self.background_color = background_color
        self.line_color = line_color
        self.alpha = alpha  # Transparency level
        
        # Standard field dimensions in meters
        self.field_length = 105  # meters
        self.field_width = 68    # meters
        
        # Field calibrator for detecting visible field region
        self.calibrator = FieldCalibrator()
        self.visible_region = None  # Will be set during calibration
        
    def calibrate_from_frame(self, frame):
        """
        Calibrate the minimap using a reference frame.
        """
        _, self.visible_region = self.calibrator.calibrate(frame)
        return self.visible_region

    def _get_minimap_drawing_area(self, frame):
        height, width, _ = frame.shape
        start_x = (width - self.minimap_size[0]) // 2  # Center horizontally
        start_y = height - self.minimap_size[1] - self.padding
        end_x = start_x + self.minimap_size[0]
        end_y = height - self.padding
        return (start_x, start_y, end_x, end_y)

    def _draw_pitch(self, frame):
        start_x, start_y, end_x, end_y = self._get_minimap_drawing_area(frame)
        
        # Create overlay for semi-transparency
        overlay = frame.copy()
        
        # Draw background (green pitch)
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), self.background_color, -1)
        
        # Pitch dimensions for scaling
        pitch_w = self.minimap_size[0]
        pitch_h = self.minimap_size[1]
        
        # Line thickness
        line_thickness = 1
        
        # Draw outer boundary
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), self.line_color, line_thickness)
        
        # Draw center line (vertical, dividing left and right halves)
        center_x = start_x + pitch_w // 2
        cv2.line(overlay, (center_x, start_y), (center_x, end_y), self.line_color, line_thickness)
        
        # Draw center circle
        center_y = start_y + pitch_h // 2
        center_radius = int(pitch_h * 0.12)
        cv2.circle(overlay, (center_x, center_y), center_radius, self.line_color, line_thickness)
        
        # Draw center spot
        cv2.circle(overlay, (center_x, center_y), 2, self.line_color, -1)
        
        # Penalty box dimensions (at left and right ends)
        penalty_box_width = int(pitch_w * 0.12)  # Depth of penalty box
        penalty_box_height = int(pitch_h * 0.6)  # Width of penalty box
        penalty_box_y_start = start_y + (pitch_h - penalty_box_height) // 2
        penalty_box_y_end = penalty_box_y_start + penalty_box_height
        
        # Left penalty box
        cv2.rectangle(overlay, (start_x, penalty_box_y_start), 
                     (start_x + penalty_box_width, penalty_box_y_end), self.line_color, line_thickness)
        
        # Right penalty box
        cv2.rectangle(overlay, (end_x - penalty_box_width, penalty_box_y_start), 
                     (end_x, penalty_box_y_end), self.line_color, line_thickness)
        
        # Six-yard box dimensions
        six_yard_width = int(pitch_w * 0.05)
        six_yard_height = int(pitch_h * 0.35)
        six_yard_y_start = start_y + (pitch_h - six_yard_height) // 2
        six_yard_y_end = six_yard_y_start + six_yard_height
        
        # Left six-yard box
        cv2.rectangle(overlay, (start_x, six_yard_y_start), 
                     (start_x + six_yard_width, six_yard_y_end), self.line_color, line_thickness)
        
        # Right six-yard box
        cv2.rectangle(overlay, (end_x - six_yard_width, six_yard_y_start), 
                     (end_x, six_yard_y_end), self.line_color, line_thickness)
        
        # Goals (at left and right ends)
        goal_width = 4
        goal_height = int(pitch_h * 0.18)
        goal_y_start = start_y + (pitch_h - goal_height) // 2
        goal_y_end = goal_y_start + goal_height
        
        # Left goal
        cv2.rectangle(overlay, (start_x - goal_width, goal_y_start), 
                     (start_x, goal_y_end), self.line_color, -1)
        
        # Right goal
        cv2.rectangle(overlay, (end_x, goal_y_start), 
                     (end_x + goal_width, goal_y_end), self.line_color, -1)
        
        # Apply semi-transparency
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        
        return frame

    def _transform_to_minimap_space(self, position, frame):
        if position is None:
            return None
        
        start_x, start_y, end_x, end_y = self._get_minimap_drawing_area(frame)
        
        # Use the view_transformer's coordinate system directly:
        # x: 0 to 23.32 (left to right on screen)
        # y: 0 to 68 (far/top to near/bottom on screen)
        court_length = 23.32
        court_width = 68
        
        # Normalize within the view_transformer's coordinate system
        norm_x = position[0] / court_length
        norm_y = position[1] / court_width
        
        # Clamp normalized values to [0, 1]
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Map to minimap coordinates
        minimap_x = int(start_x + norm_x * self.minimap_size[0])
        minimap_y = int(start_y + norm_y * self.minimap_size[1])
        
        # Clamp to minimap bounds
        minimap_x = max(start_x, min(end_x - 1, minimap_x))
        minimap_y = max(start_y, min(end_y - 1, minimap_y))
        
        return (minimap_x, minimap_y)
    
    def _transform_pixel_to_minimap(self, pixel_position, frame):
        """
        Transform directly from pixel position to minimap, bypassing the view transformer.
        Uses the field calibrator's homography.
        """
        if pixel_position is None:
            return None
        
        # Transform pixel to world coordinates using calibrator
        world_pos = self.calibrator.transform_point(pixel_position)
        if world_pos is None:
            return None
        
        return self._transform_to_minimap_space(world_pos, frame)

    def draw(self, frame, tracks):
        output_frame = self._draw_pitch(frame)
        
        # Draw players
        for track_id, player in tracks.get('players', {}).items():
            position = player.get('position_transformed')
            team_color = player.get('team_color')
            minimap_pos = self._transform_to_minimap_space(position, frame)
            
            if minimap_pos:
                # Convert team_color to tuple of ints if it exists
                if team_color is not None:
                    team_color = tuple(int(c) for c in team_color)
                else:
                    team_color = (200, 200, 200)  # Default gray
                
                radius = 4
                if player.get('has_ball', False):
                    # Highlight player with ball - larger size with outline
                    cv2.circle(output_frame, minimap_pos, radius + 3, (255, 255, 0), 2)  # Yellow outline
                    cv2.circle(output_frame, minimap_pos, radius, team_color, -1)
                else:
                    cv2.circle(output_frame, minimap_pos, radius, team_color, -1)
                    cv2.circle(output_frame, minimap_pos, radius, (0, 0, 0), 1)  # Black outline for visibility

        # Draw ball
        ball_track = tracks.get('ball', {})
        if ball_track:
            ball_info = next(iter(ball_track.values()), None)
            if ball_info:
                position = ball_info.get('position_transformed')
                minimap_pos = self._transform_to_minimap_space(position, frame)
                if minimap_pos:
                    # White ball with black outline
                    cv2.circle(output_frame, minimap_pos, 3, (255, 255, 255), -1)
                    cv2.circle(output_frame, minimap_pos, 3, (0, 0, 0), 1)

        return output_frame

    def draw_minimap_on_frames(self, frames, tracks):
        output_frames = []
        
        # Calibrate on the first frame
        if len(frames) > 0:
            self.calibrate_from_frame(frames[0])
            print(f"Field calibration: visible region = {self.visible_region}")
        
        for frame_num, frame in enumerate(frames):
            frame_with_minimap = frame.copy()
            
            player_tracks_for_frame = tracks['players'][frame_num]
            ball_tracks_for_frame = tracks['ball'][frame_num]
            
            tracks_for_frame = {
                'players': player_tracks_for_frame,
                'ball': ball_tracks_for_frame
            }

            frame_with_minimap = self.draw(frame_with_minimap, tracks_for_frame)
            output_frames.append(frame_with_minimap)
        return output_frames
