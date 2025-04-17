import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseFeatureExtractor:
    """
    Extract meaningful features from pose landmarks
    """
    def __init__(self, config: Dict):
        self.config = config
        self.feature_dim = config['pose']['feature_dim']
        logger.info(f"Initialized PoseFeatureExtractor with feature dimension {self.feature_dim}")
    
    def extract_features(self, pose_data: Dict) -> torch.Tensor:
        """
        Extract features from pose landmarks
        
        Args:
            pose_data: Dictionary containing pose landmarks
            
        Returns:
            Tensor of extracted features
        """
        landmarks = pose_data["landmarks"]
        num_frames = pose_data["num_frames"]
        
        # Extract basic features (positions, velocities, angles)
        position_features = self._extract_position_features(landmarks)
        velocity_features = self._extract_velocity_features(landmarks)
        angle_features = self._extract_angle_features(landmarks)
        
        # Combine features
        all_features = np.concatenate([position_features, velocity_features, angle_features], axis=-1)
        
        # Project to the required feature dimension if needed
        if all_features.shape[-1] != self.feature_dim:
            all_features = self._project_features(all_features)
        
        logger.info(f"Extracted pose features with shape {all_features.shape}")
        
        # Convert to torch tensor
        return torch.tensor(all_features, dtype=torch.float32)
    
    def _extract_position_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract position-based features from landmarks
        
        Args:
            landmarks: Numpy array of shape [num_frames, num_keypoints, dims]
            
        Returns:
            Position features
        """
        # Normalize positions relative to a reference point (e.g., hip center)
        # For MediaPipe, landmark 23 and 24 are left and right hip
        if landmarks.shape[1] >= 24:  # Checking that we have enough landmarks
            hip_center = (landmarks[:, 23, :2] + landmarks[:, 24, :2]) / 2
            hip_center = np.expand_dims(hip_center, axis=1)
            
            # Normalize XY coordinates relative to hip center
            normalized_positions = landmarks[:, :, :2] - hip_center
            
            # Flatten keypoints for each frame
            return normalized_positions.reshape(landmarks.shape[0], -1)
        else:
            # Fallback if we don't have the expected landmark structure
            return landmarks[:, :, :2].reshape(landmarks.shape[0], -1)
    
    def _extract_velocity_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract velocity-based features from landmarks
        
        Args:
            landmarks: Numpy array of shape [num_frames, num_keypoints, dims]
            
        Returns:
            Velocity features
        """
        if landmarks.shape[0] < 2:
            # Return zeros if we don't have enough frames for velocity
            return np.zeros((landmarks.shape[0], landmarks.shape[1] * 2))
        
        # Calculate frame-to-frame differences (velocities)
        velocities = landmarks[1:, :, :2] - landmarks[:-1, :, :2]
        
        # Pad with zeros for the first frame to maintain the same number of frames
        padded_velocities = np.vstack([
            np.zeros((1, landmarks.shape[1], 2)),
            velocities
        ])
        
        # Flatten keypoints for each frame
        return padded_velocities.reshape(landmarks.shape[0], -1)
    
    def _extract_angle_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract angle-based features from landmarks
        
        Args:
            landmarks: Numpy array of shape [num_frames, num_keypoints, dims]
            
        Returns:
            Angle features
        """
        num_frames = landmarks.shape[0]
        
        # Define key joints to calculate angles
        # This is a simplified example - you would define the specific joints based on
        # your pose estimation model and the angles you're interested in
        
        # For MediaPipe, we can define some common angles (simplified example)
        angles = []
        
        for frame_idx in range(num_frames):
            frame_angles = []
            
            # Calculate some example angles if we have enough landmarks
            if landmarks.shape[1] >= 32:
                # Right elbow angle (shoulder-elbow-wrist)
                shoulder = landmarks[frame_idx, 12, :2]
                elbow = landmarks[frame_idx, 14, :2]
                wrist = landmarks[frame_idx, 16, :2]
                right_elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                frame_angles.append(right_elbow_angle)
                
                # Left elbow angle
                shoulder = landmarks[frame_idx, 11, :2]
                elbow = landmarks[frame_idx, 13, :2]
                wrist = landmarks[frame_idx, 15, :2]
                left_elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                frame_angles.append(left_elbow_angle)
                
                # Right knee angle
                hip = landmarks[frame_idx, 24, :2]
                knee = landmarks[frame_idx, 26, :2]
                ankle = landmarks[frame_idx, 28, :2]
                right_knee_angle = self._calculate_angle(hip, knee, ankle)
                frame_angles.append(right_knee_angle)
                
                # Left knee angle
                hip = landmarks[frame_idx, 23, :2]
                knee = landmarks[frame_idx, 25, :2]
                ankle = landmarks[frame_idx, 27, :2]
                left_knee_angle = self._calculate_angle(hip, knee, ankle)
                frame_angles.append(left_knee_angle)
            
            angles.append(frame_angles)
        
        # Convert to numpy array with proper shape
        if angles and angles[0]:
            return np.array(angles)
        else:
            # Fallback if we couldn't calculate angles
            return np.zeros((num_frames, 4))  # Assuming 4 angles as in the example
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate the angle between three points (in degrees)
        
        Args:
            a: First point coordinates
            b: Second point coordinates (vertex of the angle)
            c: Third point coordinates
            
        Returns:
            Angle in degrees
        """
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip to handle floating point errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def _project_features(self, features: np.ndarray) -> np.ndarray:
        """
        Project features to the desired dimension using PCA-like approach
        
        Args:
            features: Input features
            
        Returns:
            Projected features with the target dimension
        """
        # Simple approach: Use a random projection matrix
        # In a real implementation, you might want to use PCA or another dimension reduction technique
        
        input_dim = features.shape[-1]
        output_dim = self.feature_dim
        
        if input_dim > output_dim:
            # Reduce dimension
            # For simplicity, we'll use a random projection
            # In practice, you'd want to use a more sophisticated method
            np.random.seed(self.config['system']['seed'])
            projection_matrix = np.random.randn(input_dim, output_dim)
            projected_features = np.matmul(features, projection_matrix)
            return projected_features
        else:
            # Pad with zeros to reach target dimension
            padded_features = np.zeros((features.shape[0], self.feature_dim))
            padded_features[:, :input_dim] = features
            return padded_features
