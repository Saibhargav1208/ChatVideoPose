import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import mediapipe as mp
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseVisualizer:
    """
    Visualize pose estimation results
    """
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['system']['output_dir']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MediaPipe drawing utils
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        logger.info(f"Initialized PoseVisualizer with output directory {self.output_dir}")
    
    def visualize_pose_on_frame(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray,
        frame_idx: int = 0,
        save: bool = False
    ) -> np.ndarray:
        """
        Visualize pose landmarks on a frame
        
        Args:
            frame: Frame image as numpy array
            landmarks: Pose landmarks for the frame
            frame_idx: Frame index
            save: Whether to save the visualization
            
        Returns:
            Frame with pose visualization
        """
        # Convert landmarks to MediaPipe format
        mp_landmarks = self._convert_to_mp_landmarks(landmarks)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            mp_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Save visualization if requested
        if save:
            output_path = os.path.join(self.output_dir, f"pose_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            logger.info(f"Saved pose visualization to {output_path}")
        
        return annotated_frame
    
    def _convert_to_mp_landmarks(self, landmarks: np.ndarray) -> mp.solutions.pose.PoseLandmark:
        """
        Convert numpy landmarks to MediaPipe format
        
        Args:
            landmarks: Pose landmarks as numpy array of shape [num_keypoints, 4]
                      Where the dimensions are [x, y, z, visibility]
            
        Returns:
            MediaPipe pose landmarks
        """
        # Create a landmark proto
        mp_landmarks = self.mp_pose.PoseLandmark()
        
        # Convert numpy landmarks to MediaPipe format
        landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        
        for i in range(landmarks.shape[0]):
            landmark = landmark_list.landmark.add()
            landmark.x = float(landmarks[i, 0])
            landmark.y = float(landmarks[i, 1])
            landmark.z = float(landmarks[i, 2])
            landmark.visibility = float(landmarks[i, 3])
        
        return landmark_list
    
    def visualize_pose_features(
        self,
        pose_features: torch.Tensor,
        save: bool = False,
        filename: str = "pose_features.png"
    ) -> Figure:
        """
        Visualize pose features
        
        Args:
            pose_features: Pose features tensor of shape [seq_len, feature_dim]
            save: Whether to save the visualization
            filename: Output filename if saving
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(pose_features, torch.Tensor):
            features = pose_features.detach().cpu().numpy()
        else:
            features = pose_features
        
        # Get dimensions
        seq_len, feature_dim = features.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot features heatmap
        im = ax.imshow(features.T, aspect='auto', cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Feature Value')
        
        # Add labels
        ax.set_xlabel('Frame')
        ax.set_ylabel('Feature Dimension')
        ax.set_title('Pose Features')
        
        # Add grid
        ax.grid(False)
        
        # Save if requested
        if save:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pose features visualization to {output_path}")
        
        return fig
    
    def visualize_token_weights(
        self,
        token_weights: torch.Tensor,
        save: bool = False,
        filename: str = "token_weights.png"
    ) -> Figure:
        """
        Visualize token weights
        
        Args:
            token_weights: Token weights tensor of shape [batch_size, num_query_tokens, num_tokens]
            save: Whether to save the visualization
            filename: Output filename if saving
            
        Returns:
            Matplotlib figure
        """
        # Squeeze batch dimension if only one sample
        if token_weights.shape[0] == 1:
            token_weights = token_weights.squeeze(0)
        
        # Convert to numpy
        if isinstance(token_weights, torch.Tensor):
            weights = token_weights.detach().cpu().numpy()
        else:
            weights = token_weights
        
        # Get dimensions
        num_query_tokens, num_tokens = weights.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot token weights heatmap
        im = ax.imshow(weights, aspect='auto', cmap='YlOrRd')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight')
        
        # Add labels
        ax.set_xlabel('Token')
        ax.set_ylabel('Query Token')
        ax.set_title('Token Weights')
        
        # Add grid
        ax.grid(False)
        
        # Save if requested
        if save:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved token weights visualization to {output_path}")
        
        return fig


class ArchitectureVisualizer:
    """
    Visualize the overall architecture
    """
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['system']['output_dir']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized ArchitectureVisualizer with output directory {self.output_dir}")
    
    def visualize_architecture(
        self,
        save: bool = True,
        filename: str = "architecture.png"
    ) -> Figure:
        """
        Visualize the overall architecture
        
        Args:
            save: Whether to save the visualization
            filename: Output filename if saving
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Define components
        components = [
            {'name': 'Video Frames', 'pos': (0.1, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Pose Estimation', 'pos': (0.3, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Pose Features', 'pos': (0.5, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Positional Embedding', 'pos': (0.7, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Q-Former', 'pos': (0.9, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Linear Layer', 'pos': (1.1, 0.7), 'width': 0.1, 'height': 0.2},
            {'name': 'Query', 'pos': (0.1, 0.3), 'width': 0.1, 'height': 0.2},
            {'name': 'LLM', 'pos': (0.3, 0.3), 'width': 0.1, 'height': 0.2},
            {'name': 'LLM Space', 'pos': (0.7, 0.3), 'width': 0.1, 'height': 0.2},
            {'name': 'Answer', 'pos': (1.1, 0.3), 'width': 0.1, 'height': 0.2},
        ]
        
        # Define connections
        connections = [
            {'start': 'Video Frames', 'end': 'Pose Estimation'},
            {'start': 'Pose Estimation', 'end': 'Pose Features'},
            {'start': 'Pose Features', 'end': 'Positional Embedding'},
            {'start': 'Positional Embedding', 'end': 'Q-Former'},
            {'start': 'Q-Former', 'end': 'Linear Layer'},
            {'start': 'Linear Layer', 'end': 'LLM Space'},
            {'start': 'Query', 'end': 'LLM'},
            {'start': 'LLM', 'end': 'LLM Space'},
            {'start': 'LLM Space', 'end': 'Answer'},
        ]
        
        # Create a component lookup dictionary
        component_lookup = {comp['name']: comp for comp in components}
        
        # Draw components
        for comp in components:
            x, y = comp['pos']
            width, height = comp['width'], comp['height']
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), width, height, facecolor='skyblue', edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            
            # Add text
            ax.text(x + width/2, y + height/2, comp['name'], ha='center', va='center', fontsize=10)
        
        # Draw connections
        for conn in connections:
            start_comp = component_lookup[conn['start']]
            end_comp = component_lookup[conn['end']]
            
            # Calculate start and end points
            start_x = start_comp['pos'][0] + start_comp['width']
            start_y = start_comp['pos'][1] + start_comp['height']/2
            
            end_x = end_comp['pos'][0]
            end_y = end_comp['pos'][1] + end_comp['height']/2
            
            # Draw arrow
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
        
        # Add SMPL token arrow
        ax.annotate('', xy=(0.7, 0.3), xytext=(1.1, 0.7),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
        ax.text(0.9, 0.5, 'SMPL token', ha='center', va='center', fontsize=10, color='red')
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title
        ax.set_title('Pose-LLM Architecture', fontsize=14)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Save if requested
        if save:
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved architecture visualization to {output_path}")
        
        return fig
