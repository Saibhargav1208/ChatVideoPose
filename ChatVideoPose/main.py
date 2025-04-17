import argparse
import cv2
import numpy as np
import torch
import yaml
import logging
import os
from typing import Dict, List, Optional

# Import modules
from src.pose.estimator import create_pose_estimator
from src.pose.features import PoseFeatureExtractor
from src.embedding.positional_embedding import TemporalPositionalEmbedding
from src.models.q_former import QFormer
from src.models.linear_layer import LLMIntegration
from src.llm.model import LLMWrapper
from src.llm.space import LLMSpace, SMPLTokenHandler
from src.query.processor import QueryProcessor, PoseQueryMatcher
from src.utils.visualization import PoseVisualizer, ArchitectureVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def extract_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of frames as numpy arrays
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file {video_path}")
            raise ValueError(f"Could not open video file {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)
        
        logger.info(f"Extracting {max_frames} frames from {video_path} (total: {total_frames}, fps: {fps:.2f})")
        
        # Extract frames
        frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
        
        # Release video capture
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        raise

class PoseLLM:
    """
    Main class for Pose-LLM integration
    """
    def __init__(self, config_path: str):
        """
        Initialize Pose-LLM
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set device
        self.device = torch.device(self.config['system']['device'])
        
        # Initialize components
        self._init_components()
        
        logger.info("Initialized PoseLLM")
    
    def _init_components(self):
        """Initialize all components"""
        # Initialize pose estimator
        self.pose_estimator = create_pose_estimator(self.config)
        
        # Initialize pose feature extractor
        self.pose_feature_extractor = PoseFeatureExtractor(self.config)
        
        # Initialize positional embedding
        self.positional_embedding = TemporalPositionalEmbedding(self.config).to(self.device)
        
        # Initialize Q-Former
        self.q_former = QFormer(self.config).to(self.device)
        
        # Initialize LLM integration
        self.llm_integration = LLMIntegration(self.config).to(self.device)
        
        # Initialize LLM wrapper
        self.llm_wrapper = LLMWrapper(self.config)
        
        # Initialize LLM space
        self.llm_space = LLMSpace(self.config)
        
        # Initialize SMPL token handler
        self.smpl_token_handler = SMPLTokenHandler(self.config)
        
        # Initialize query processor
        self.query_processor = QueryProcessor(self.config)
        
        # Initialize pose query matcher
        self.pose_query_matcher = PoseQueryMatcher(self.config)
        
        # Initialize visualizers
        self.pose_visualizer = PoseVisualizer(self.config)
        self.architecture_visualizer = ArchitectureVisualizer(self.config)
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process a video to extract pose information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing processed pose information
        """
        # Extract frames
        max_frames = self.config['pose']['max_frames']
        frames = extract_frames(video_path, max_frames)
        
        # Estimate pose
        pose_data = self.pose_estimator.process_frames(frames)
        
        # Extract pose features
        pose_features = self.pose_feature_extractor.extract_features(pose_data)
        
        # Add batch dimension if needed
        if len(pose_features.shape) == 2:
            pose_features = pose_features.unsqueeze(0)
        
        # Move to device
        pose_features = pose_features.to(self.device)
        
        # Apply positional embedding
        embedded_features = self.positional_embedding(pose_features)
        
        # Process through Q-Former
        q_former_output = self.q_former(embedded_features)
        
        # Apply LLM integration
        llm_features = self.llm_integration(q_former_output)
        
        # Prepare for LLM
        llm_inputs = self.llm_space.prepare_inputs_for_llm(
            llm_features["token_weights"],
            llm_features["token_embeddings"]
        )
        
        # Visualize poses on frames (first frame as example)
        if len(frames) > 0:
            visualized_frame = self.pose_visualizer.visualize_pose_on_frame(
                frames[0],
                pose_data["landmarks"][0],
                frame_idx=0,
                save=True
            )
        
        return {
            "pose_data": pose_data,
            "pose_features": pose_features,
            "embedded_features": embedded_features,
            "q_former_output": q_former_output,
            "llm_features": llm_features,
            "llm_inputs": llm_inputs,
            "frames": frames
        }
    
    def answer_query(self, query: str, pose_features: Dict) -> str:
        """
        Answer a query based on pose information
        
        Args:
            query: User query
            pose_features: Processed pose features
            
        Returns:
            Answer to the query
        """
        # Process query
        query_data = self.query_processor.process_query(query)
        
        # Match query with pose
        enhanced_query_data = self.pose_query_matcher.match_query_with_pose(
            query_data,
            pose_features["llm_features"]
        )
        
        # Get enhanced query
        enhanced_query = enhanced_query_data["enhanced_query"]
        
        # Answer query with LLM
        answer = self.llm_wrapper.answer_question(
            enhanced_query,
            pose_features["llm_features"]
        )
        
        return answer
    
    def visualize_architecture(self):
        """Visualize the architecture"""
        self.architecture_visualizer.visualize_architecture(save=True)
        logger.info("Architecture visualization saved")


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pose-LLM Integration")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--query", type=str, default=None, help="Query to answer")
    parser.add_argument("--visualize", action="store_true", help="Visualize architecture")
    
    args = parser.parse_args()
    
    try:
        # Initialize Pose-LLM
        pose_llm = PoseLLM(args.config)
        
        # Visualize architecture if requested
        if args.visualize:
            pose_llm.visualize_architecture()
        
        # Process video
        logger.info(f"Processing video: {args.video}")
        pose_features = pose_llm.process_video(args.video)
        
        # Answer query if provided
        if args.query:
            logger.info(f"Answering query: {args.query}")
            answer = pose_llm.answer_query(args.query, pose_features)
            
            print("\nQuery:", args.query)
            print("Answer:", answer)
        
        logger.info("Processing completed successfully")
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
