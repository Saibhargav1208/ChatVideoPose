import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Process user queries and prepare them for the LLM
    """
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Initialized QueryProcessor")
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a user query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing processed query information
        """
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        # Extract query type (what/how/why/etc.)
        query_type = self._extract_query_type(cleaned_query)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_query)
        
        logger.info(f"Processed query of type '{query_type}' with {len(keywords)} keywords")
        
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "query_type": query_type,
            "keywords": keywords
        }
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize the query
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove special characters (except question marks)
        query = re.sub(r'[^\w\s\?]', '', query)
        
        return query
    
    def _extract_query_type(self, query: str) -> str:
        """
        Extract the type of query (what, how, why, etc.)
        
        Args:
            query: Cleaned query string
            
        Returns:
            Query type
        """
        # Define query type patterns
        query_types = {
            "what": r'^\s*what\b',
            "how": r'^\s*how\b',
            "why": r'^\s*why\b',
            "when": r'^\s*when\b',
            "where": r'^\s*where\b',
            "who": r'^\s*who\b',
            "is": r'^\s*is\b',
            "are": r'^\s*are\b',
            "can": r'^\s*can\b',
            "does": r'^\s*does\b',
            "do": r'^\s*do\b',
        }
        
        # Check each pattern
        for query_type, pattern in query_types.items():
            if re.search(pattern, query):
                return query_type
        
        # Default query type
        return "general"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from the query
        
        Args:
            query: Cleaned query string
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'how', 'why', 'when', 'where', 'who', 'is', 'are', 'can', 'does',
            'do', 'this', 'that', 'these', 'those', 'of', 'for', 'in', 'on',
            'at', 'by', 'to', 'with', 'about'
        }
        
        # Split into words
        words = query.split()
        
        # Filter out stop words
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def enhance_query_with_pose_context(
        self, 
        query_data: Dict[str, any], 
        pose_description: Optional[str] = None
    ) -> str:
        """
        Enhance the query with pose context
        
        Args:
            query_data: Processed query data
            pose_description: Optional textual description of the pose
            
        Returns:
            Enhanced query string
        """
        original_query = query_data["original_query"]
        
        # If no pose description is provided, return the original query
        if pose_description is None:
            return original_query
        
        # Enhance the query with pose context
        enhanced_query = f"Given the following human pose: '{pose_description}', {original_query}"
        
        return enhanced_query


class PoseQueryMatcher:
    """
    Match pose features with query to better understand the intent
    """
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Initialized PoseQueryMatcher")
    
    def match_query_with_pose(
        self, 
        query_data: Dict[str, any], 
        pose_features: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        Match query with pose features to enhance understanding
        
        Args:
            query_data: Processed query data
            pose_features: Pose feature data
            
        Returns:
            Enhanced query data with pose matching information
        """
        # Extract query keywords
        keywords = query_data["keywords"]
        
        # Generate pose description based on the features
        pose_description = self._generate_pose_description(pose_features)
        
        # Identify pose-related keywords
        pose_related_keywords = self._identify_pose_related_keywords(keywords)
        
        # Create enhanced query
        enhanced_query = self._create_enhanced_query(query_data, pose_description, pose_related_keywords)
        
        logger.info(f"Matched query with pose features, identified {len(pose_related_keywords)} pose-related keywords")
        
        return {
            **query_data,
            "pose_description": pose_description,
            "pose_related_keywords": pose_related_keywords,
            "enhanced_query": enhanced_query
        }
    
    def _generate_pose_description(self, pose_features: Dict[str, torch.Tensor]) -> str:
        """
        Generate a textual description of the pose
        
        Args:
            pose_features: Pose feature data
            
        Returns:
            Textual description of the pose
        """
        # This is a placeholder - in a real implementation, you would analyze the pose features
        # to generate a detailed description
        
        # For example, you might detect specific actions or poses
        
        # Simple example description
        description = "a person standing with arms extended"
        
        return description
    
    def _identify_pose_related_keywords(self, keywords: List[str]) -> List[str]:
        """
        Identify keywords related to pose and movement
        
        Args:
            keywords: List of keywords from query
            
        Returns:
            List of pose-related keywords
        """
        # Define pose-related keyword patterns
        pose_keywords = {
            "move", "movement", "pose", "position", "stand", "standing", "sit", "sitting",
            "walk", "walking", "run", "running", "jump", "jumping", "dance", "dancing",
            "arm", "arms", "leg", "legs", "hand", "hands", "foot", "feet", "head", "body",
            "gesture", "posture", "action", "motion"
        }
        
        # Extract matching keywords
        pose_related = [keyword for keyword in keywords if keyword in pose_keywords]
        
        return pose_related
    
    def _create_enhanced_query(
        self, 
        query_data: Dict[str, any], 
        pose_description: str, 
        pose_related_keywords: List[str]
    ) -> str:
        """
        Create an enhanced query with pose context
        
        Args:
            query_data: Processed query data
            pose_description: Pose description
            pose_related_keywords: List of pose-related keywords
            
        Returns:
            Enhanced query string
        """
        original_query = query_data["original_query"]
        
        # If there are pose-related keywords, enhance the query with pose context
        if pose_related_keywords:
            return f"Based on the human pose showing {pose_description}, answer this question: {original_query}"
        else:
            # If no pose-related keywords, still add some context
            return f"With reference to the human pose in the video ({pose_description}), {original_query}"
