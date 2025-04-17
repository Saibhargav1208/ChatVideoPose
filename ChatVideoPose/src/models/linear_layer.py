import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinearProjection(nn.Module):
    """
    Linear projection layer to transform Q-Former outputs to LLM-compatible space
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.in_features = config['linear_layer']['in_features']
        self.out_features = config['linear_layer']['out_features']
        
        # Linear projection
        self.linear = nn.Linear(self.in_features, self.out_features)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.out_features)
        
        logger.info(f"Initialized LinearProjection with in_features={self.in_features}, out_features={self.out_features}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Input tensor from Q-Former
                     Shape: [batch_size, num_query_tokens, in_features]
            
        Returns:
            Projected features
            Shape: [batch_size, num_query_tokens, out_features]
        """
        # Apply linear projection
        projected_features = self.linear(features)
        
        # Apply layer norm
        projected_features = self.layer_norm(projected_features)
        
        return projected_features


class SMPLTokenizer(nn.Module):
    """
    Convert projected features to SMPL tokens for LLM integration
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_tokens = config['smpl']['num_tokens']
        self.dimension = config['smpl']['dimension']
        self.out_features = config['linear_layer']['out_features']
        
        # Token embeddings
        self.token_embeddings = nn.Parameter(
            torch.zeros(self.num_tokens, self.dimension)
        )
        self._init_token_embeddings()
        
        # Projection from Q-Former output to token weights
        self.projection = nn.Linear(self.out_features, self.num_tokens)
        
        logger.info(f"Initialized SMPLTokenizer with {self.num_tokens} tokens of dimension {self.dimension}")
    
    def _init_token_embeddings(self):
        """Initialize token embeddings"""
        nn.init.normal_(self.token_embeddings, std=0.02)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert features to SMPL tokens
        
        Args:
            features: Input tensor from linear projection
                     Shape: [batch_size, num_query_tokens, out_features]
            
        Returns:
            tuple:
                - token_weights: Token weights
                  Shape: [batch_size, num_query_tokens, num_tokens]
                - token_embeddings: Token embeddings
                  Shape: [num_tokens, dimension]
        """
        # Project to token weights
        token_weights = self.projection(features)
        
        # Apply softmax to get token distribution
        token_weights = torch.softmax(token_weights, dim=-1)
        
        return token_weights, self.token_embeddings


class LLMIntegration(nn.Module):
    """
    Integration of pose tokens with LLM
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Linear projection
        self.linear_projection = LinearProjection(config)
        
        # SMPL tokenizer
        self.smpl_tokenizer = SMPLTokenizer(config)
        
        logger.info("Initialized LLMIntegration module")
    
    def forward(self, q_former_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            q_former_output: Output from Q-Former
                            Shape: [batch_size, num_query_tokens, hidden_size]
            
        Returns:
            Dictionary containing:
                - projected_features: Features after linear projection
                - token_weights: Token weights for SMPL tokens
                - token_embeddings: SMPL token embeddings
        """
        # Apply linear projection
        projected_features = self.linear_projection(q_former_output)
        
        # Convert to SMPL tokens
        token_weights, token_embeddings = self.smpl_tokenizer(projected_features)
        
        return {
            "projected_features": projected_features,
            "token_weights": token_weights,
            "token_embeddings": token_embeddings
        }
