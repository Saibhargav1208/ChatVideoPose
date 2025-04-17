import torch
import torch.nn as nn
import math
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionalEmbedding(nn.Module):
    """
    Positional embedding module to provide temporal context to pose features
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config['embedding']['hidden_dim']
        self.max_position_embeddings = config['embedding']['max_position_embeddings']
        self.dropout = nn.Dropout(config['embedding']['dropout'])
        
        # Create position embeddings
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_dim)
        
        # Initialize with sinusoidal embeddings
        self._init_weights()
        
        logger.info(f"Initialized PositionalEmbedding with hidden_dim={self.hidden_dim}")
    
    def _init_weights(self):
        """Initialize with sinusoidal embeddings"""
        position = torch.arange(self.max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * -(math.log(10000.0) / self.hidden_dim))
        
        # Create sinusoidal pattern
        pe = torch.zeros(self.max_position_embeddings, self.hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Register as buffer (won't be trained)
        self.register_buffer('pe', pe)
        
        # Copy to embedding weights (will be fine-tuned)
        with torch.no_grad():
            self.position_embeddings.weight.copy_(pe)
    
    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to the pose features
        
        Args:
            pose_features: Tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            Tensor with positional information added
        """
        seq_len = pose_features.size(1)
        
        if seq_len > self.max_position_embeddings:
            logger.warning(f"Input sequence length {seq_len} exceeds maximum position embeddings " 
                          f"{self.max_position_embeddings}. Truncating sequence.")
            pose_features = pose_features[:, :self.max_position_embeddings, :]
            seq_len = self.max_position_embeddings
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=pose_features.device).unsqueeze(0)
        
        # Get position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        
        # Input shape: [batch_size, seq_len, feature_dim]
        # Position shape: [1, seq_len, hidden_dim]
        
        # Project pose features to match hidden dimension if needed
        if pose_features.size(-1) != self.hidden_dim:
            pose_features = self.project_features(pose_features)
        
        # Add positional embeddings
        embeddings = pose_features + position_embeddings
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to match the hidden dimension
        
        Args:
            features: Tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            Projected features of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, feature_dim = features.size()
        
        # Lazily create the projection layer when needed
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(feature_dim, self.hidden_dim).to(features.device)
            logger.info(f"Created projection layer from {feature_dim} to {self.hidden_dim}")
        
        # Project features
        return self.projection(features)


class TemporalPositionalEmbedding(PositionalEmbedding):
    """
    Enhanced positional embedding with temporal awareness for motion sequences
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        # Additional temporal modeling components can be added here
        
        # For example, relative position bias
        self.use_relative_positions = True
        if self.use_relative_positions:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.max_position_embeddings - 1), self.hidden_dim)
            )
            self._init_relative_position_bias()
            
        logger.info("Initialized TemporalPositionalEmbedding with relative position bias")
    
    def _init_relative_position_bias(self):
        """Initialize relative position bias with sinusoidal pattern"""
        position = torch.arange(-(self.max_position_embeddings - 1), self.max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * -(math.log(10000.0) / self.hidden_dim))
        
        # Create sinusoidal pattern
        pe = torch.zeros(2 * self.max_position_embeddings - 1, self.hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        # Copy to relative position bias table
        with torch.no_grad():
            self.relative_position_bias_table.copy_(pe)
    
    def get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """
        Calculate relative position bias
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position bias tensor
        """
        # Calculate position indices
        position_ids = torch.arange(seq_len, device=self.relative_position_bias_table.device)
        
        # Calculate relative position indices
        relative_position_indices = position_ids.unsqueeze(1) - position_ids.unsqueeze(0)
        
        # Shift to make all indices non-negative
        relative_position_indices = relative_position_indices + (self.max_position_embeddings - 1)
        
        # Look up relative position bias
        relative_position_bias = self.relative_position_bias_table[relative_position_indices]
        
        return relative_position_bias
    
    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional embeddings to the pose features
        
        Args:
            pose_features: Tensor of shape [batch_size, seq_len, feature_dim]
            
        Returns:
            Tensor with positional information added
        """
        # Get basic positional embeddings from parent class
        embeddings = super().forward(pose_features)
        
        # Add relative position bias if enabled
        if self.use_relative_positions:
            seq_len = embeddings.size(1)
            relative_position_bias = self.get_relative_position_bias(seq_len)
            
            # Apply relative position bias (as multiplicative attention)
            # This is a simplified version - in practice, you'd integrate this into attention
            embeddings = embeddings * (1 + 0.1 * torch.mean(relative_position_bias, dim=-1).unsqueeze(0).unsqueeze(-1))
        
        return embeddings
