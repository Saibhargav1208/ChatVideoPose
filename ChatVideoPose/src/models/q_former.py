import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_size = config['q_former']['hidden_size']
        self.num_attention_heads = config['q_former']['num_attention_heads']
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, key, value projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config['q_former']['dropout'])
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape and transpose tensor for attention computation
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Reshaped tensor of shape [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        cross_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            cross_hidden_states: Cross-attention inputs
            
        Returns:
            Output tensor
        """
        # Determine if this is self-attention or cross-attention
        if cross_hidden_states is None:
            # Self-attention
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            # Cross-attention
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(cross_hidden_states)
            mixed_value_layer = self.value(cross_hidden_states)
        
        # Transpose and reshape
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.dropout(attention_probs)
        
        # Calculate context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape to [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output_layer = self.output(context_layer)
        
        return output_layer


class FeedForward(nn.Module):
    """
    Feed-forward layer
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_size = config['q_former']['hidden_size']
        self.intermediate_size = config['q_former']['intermediate_size']
        
        self.dense1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config['q_former']['dropout'])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor
        """
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class TransformerLayer(nn.Module):
    """
    Transformer layer
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_size = config['q_former']['hidden_size']
        
        # Self-attention
        self.self_attention = MultiHeadAttention(config)
        self.self_attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(config)
        self.cross_attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Feed-forward
        self.feed_forward = FeedForward(config)
        self.feed_forward_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Layer norm
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout = nn.Dropout(config['q_former']['dropout'])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Input tensor
            attention_mask: Self-attention mask
            cross_hidden_states: Cross-attention inputs
            cross_attention_mask: Cross-attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        self_attention_output = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + self.dropout(self_attention_output)
        
        # Cross-attention (if cross_hidden_states is provided)
        if cross_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            cross_attention_output = self.cross_attention(
                hidden_states=hidden_states,
                attention_mask=cross_attention_mask,
                cross_hidden_states=cross_hidden_states,
            )
            hidden_states = residual + self.dropout(cross_attention_output)
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.feed_forward_norm(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(feed_forward_output)
        
        return hidden_states


class QFormer(nn.Module):
    """
    Q-Former model for pose feature transformation
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_size = config['q_former']['hidden_size']
        self.num_hidden_layers = config['q_former']['num_hidden_layers']
        self.num_query_tokens = config['q_former']['num_query_tokens']
        
        # Query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, self.hidden_size)
        )
        self._init_query_tokens()
        
        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(self.num_hidden_layers)]
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        logger.info(f"Initialized QFormer with {self.num_hidden_layers} layers")
    
    def _init_query_tokens(self):
        """Initialize query tokens"""
        nn.init.normal_(self.query_tokens, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            hidden_states: Input tensor from positional embedding
                           Shape: [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
                           Shape: [batch_size, 1, 1, seq_len]
            
        Returns:
            Query features
            Shape: [batch_size, num_query_tokens, hidden_size]
        """
        batch_size = hidden_states.size(0)
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Process through transformer layers
        for layer in self.layers:
            query_tokens = layer(
                hidden_states=query_tokens,
                cross_hidden_states=hidden_states,
                cross_attention_mask=attention_mask,
            )
        
        # Apply final layer norm
        query_tokens = self.layer_norm(query_tokens)
        
        return query_tokens
