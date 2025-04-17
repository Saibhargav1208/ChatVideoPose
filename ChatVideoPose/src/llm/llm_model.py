import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple, Union
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMWrapper:
    """
    Wrapper for LLM models from HuggingFace
    """
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['llm']['model_name']
        self.device = config['llm']['device']
        self.max_length = config['llm']['max_length']
        
        # Use half precision if specified
        self.use_half_precision = config['llm']['use_half_precision']
        self.dtype = torch.float16 if self.use_half_precision else torch.float32
        
        # Create cache dir if it doesn't exist
        os.makedirs(config['system']['cache_dir'], exist_ok=True)
        
        # Load tokenizer and model
        self._load_tokenizer_and_model()
        
        logger.info(f"Initialized LLMWrapper with model {self.model_name}")
    
    def _load_tokenizer_and_model(self):
        """Load tokenizer and model from HuggingFace"""
        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.config['system']['cache_dir']
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.config['system']['cache_dir'],
                torch_dtype=self.dtype,
                device_map=self.device if self.device == "auto" else None
            )
            
            # Move model to device if not using device_map="auto"
            if self.device != "auto":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_embedding_size(self) -> int:
        """
        Get the embedding size of the model
        
        Returns:
            Embedding size
        """
        return self.model.config.hidden_size
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the model
        
        Returns:
            Vocabulary size
        """
        return len(self.tokenizer)
    
    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        pose_token_weights: Optional[torch.Tensor] = None,
        pose_token_embeddings: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> str:
        """
        Generate text based on prompt and optional pose tokens
        
        Args:
            prompt: Input prompt
            pose_token_weights: Token weights from pose analysis
            pose_token_embeddings: Token embeddings from pose analysis
            max_length: Maximum length of generated text (overrides config)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated text
        """
        # Set maximum length
        if max_length is None:
            max_length = self.max_length
        
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # The pose token integration would happen here in a real implementation
                )
            
            # Decode the generated text
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Return a single string if num_return_sequences is 1
            if num_return_sequences == 1:
                return generated_texts[0]
            
            return generated_texts
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error during generation: {str(e)}"
    
    def answer_question(self, query: str, pose_features: Dict[str, torch.Tensor]) -> str:
        """
        Answer a query based on pose features
        
        Args:
            query: User query
            pose_features: Dict containing pose feature information
            
        Returns:
            Generated answer
        """
        # In a real implementation, this would integrate the pose tokens with the LLM
        # For this example, we'll just use the query with a simple prompt
        
        # Extract pose token information
        token_weights = pose_features.get("token_weights", None)
        token_embeddings = pose_features.get("token_embeddings", None)
        
        # Create a prompt that includes context about the pose
        prompt = f"Based on the human pose information provided, answer the following question: {query}"
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            pose_token_weights=token_weights,
            pose_token_embeddings=token_embeddings,
        )
        
        return response
