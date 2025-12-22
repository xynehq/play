"""
Unified API Configuration for Model Flexibility

This module provides a single source of truth for API credentials and configuration.
All LLM clients use this configuration, differing only by the model parameter.
"""

# Single source of truth for API configuration
API_CONFIG = {
    "api_key": " ",
    "base_url": "https://grid.ai.juspay.net"
}

# Fixed model for prompt generation (ALWAYS uses minimaxai/minimax-m2)
PROMPT_GENERATION_MODEL = "minimaxai/minimax-m2"

# Default timeout for API calls (in seconds)
DEFAULT_TIMEOUT = 600

def get_api_config():
    """Get the unified API configuration."""
    return API_CONFIG.copy()

def get_prompt_generation_model():
    """Get the fixed model used for prompt generation."""
    return PROMPT_GENERATION_MODEL

def validate_model(model: str) -> bool:
    """Validate that a model name is provided and non-empty.
    
    Args:
        model: Model name to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If model is invalid
    """
    if not model or not model.strip():
        raise ValueError("Model name cannot be empty or None")
    return True

def create_client(model: str, **kwargs):
    """Create an LLM client with the unified API config.
    
    Args:
        model: Model name to use
        **kwargs: Additional arguments for the LLM client
        
    Returns:
        Configured LLMClient instance
        
    Raises:
        ValueError: If model is invalid
    """
    validate_model(model)
    
    config = get_api_config()
    config.update(kwargs)
    
    from scripts.llm_client import LLMClient
    return LLMClient(model=model, **config)
