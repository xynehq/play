"""
Unified API Configuration for Model Flexibility

This module provides a single source of truth for API credentials and configuration.
All LLM clients use this configuration, differing only by the model parameter.

All configuration is read from environment variables with fail-fast validation.
"""

import os
import sys
from pathlib import Path

# Load .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    # Try to load .env from automation_2 directory
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
except ImportError:
    # python-dotenv not installed - skip .env loading
    pass

# Fixed model for prompt generation (ALWAYS uses minimaxai/minimax-m2)
PROMPT_GENERATION_MODEL = "minimaxai/minimax-m2"

# Default timeout for API calls (in seconds)
# Increased to 2400 seconds (40 minutes) to handle large evaluation prompts with context diffs and API latency
DEFAULT_TIMEOUT = 2400


def _get_required_env_var(var_name: str) -> str:
    """
    Get a required environment variable with fail-fast error handling.
    
    Args:
        var_name: Name of the environment variable
        
    Returns:
        Value of the environment variable
        
    Raises:
        SystemExit: If the environment variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        print(f"ERROR: Required environment variable '{var_name}' is not set.", file=sys.stderr)
        print(f"Please set {var_name} in your environment or .env file.", file=sys.stderr)
        sys.exit(1)
    return value


def get_anthropic_auth_key() -> str:
    """Get the Anthropic API authentication key from environment."""
    return _get_required_env_var("ANTHROPIC_AUTH_TOKEN")


def get_anthropic_base_url() -> str:
    """Get the Anthropic API base URL from environment."""
    return _get_required_env_var("ANTHROPIC_BASE_URL")


def get_api_config():
    """
    Get the unified API configuration from environment variables.
    
    Returns:
        Dictionary with api_key and base_url
        
    Raises:
        SystemExit: If required environment variables are not set
    """
    return {
        "api_key": get_anthropic_auth_key(),
        "base_url": get_anthropic_base_url()
    }

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
        Configured LLMClient instance with 600s default timeout
        
    Raises:
        ValueError: If model is invalid
    """
    validate_model(model)
    
    config = get_api_config()
    # Add default timeout if not provided
    if 'default_timeout' not in kwargs:
        config['default_timeout'] = DEFAULT_TIMEOUT
    config.update(kwargs)
    
    from scripts.llm_client import LLMClient
    return LLMClient(model=model, **config)
