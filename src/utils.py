"""
Utility functions for the RoadRep project.

This module contains various helper functions used throughout the project.
"""

import os
import json
import yaml
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import random
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        ValueError: If the file extension is not .yaml, .yml, or .json.
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    with open(config_path, 'r') as f:
        if suffix in ('.yaml', '.yml'):
            config = yaml.safe_load(f)
        elif suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml or .json")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path to save the configuration file.
        
    Raises:
        ValueError: If the file extension is not .yaml, .yml, or .json.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = config_path.suffix.lower()
    with open(config_path, 'w') as f:
        if suffix in ('.yaml', '.yml'):
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == '.json':
            json.dump(config, f, indent=4, sort_keys=False)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml or .json")
    
    logger.info(f"Saved configuration to {config_path}")


def create_experiment_dir(
    base_dir: str,
    experiment_name: str,
    exist_ok: bool = False
) -> Tuple[Path, Path, Path]:
    """Create a directory structure for an experiment.
    
    Args:
        base_dir: Base directory for all experiments.
        experiment_name: Name of the experiment.
        exist_ok: If False, raise an error if the directory already exists.
        
    Returns:
        A tuple of (experiment_dir, checkpoints_dir, logs_dir) paths.
    """
    base_dir = Path(base_dir)
    experiment_dir = base_dir / experiment_name
    checkpoints_dir = experiment_dir / 'checkpoints'
    logs_dir = experiment_dir / 'logs'
    
    if not exist_ok and experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_dir}. "
            "Set exist_ok=True to ignore this error."
        )
    
    # Create directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory at {experiment_dir}")
    return experiment_dir, checkpoints_dir, logs_dir


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad_)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device (CPU or GPU) for PyTorch.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto-detection).
        
    Returns:
        A PyTorch device object.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    return torch.device(device)


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a numpy array.
    
    Args:
        tensor: Input tensor (PyTorch or numpy).
        
    Returns:
        Numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Expected torch.Tensor or numpy.ndarray, got {type(tensor)}")


def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    Args:
        image: Input image as a numpy array in HWC format.
        mean: Mean values for each channel.
        std: Standard deviation values for each channel.
        
    Returns:
        Normalized image.
    """
    image = image.astype(np.float32) / 255.0
    if len(image.shape) == 3:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        return (image - mean) / std
    else:
        return (image - mean[0]) / std[0]


def denormalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """Denormalize an image with mean and standard deviation.
    
    Args:
        image: Normalized image as a numpy array.
        mean: Mean values for each channel.
        std: Standard deviation values for each channel.
        
    Returns:
        Denormalized image in the range [0, 255].
    """
    if len(image.shape) == 3:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = image * std + mean
    else:
        image = image * std[0] + mean[0]
    
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)
