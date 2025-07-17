"""
Data loading and preprocessing utilities for RoadRep.

This module provides functionality to load and preprocess images for the RoadRep model.
"""

import os
from typing import Union, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load an image from the given path.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Image as a numpy array in RGB format (H, W, 3).
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    try:
        # Read image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int], 
    keep_aspect_ratio: bool = True,
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Resize an image to the target size while optionally maintaining aspect ratio.
    
    Args:
        image: Input image as a numpy array in HWC format.
        target_size: Target size as (height, width).
        keep_aspect_ratio: Whether to maintain the aspect ratio. If True, the image will be
            resized to fit within the target dimensions while maintaining aspect ratio,
            and then padded if necessary.
        padding_color: Color to use for padding if keep_aspect_ratio is True.
        
    Returns:
        Resized (and possibly padded) image as a numpy array.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    if keep_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize while maintaining aspect ratio
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad if necessary
        if new_h < target_h or new_w < target_w:
            # Calculate padding
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            # Create padded image
            padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
            padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
            return padded
        return resized
    else:
        # Simple resize without maintaining aspect ratio
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def load_and_preprocess_image(
    image_path: str, 
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """Load and preprocess an image for the model.
    
    Args:
        image_path: Path to the image file.
        target_size: Target size as (height, width).
        keep_aspect_ratio: Whether to maintain the aspect ratio when resizing.
        
    Returns:
        Preprocessed image as a numpy array in HWC format (0-255).
    """
    # Load image
    image = load_image(image_path)
    
    # Resize image
    image = resize_image(image, target_size, keep_aspect_ratio=keep_aspect_ratio)
    
    return image


def load_images_from_directory(
    directory: str, 
    extensions: Optional[List[str]] = None,
    target_size: Optional[Tuple[int, int]] = None,
    max_images: Optional[int] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """Load all images from a directory.
    
    Args:
        directory: Path to the directory containing images.
        extensions: List of file extensions to include (e.g., ['.jpg', '.png']).
                   If None, uses common image extensions.
        target_size: If provided, resize images to this (height, width).
        max_images: Maximum number of images to load. If None, loads all images.
        
    Returns:
        A tuple of (images, filenames) where:
            - images: List of numpy arrays containing the loaded images.
            - filenames: List of corresponding filenames.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Get all image files in directory
    image_files = []
    for ext in extensions:
        image_files.extend([f for f in os.listdir(directory) 
                          if f.lower().endswith(ext)])
    
    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    # Load images
    images = []
    valid_files = []
    
    for filename in image_files:
        try:
            filepath = os.path.join(directory, filename)
            if target_size is not None:
                image = load_and_preprocess_image(filepath, target_size)
            else:
                image = load_image(filepath)
            
            images.append(image)
            valid_files.append(filename)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {str(e)}")
            continue
    
    return images, valid_files
