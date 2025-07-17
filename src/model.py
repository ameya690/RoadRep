"""
Model loading and inference for RoadRep.

This module provides functionality to load the ONNX model and perform inference.
"""

import os
import onnxruntime as ort
import numpy as np
from typing import Tuple, Optional, Dict, Any
import torch

class RoadRepModel:
    """A class to handle the RoadRep ONNX model loading and inference."""
    
    def __init__(self, model_path: str, providers: Optional[list] = None):
        """Initialize the RoadRep model.
        
        Args:
            model_path: Path to the ONNX model file.
            providers: List of execution providers to use. Defaults to CUDA if available, else CPU.
        """
        self.model_path = model_path
        
        # Set up providers with fallback to CPU if CUDA is not available
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if 'CUDAExecutionProvider' in ort.get_available_providers() \
                else ['CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Store input shape for validation
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]  # Assuming NCHW format
        self.input_width = self.input_shape[3]
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the input image for the model.
        
        Args:
            image: Input image as a numpy array in HWC format (0-255).
            
        Returns:
            Preprocessed image as a numpy array in NCHW format (normalized 0-1).
        """
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        return image
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on the input image.
        
        Args:
            image: Input image as a numpy array in HWC format (0-255).
            
        Returns:
            Model predictions as a numpy array.
        """
        # Preprocess the image
        processed_img = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: processed_img}
        )
        
        return outputs[0]
    
    @staticmethod
    def load_model(model_path: str, use_quantized: bool = False) -> 'RoadRepModel':
        """Load the RoadRep model from the specified path.
        
        Args:
            model_path: Base path to the model directory.
            use_quantized: Whether to load the quantized model. Defaults to False.
            
        Returns:
            An instance of RoadRepModel.
        """
        model_name = "roadrep_clip_int8.onnx" if use_quantized else "roadrep_clip.onnx"
        full_path = os.path.join(model_path, model_name)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model not found at {full_path}")
            
        return RoadRepModel(full_path)
