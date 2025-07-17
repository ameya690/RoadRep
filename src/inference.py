"""
Inference pipeline for RoadRep.

This module provides a high-level interface for running inference with the RoadRep model.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .model import RoadRepModel
from .data_loader import load_and_preprocess_image, resize_image


class RoadRepInference:
    """A class to handle end-to-end inference with the RoadRep model."""
    
    def __init__(self, model_path: str, use_quantized: bool = False):
        """Initialize the inference pipeline.
        
        Args:
            model_path: Path to the directory containing the ONNX model files.
            use_quantized: Whether to use the quantized model. Defaults to False.
        """
        # Load the model
        self.model = RoadRepModel.load_model(
            model_path=model_path,
            use_quantized=use_quantized
        )
        
        # Get input shape from model
        self.input_height = self.model.input_height
        self.input_width = self.model.input_width
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image for the model.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            Preprocessed image as a numpy array.
        """
        return load_and_preprocess_image(
            image_path=image_path,
            target_size=(self.input_height, self.input_width),
            keep_aspect_ratio=True
        )
    
    def predict(self, image_path: str) -> np.ndarray:
        """Run inference on a single image.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            Model predictions as a numpy array.
        """
        # Load and preprocess the image
        image = self.preprocess_image(image_path)
        
        # Run inference
        predictions = self.model.predict(image)
        
        return predictions
    
    def predict_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Run inference on a batch of images.
        
        Args:
            image_paths: List of paths to input images.
            
        Returns:
            List of model predictions as numpy arrays.
        """
        return [self.predict(img_path) for img_path in image_paths]
    
    def visualize_prediction(
        self, 
        image_path: str, 
        prediction: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Visualize the model's prediction on an image.
        
        Args:
            image_path: Path to the input image.
            prediction: Model prediction for the image.
            save_path: If provided, save the visualization to this path.
            show: Whether to display the visualization. Defaults to True.
        """
        # Load the original image
        image = load_and_preprocess_image(
            image_path=image_path,
            target_size=(self.input_height, self.input_width),
            keep_aspect_ratio=True
        )
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display the original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display the prediction (assuming it's a segmentation mask)
        # Adjust this based on your model's output format
        if len(prediction.shape) == 3:  # If prediction is a single-channel mask
            ax2.imshow(prediction[0], cmap='viridis')
        else:  # If prediction is multi-class
            ax2.imshow(np.argmax(prediction, axis=0), cmap='viridis')
        ax2.set_title('Prediction')
        ax2.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the visualization if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        # Show the plot if requested
        if show:
            plt.show()
        
        plt.close()


def main():
    """Example usage of the RoadRepInference class."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run inference with RoadRep model.')
    parser.add_argument('--model_dir', type=str, default='../models',
                       help='Path to the directory containing the ONNX models')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--quantized', action='store_true',
                       help='Use the quantized model')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the visualization')
    
    args = parser.parse_args()
    
    # Initialize the inference pipeline
    try:
        inference = RoadRepInference(
            model_path=args.model_dir,
            use_quantized=args.quantized
        )
        
        # Run inference
        prediction = inference.predict(args.image_path)
        
        # Visualize the result
        inference.visualize_prediction(
            image_path=args.image_path,
            prediction=prediction,
            save_path=args.output_path,
            show=True
        )
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
