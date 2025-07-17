"""
Training script for RoadRep model.

This module provides functionality to train the RoadRep model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .model import RoadRepModel
from .data_loader import load_images_from_directory
from .utils import (
    set_seed, 
    get_device, 
    create_experiment_dir,
    count_parameters,
    save_config
)


class RoadRepTrainer:
    """A class to handle training of the RoadRep model."""
    
    def __init__(self, config: dict):
        """Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary containing training parameters.
        """
        self.config = config
        self.device = get_device(config.get('device'))
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Create experiment directory
        self.experiment_dir, self.checkpoints_dir, self.logs_dir = create_experiment_dir(
            base_dir=config['training']['output_dir'],
            experiment_name=config['experiment_name'],
            exist_ok=config.get('overwrite', False)
        )
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # Initialize optimizer and loss function
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()
        
        # Learning rate scheduler
        self.scheduler = self._init_scheduler()
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')  # For saving best model
        
    def _init_model(self) -> nn.Module:
        """Initialize the model."""
        # Load the ONNX model for inference
        model = RoadRepModel.load_model(
            model_path=self.config['model']['model_dir'],
            use_quantized=self.config['model'].get('use_quantized', False)
        )
        
        # If you need to modify the model for training, do it here
        # For example, you might want to extract the PyTorch model from the ONNX model
        # This is a placeholder - you'll need to adjust based on your actual model structure
        # pytorch_model = YourPyTorchModel()
        # pytorch_model.load_state_dict(extract_weights_from_onnx(model))
        # return pytorch_model
        
        return model
    
    def _init_data_loaders(self) -> tuple:
        """Initialize data loaders for training and validation."""
        # This is a placeholder - replace with your actual data loading logic
        # Example:
        # train_dataset = YourDataset(
        #     data_dir=self.config['data']['train_dir'],
        #     transform=your_transforms,
        #     **self.config['data'].get('train_kwargs', {})
        # )
        # val_dataset = YourDataset(
        #     data_dir=self.config['data']['val_dir'],
        #     transform=your_val_transforms,
        #     **self.config['data'].get('val_kwargs', {})
        # )
        # 
        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=self.config['training']['batch_size'],
        #     shuffle=True,
        #     num_workers=self.config['data'].get('num_workers', 4),
        #     pin_memory=True
        # )
        # 
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=self.config['validation']['batch_size'],
        #     shuffle=False,
        #     num_workers=self.config['data'].get('num_workers', 4),
        #     pin_memory=True
        # )
        # 
        # return train_loader, val_loader
        
        # For now, return empty data loaders as placeholders
        class DummyDataset(Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                # Return dummy data
                return torch.randn(3, 224, 224), 0
        
        dummy_loader = DataLoader(DummyDataset(), batch_size=2)
        return dummy_loader, dummy_loader
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer."""
        # This is a placeholder - replace with your actual optimizer
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.0)
        )
    
    def _init_criterion(self) -> nn.Module:
        """Initialize the loss function."""
        # This is a placeholder - replace with your actual loss function
        return nn.CrossEntropyLoss()
    
    def _init_scheduler(self):
        """Initialize the learning rate scheduler."""
        # This is a placeholder - replace with your actual scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self) -> float:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('train/loss', avg_loss, self.current_epoch)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update statistics
                total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, self.current_epoch)
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoints_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / 'checkpoint_best.pth')
    
    def train(self) -> None:
        """Run the training loop."""
        # Save config
        save_config(self.config, self.experiment_dir / 'config.yaml')
        
        # Print model summary
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Training loop
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Step the scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Check if this is the best model so far
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Best Val Loss: {self.best_metric:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RoadRep model')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the configuration file')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name of the experiment')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing experiment directory')
    
    return parser.parse_args()


def main():
    """Main function to run training."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['experiment_name'] = args.experiment_name
    if args.device is not None:
        config['device'] = args.device
    if 'seed' not in config:
        config['seed'] = args.seed
    if 'overwrite' not in config:
        config['overwrite'] = args.overwrite
    
    # Initialize and run trainer
    trainer = RoadRepTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
