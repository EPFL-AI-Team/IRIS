"""Utility functions for model management"""

import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import VideoMAEConfig


def get_model_config(model_name: str = "MCG-NJU/videomae-base") -> Dict[str, Any]:
    """
    Get configuration for a VideoMAE model.

    Args:
        model_name: Model name or path

    Returns:
        config_dict: Dictionary of model configuration
    """
    config = VideoMAEConfig.from_pretrained(model_name)
    return config.to_dict()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        num_params: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module):
    """
    Print summary of model architecture.

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)

    # Print module breakdown
    print("\nModule breakdown:")
    for name, module in model.named_children():
        num_params = count_parameters(module)
        print(f"  {name}: {num_params:,} parameters")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **kwargs,
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **kwargs,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        device: Device to load on

    Returns:
        checkpoint: Dictionary with checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return checkpoint


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get available device.

    Args:
        prefer_gpu: If True, use GPU if available

    Returns:
        device: PyTorch device
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (forced)")

    return device


def freeze_layers(model: nn.Module, freeze_backbone: bool = True):
    """
    Freeze/unfreeze model layers.

    Args:
        model: PyTorch model
        freeze_backbone: If True, freeze backbone layers
    """
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        print(f"Backbone frozen: {trainable:,} / {total:,} parameters trainable")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")
