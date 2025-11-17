"""
VideoMAE Single-Head Architecture for Action Recognition

This module implements a single-output architecture based on VideoMAE
for technical action classification in egocentric videos.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig
from typing import Dict, Optional, Tuple


class VideoMAEClassifier(nn.Module):
    """
    VideoMAE with a single classification head for action recognition.

    Architecture:
    - VideoMAE backbone (encoder)
    - Classification head: Temporal pooling + classifier for actions
    """

    def __init__(
        self,
        model_name: str = "IRIS/videomae-base",
        num_classes: int = 12,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        pretrained: bool = True,
        pretrained_checkpoint: Optional[str] = None,
    ):
        """
        Args:
            model_name: Pretrained VideoMAE model name (for example from Hugging Face)
            num_classes: Number of action classes
            dropout: Dropout rate
            freeze_backbone: If True, freeze VideoMAE encoder weights
            pretrained: If True, load pretrained weights from HuggingFace
            pretrained_checkpoint: Path to custom pretrained checkpoint (overrides pretrained)
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained VideoMAE
        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(model_name)
        else:
            config = VideoMAEConfig.from_pretrained(model_name)
            self.backbone = VideoMAEModel(config)

        # Load custom pretrained checkpoint if provided
        if pretrained_checkpoint is not None:
            print(f"Loading pretrained backbone from {pretrained_checkpoint}")
            checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")

            # Extract backbone weights from checkpoint
            # Support multiple checkpoint formats:
            # 1. Our custom format: {"model_state_dict": {...}}
            # 2. Official VideoMAE format: {"model": {"encoder.*": ...}}
            # 3. Direct state dict: {"backbone.*": ...}
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Try different prefixes for backbone weights
            # First try "backbone." (our custom format)
            backbone_state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }

            # If not found, try "encoder." (official VideoMAE format)
            if not backbone_state_dict:
                backbone_state_dict = {
                    k.replace("encoder.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("encoder.") and not k.startswith("encoder_to_decoder")
                }

            if backbone_state_dict:
                # Load weights with strict=False to allow missing keys (e.g., decoder weights)
                missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
                print(f"✓ Loaded {len(backbone_state_dict)} backbone parameters from checkpoint")
                if missing_keys:
                    print(f"  Missing keys: {len(missing_keys)} (will use random initialization)")
                if unexpected_keys:
                    print(f"  Unexpected keys: {len(unexpected_keys)} (ignored)")
            else:
                print("Warning: No backbone weights found in checkpoint, using HuggingFace weights")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get hidden size from backbone
        self.hidden_size = self.backbone.config.hidden_size

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes),
        )

        # Initialize weights
        self._init_head()

    def _init_head(self):
        """Initialize head layers with Xavier initialization"""
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            pixel_values: Video frames [batch_size, num_frames, channels, height, width]

        Returns:
            Dictionary containing:
                - logits: [batch_size, num_classes]
                - features: [batch_size, hidden_size] (CLS token features)
        """
        # Get VideoMAE outputs
        outputs = self.backbone(pixel_values)

        # Extract CLS token (first token of last hidden state)
        cls_features = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]

        # Compute action logits
        logits = self.classification_head(cls_features)

        return {
            "logits": logits,
            "features": cls_features,
        }

    def predict(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action classes.

        Args:
            pixel_values: Video frames [batch_size, num_frames, channels, height, width]

        Returns:
            logits: [batch_size, num_classes]
            predictions: [batch_size] (class indices)
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1)
        return logits, predictions

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features from the backbone.

        Args:
            pixel_values: Video frames [batch_size, num_frames, channels, height, width]

        Returns:
            features: [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.forward(pixel_values)
        return outputs["features"]


def create_loss_fn(
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Create cross-entropy loss function for classification.

    Args:
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
        class_weights: Optional class weights for imbalanced datasets

    Returns:
        loss_fn: CrossEntropyLoss module
    """
    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        weight=class_weights,
    )


def load_model(
    checkpoint_path: str,
    num_classes: int,
    device: str = "cuda",
) -> VideoMAEClassifier:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of action classes
        device: Device to load model on

    Returns:
        model: Loaded model
    """
    model = VideoMAEClassifier(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def save_model(
    model: VideoMAEClassifier,
    save_path: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Dict] = None,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        save_path: Path to save checkpoint
        epoch: Current epoch number
        optimizer: Optional optimizer state
        metrics: Optional training metrics
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Test model
    print("Testing VideoMAEClassifier model...")

    model = VideoMAEClassifier(num_classes=15)

    # Create dummy input: [batch_size=2, num_frames=16, channels=3, height=224, width=224]
    dummy_input = torch.randn(2, 16, 3, 224, 224)

    # Forward pass
    outputs = model(dummy_input)

    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Features: {outputs['features'].shape}")

    # Test prediction
    logits, predictions = model.predict(dummy_input)
    print(f"\nPredictions: {predictions}")

    # Test loss
    loss_fn = create_loss_fn(label_smoothing=0.1)
    labels = torch.tensor([0, 1])
    loss = loss_fn(outputs["logits"], labels)

    print(f"\nLoss: {loss.item():.4f}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\nModel test passed!")
