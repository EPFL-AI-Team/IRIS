"""Training script for VideoMAE classifier (action recognition only)"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Optional
import argparse
import sys

from ..models.videomae_classifier import VideoMAEClassifier, create_loss_fn
from ..models.model_utils import save_checkpoint, get_device, print_model_summary
from ..data.video_dataset import create_dataloaders, create_simple_dataloaders
from ..data.preprocessing import VideoPreprocessor
from ..data.augmentation import VideoAugmentation


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    accumulation_steps: int = 1,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch with gradient accumulation and mixed precision support

    Args:
        accumulation_steps: Number of batches to accumulate gradients over (simulates larger batch size)
        use_amp: Whether to use automatic mixed precision (FP16)
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, (frames, labels, metadata) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)

        # Forward pass with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(frames)
                loss = criterion(outputs["logits"], labels)
        else:
            outputs = model(frames)
            loss = criterion(outputs["logits"], labels)

        # Scale loss by accumulation steps for correct gradient averaging
        loss = loss / accumulation_steps

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights only every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Track metrics (use unscaled loss for logging)
        total_loss += loss.item() * accumulation_steps

        # Accuracy
        _, predicted = torch.max(outputs["logits"], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item() * accumulation_steps:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })

    # Final update if there are remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    metrics = {
        "train_loss": total_loss / len(train_loader),
        "train_accuracy": 100 * correct / total,
    }

    return metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels, metadata in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs["logits"], labels)

            # Track metrics
            total_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(outputs["logits"], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    metrics = {
        "val_loss": total_loss / len(val_loader),
        "val_accuracy": 100 * correct / total,
    }

    return metrics


def train_model(config_path: str):
    """
    Main training function for fine-tuning.

    Args:
        config_path: Path to configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Starting VideoMAE Classifier Fine-tuning")
    print("=" * 80)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 80)

    # Setup device
    device = get_device()

    # Create model with pretrained checkpoint
    pretrained_checkpoint = config["model"].get("pretrained_checkpoint", None)  # need to add the code to get it from HuggingFace too

    model = VideoMAEClassifier(
        model_name=config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"]["freeze_backbone"],
        pretrained=config["model"].get("pretrained", True),
        pretrained_checkpoint=pretrained_checkpoint,
    )
    model.to(device)

    print_model_summary(model)

    # Print fine-tuning strategy
    if config["model"]["freeze_backbone"]:
        print("\n[FINE-TUNING] Strategy: Freeze backbone, train classification head only")
    else:
        print("\n[FINE-TUNING] Strategy: Train all parameters (full fine-tuning)")

    
    # Create data preprocessor and augmentation
    # Applique des transformations aléatoires aux clips pendant l'entraînement pour augmenter la robustesse du modèle.
    preprocessor = VideoPreprocessor(
        num_frames=config["data"]["num_frames"],
        frame_size=config["data"]["frame_size"],
        frame_rate=config["data"]["frame_rate"],
    )

    augmentation = None
    if config["data"]["augmentation"]["enabled"]:
        augmentation = VideoAugmentation(
            horizontal_flip_prob=config["data"]["augmentation"]["horizontal_flip"],
            color_jitter_strength=config["data"]["augmentation"]["color_jitter"],
            rotation_degrees=config["data"]["augmentation"]["rotation_degrees"],
        )


    # Create dataloaders
    # Check if using simple text format or JSON annotation format
    data_format = config["data"].get("format", "json")  # default to JSON format for backward compatibility

    if data_format == "simple":
        # Use simple text list format: video_path label
        train_loader, val_loader, test_loader = create_simple_dataloaders(
            train_list_file=config["data"]["train_list"],
            val_list_file=config["data"]["val_list"],
            test_list_file=config["data"]["test_list"],
            preprocessor=preprocessor,
            augmentation=augmentation,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )
    else:
        # Use JSON annotation format with temporal annotations
        train_loader, val_loader, test_loader = create_dataloaders(
            train_annotation_file=f"{config['data']['annotation_dir']}/train.json",
            val_annotation_file=f"{config['data']['annotation_dir']}/val.json",
            test_annotation_file=f"{config['data']['annotation_dir']}/test.json",
            video_dir=config["data"]["video_dir"],
            preprocessor=preprocessor,
            augmentation=augmentation,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    

    # Create loss function
    criterion = create_loss_fn(
        label_smoothing=config["training"]["label_smoothing"],
    )

    # Create optimizer
    if config["training"]["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=config["training"]["momentum"],
            weight_decay=config["training"]["weight_decay"],
        )

    # Create scheduler
    if config["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
        )
    elif config["training"]["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=config["training"]["lr_decay_rate"],
        )
    else:
        scheduler = None

    # Get gradient accumulation and mixed precision settings
    accumulation_steps = config["training"].get("accumulation_steps", 1)
    use_amp = config["training"].get("use_amp", False)

    # Print training strategy
    if accumulation_steps > 1:
        effective_batch_size = config["data"]["batch_size"] * accumulation_steps
        print(f"\n[GRADIENT ACCUMULATION] Enabled:")
        print(f"  Physical batch size: {config['data']['batch_size']}")
        print(f"  Accumulation steps: {accumulation_steps}")
        print(f"  Effective batch size: {effective_batch_size}")

    if use_amp:
        print(f"\n[MIXED PRECISION] Enabled: Using FP16 for training")

    # Create output directories
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 80}")

        # Train with gradient accumulation and mixed precision
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
        print(f"Train Accuracy: {train_metrics['train_accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['val_accuracy']:.2f}%")

        # Save checkpoint
        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            save_path = save_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["val_loss"], save_path,
                val_accuracy=val_metrics["val_accuracy"],
                num_classes=config["model"]["num_classes"],
            )
            print(f"✓ Saved best model with val accuracy: {best_val_acc:.2f}%")

        # Save periodic checkpoint
        if epoch % config["training"]["save_frequency"] == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["val_loss"], save_path,
                num_classes=config["model"]["num_classes"],
            )

    # Final evaluation on test set
    print(f"\n{'=' * 80}")
    print("Final evaluation on test set")
    print(f"{'=' * 80}")

    test_metrics = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['val_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['val_accuracy']:.2f}%")

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    final_save_path = save_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, config["training"]["num_epochs"],
        test_metrics["val_loss"], final_save_path,
        test_accuracy=test_metrics["val_accuracy"],
        num_classes=config["model"]["num_classes"],
    )
    print(f"Final model saved to {final_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE classifier for action recognition")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    train_model(args.config)


if __name__ == "__main__":
    main()
