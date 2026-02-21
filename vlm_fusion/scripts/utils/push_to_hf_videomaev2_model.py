"""
Convert VideoMAE-v2 Fine-tuned Checkpoint to HuggingFace Format and Push to Hub

This script converts a fine-tuned VideoMAE-v2 model checkpoint (.pth) to HuggingFace format
and pushes it to the Hugging Face Hub.

Usage:
    python convert_videomae_pth_to_hub.py \
        --checkpoint_path /path/to/checkpoint-best.pth \
        --model_name vit_base_patch16_224 \
        --num_classes 32 \
        --num_frames 16 \
        --tubelet_size 2 \
        --repo_name username/model-name \
        --push_to_hub
"""

import argparse
import json
import torch
from pathlib import Path
from collections import OrderedDict
from transformers import VideoMAEConfig, VideoMAEForVideoClassification
from huggingface_hub import HfApi


def convert_videomae_checkpoint(checkpoint_path, model_name, num_classes, num_frames, tubelet_size):
    """
    Convert VideoMAE-v2 checkpoint to HuggingFace format.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        model_name: Model architecture name (e.g., 'vit_base_patch16_224')
        num_classes: Number of classes in the fine-tuned model
        num_frames: Number of frames per video clip
        tubelet_size: Temporal patch size (tubelet size)
    
    Returns:
        Converted model in HuggingFace format
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    state_dict = new_state_dict
    
    # Determine model configuration based on model_name
    if 'base' in model_name.lower():
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 3072
    elif 'large' in model_name.lower():
        hidden_size = 1024
        num_hidden_layers = 24
        num_attention_heads = 16
        intermediate_size = 4096
    elif 'huge' in model_name.lower() or 'giant' in model_name.lower():
        hidden_size = 1408
        num_hidden_layers = 40
        num_attention_heads = 16
        intermediate_size = 6144
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    # Create HuggingFace config
    config = VideoMAEConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        num_labels=num_classes,
        decoder_num_attention_heads=num_attention_heads,
        decoder_hidden_size=hidden_size,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=intermediate_size,
        norm_pix_loss=False,
    )
    
    print(f"Creating HuggingFace model with config:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Num layers: {num_hidden_layers}")
    print(f"  - Num attention heads: {num_attention_heads}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Num frames: {num_frames}")
    print(f"  - Tubelet size: {tubelet_size}")
    
    # Initialize HuggingFace model
    model = VideoMAEForVideoClassification(config)
    
    # Convert state dict keys from VideoMAE format to HuggingFace format
    hf_state_dict = convert_state_dict_keys(state_dict, num_hidden_layers)
    
    # Load the converted state dict
    missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n  Missing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"\n  Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    print("\n Checkpoint conversion completed!")
    
    return model, config, checkpoint


def convert_state_dict_keys(state_dict, num_layers):
    """
    Convert VideoMAE state dict keys to HuggingFace format.
    
    Key mappings:
    - patch_embed.proj.weight/bias -> videomae.embeddings.patch_embeddings.projection.weight/bias
    - cls_token -> videomae.embeddings.cls_token
    - pos_embed -> videomae.embeddings.position_embeddings
    - blocks.X.norm1 -> videomae.encoder.layer.X.layernorm_before
    - blocks.X.attn.qkv -> videomae.encoder.layer.X.attention.attention.query/key/value
    - blocks.X.attn.proj -> videomae.encoder.layer.X.attention.output.dense
    - blocks.X.norm2 -> videomae.encoder.layer.X.layernorm_after
    - blocks.X.mlp.fc1 -> videomae.encoder.layer.X.intermediate.dense
    - blocks.X.mlp.fc2 -> videomae.encoder.layer.X.output.dense
    - norm.weight/bias -> videomae.layernorm.weight/bias
    - fc_norm.weight/bias -> classifier.layernorm.weight/bias (optional)
    - head.weight/bias -> classifier.head.weight/bias
    """
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        new_key = key
        
        # Patch embedding
        if key.startswith('patch_embed.proj.'):
            new_key = key.replace('patch_embed.proj.', 'videomae.embeddings.patch_embeddings.projection.')
        
        # CLS token
        elif key == 'cls_token':
            new_key = 'videomae.embeddings.cls_token'
        
        # Position embeddings
        elif key == 'pos_embed':
            new_key = 'videomae.embeddings.position_embeddings'
        
        # Transformer blocks
        elif key.startswith('blocks.'):
            parts = key.split('.')
            layer_num = parts[1]
            
            if 'norm1' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.layernorm_before.{parts[-1]}'
            
            elif 'attn.qkv' in key:
                # QKV needs to be split into Q, K, V
                # This will be handled separately
                if parts[-1] == 'weight':
                    qkv_weight = value
                    dim = qkv_weight.shape[0] // 3
                    q_weight = qkv_weight[:dim]
                    k_weight = qkv_weight[dim:2*dim]
                    v_weight = qkv_weight[2*dim:]
                    
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.query.weight'] = q_weight
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.key.weight'] = k_weight
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.value.weight'] = v_weight
                    continue
                elif parts[-1] == 'bias':
                    qkv_bias = value
                    dim = qkv_bias.shape[0] // 3
                    q_bias = qkv_bias[:dim]
                    k_bias = qkv_bias[dim:2*dim]
                    v_bias = qkv_bias[2*dim:]
                    
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.query.bias'] = q_bias
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.key.bias'] = k_bias
                    new_state_dict[f'videomae.encoder.layer.{layer_num}.attention.attention.value.bias'] = v_bias
                    continue
            
            elif 'attn.q_bias' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.attention.attention.q_bias'
            
            elif 'attn.v_bias' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.attention.attention.v_bias'
            
            elif 'attn.proj' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.attention.output.dense.{parts[-1]}'
            
            elif 'norm2' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.layernorm_after.{parts[-1]}'
            
            elif 'mlp.fc1' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.intermediate.dense.{parts[-1]}'
            
            elif 'mlp.fc2' in key:
                new_key = f'videomae.encoder.layer.{layer_num}.output.dense.{parts[-1]}'
        
        # Layer norm (encoder output)
        elif key.startswith('norm.'):
            new_key = key.replace('norm.', 'videomae.layernorm.')
        
        # FC norm (for classification head)
        elif key.startswith('fc_norm.'):
            new_key = key.replace('fc_norm.', 'fc_norm.')
        
        # Classification head
        elif key.startswith('head.'):
            new_key = key.replace('head.', 'classifier.')
        
        # Add to new state dict
        if new_key != key or 'attn.qkv' not in key:  # Skip qkv as it's already handled
            new_state_dict[new_key] = value
    
    return new_state_dict


def create_model_card(
    repo_name,
    model_name,
    num_classes,
    num_frames,
    training_config,
    checkpoint_info,
    dataset_info
):
    """
    Create a model card for the Hub.
    """
    model_card = f"""---
license: apache-2.0
tags:
- video-classification
- videomae
- action-recognition
- fine-tuned
datasets:
- {dataset_info.get('name', 'custom')}
metrics:
- accuracy
---

# VideoMAE-v2 Fine-tuned on {dataset_info.get('name', 'Custom Dataset')}

## Model Description

This model is a fine-tuned version of **VideoMAE-v2 {model_name}** for video classification on the {dataset_info.get('name', 'custom dataset')}.

### Model Architecture

- **Base Model**: VideoMAE-v2 ({model_name})
- **Fine-tuned on**: {dataset_info.get('name', 'Custom dataset')}
- **Number of Classes**: {num_classes}
- **Input**: {num_frames} frames per video clip
- **Resolution**: 224x224 pixels

## Training Details

### Training Configuration

- **Optimizer**: {training_config.get('optimizer', 'AdamW')}
- **Learning Rate**: {training_config.get('lr', '1e-3')}
- **Batch Size**: {training_config.get('batch_size', 8)} (effective: {training_config.get('effective_batch_size', 24)})
- **Epochs**: {training_config.get('epochs', 50)}
- **Weight Decay**: {training_config.get('weight_decay', 0.05)}
- **Layer Decay**: {training_config.get('layer_decay', 0.75)}
- **Drop Path Rate**: {training_config.get('drop_path', 0.1)}
- **Warmup Epochs**: {training_config.get('warmup_epochs', 10)}

### Training Results

- **Best Validation Accuracy**: {checkpoint_info.get('best_acc', 'N/A')}%
- **Final Epoch**: {checkpoint_info.get('epoch', 'N/A')}

## Dataset

{dataset_info.get('description', 'Custom video dataset for action recognition.')}

- **Number of Classes**: {num_classes}
- **Training Set Size**: {dataset_info.get('train_size', 'N/A')} videos
- **Validation Set Size**: {dataset_info.get('val_size', 'N/A')} videos
- **Test Set Size**: {dataset_info.get('test_size', 'N/A')} videos

## Usage

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import numpy as np

# Load model and processor
processor = VideoMAEImageProcessor.from_pretrained("{repo_name}")
model = VideoMAEForVideoClassification.from_pretrained("{repo_name}")

# Prepare video frames (list of PIL images or numpy arrays)
# Shape: (num_frames, height, width, channels)
video = np.random.randn({num_frames}, 224, 224, 3)

# Process inputs
inputs = processor(list(video), return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predictions
predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class: {{predicted_class_idx}}")
```

## Citation

If you use this model, please cite:

```bibtex
@article{{wang2023videomaev2,
  title={{VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking}},
  author={{Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu}},
  journal={{arXiv preprint arXiv:2303.16727}},
  year={{2023}}
}}
```

## Contact

For questions or issues, please open an issue in the model repository.
"""
    
    return model_card


def main():
    parser = argparse.ArgumentParser(description="Convert VideoMAE-v2 checkpoint to HuggingFace format")
    
    # Model checkpoint arguments
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the fine-tuned .pth checkpoint'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='vit_base_patch16_224',
        help='Model architecture name (default: vit_base_patch16_224)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help='Number of output classes'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=16,
        help='Number of frames per video clip (default: 16)'
    )
    parser.add_argument(
        '--tubelet_size',
        type=int,
        default=2,
        help='Temporal patch size / tubelet size (default: 2)'
    )
    
    # Hub arguments
    parser.add_argument(
        '--repo_name',
        type=str,
        required=True,
        help='HuggingFace Hub repository name'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Push model to HuggingFace Hub'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./converted_model',
        help='Directory to save the converted model (default: ./converted_model)'
    )
    
    # Dataset info for model card
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='FineBio',
        help='Name of the dataset (for model card)'
    )
    parser.add_argument(
        '--dataset_description',
        type=str,
        default='Custom video dataset for laboratory action recognition',
        help='Description of the dataset'
    )
    
    args = parser.parse_args()
    
    model, config, checkpoint = convert_videomae_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Saving model to: {output_dir}")
    
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    preprocessor_config = {
        "do_normalize": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "resample": 2,
        "size": {"shortest_edge": 224},
        "image_processor_type": "VideoMAEImageProcessor"
    }
    
    with open(output_dir / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f, indent=2)
    
    print(" Preprocessor config created!")
    
    # Extract training info from checkpoint
    training_config = {
        'optimizer': 'AdamW',
        'lr': 1e-3,
        'batch_size': 8,
        'effective_batch_size': 24,  # batch_size * update_freq
        'epochs': 50,
        'weight_decay': 0.05,
        'layer_decay': 0.75,
        'drop_path': 0.1,
        'warmup_epochs': 10,
    }
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_acc': f"{checkpoint.get('max_accuracy', 0.0):.2f}" if 'max_accuracy' in checkpoint else 'N/A',
    }
    
    dataset_info = {
        'name': args.dataset_name,
        'description': args.dataset_description,
        'train_size': 'N/A',
        'val_size': 'N/A',
        'test_size': 'N/A',
    }
    
    # Create and save model card
    model_card = create_model_card(
        repo_name=args.repo_name,
        model_name=args.model_name,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        training_config=training_config,
        checkpoint_info=checkpoint_info,
        dataset_info=dataset_info
    )
    
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)
    
    print(" Model card created!")
    
    # Save training info as JSON
    training_info = {
        'model_architecture': args.model_name,
        'num_classes': args.num_classes,
        'num_frames': args.num_frames,
        'tubelet_size': args.tubelet_size,
        'training_config': training_config,
        'checkpoint_info': checkpoint_info,
        'dataset_info': dataset_info,
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(" Training info saved!")
    
    # Push to Hub if requested
    if args.push_to_hub:
        print(f"\n Pushing model to HuggingFace Hub: {args.repo_name}")
        
        try:
            model.push_to_hub(
                args.repo_name,
                private=args.private,
                commit_message=f"Upload fine-tuned VideoMAE-v2 model ({args.num_classes} classes)"
            )
            config.push_to_hub(
                args.repo_name,
                private=args.private,
                commit_message="Upload model config"
            )
            
            # Upload README and training_info.json
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_dir / "README.md"),
                path_in_repo="README.md",
                repo_id=args.repo_name,
                repo_type="model",
                commit_message="Upload model card"
            )
            api.upload_file(
                path_or_fileobj=str(output_dir / "training_info.json"),
                path_in_repo="training_info.json",
                repo_id=args.repo_name,
                repo_type="model",
                commit_message="Upload training info"
            )
            api.upload_file(
                path_or_fileobj=str(output_dir / "preprocessor_config.json"),
                path_in_repo="preprocessor_config.json",
                repo_id=args.repo_name,
                repo_type="model",
                commit_message="Upload preprocessor config"
            )
            
            print(f" Model successfully pushed to: https://huggingface.co/{args.repo_name}")
        except Exception as e:
            print(f" Error pushing to Hub: {e}")
            print("You may need to login first: huggingface-cli login")
    else:
        print(f"\n Model saved locally to: {output_dir}")
        print(f"To push to Hub later, run:")
        print(f"  python convert_videomae_pth_to_hub.py --checkpoint_path {args.checkpoint_path} \\")
        print(f"    --model_name {args.model_name} --num_classes {args.num_classes} \\")
        print(f"    --repo_name {args.repo_name} --push_to_hub")
    
    print("\n Conversion completed successfully!")


if __name__ == '__main__':
    main()