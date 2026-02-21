"""
VideoMAEv2-Qwen2.5-VL Fusion Training Script
Implementation for video understanding with language generation
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from transformers import (
    VideoMAEModel,
    VideoMAEImageProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Helper Function to Detect Qwen Version

def get_qwen_model_class(model_path: str):
    """
    Detect Qwen version from model path and return the appropriate model class

    This fixes the size mismatch issue where Qwen2VL and Qwen2.5VL have different
    architectures but similar names.

    Args:
        model_path: Path to Qwen model (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")

    Returns:
        Appropriate Qwen model class (Qwen2VLForConditionalGeneration or Qwen2_5_VLForConditionalGeneration)
    """
    model_path_lower = model_path.lower()

    if any(x in model_path_lower for x in ['2.5', '2_5', 'qwen25', 'qwen2.5']):
        logger.info("Detected Qwen 2.5 model - using Qwen2_5_VLForConditionalGeneration")
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        logger.info("Detected Qwen 2.0 model - using Qwen2VLForConditionalGeneration")
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


# Perceiver Resampler

class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler to compress VideoMAE tokens (1568) to a fixed number (e.g., 64).
    Uses learnable query tokens and cross-attention to compress the sequence.

    Architecture:
        Learnable Queries [num_queries, dim] + Cross-Attention(Q=queries, K=V=video_tokens) -> Compressed representation [num_queries, dim]
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Hidden dimension 
            num_queries: Number of output tokens 
            num_heads: Number of attention heads
            num_layers: Number of resampler layers
            ff_dim: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.num_queries = num_queries
        self.dim = dim

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Resampler layers (cross-attention + feedforward)
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_tokens, dim] , [B, 1568, 768]
        Returns:
            [batch_size, num_queries, dim] , [B, 64, 768]
        """
        batch_size = x.shape[0]

        # Expand queries for batch: [num_queries, dim] -> [batch, num_queries, dim]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply resampler layers
        for layer in self.layers:
            queries = layer(queries, x)

        queries = self.norm(queries)

        return queries


class PerceiverResamplerLayer(nn.Module):
    """Single Perceiver resampler layer with cross-attention and feedforward"""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(dim)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [batch, num_queries, dim]
            context: [batch, num_tokens, dim] - input tokens to attend to
        Returns:
            [batch, num_queries, dim]
        """
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=queries,
            key=context,
            value=context,
            need_weights=False
        )
        queries = self.cross_attn_norm(queries + attn_out)

        # Self-attention
        attn_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
            need_weights=False
        )
        queries = self.self_attn_norm(queries + attn_out)

        # Feedforward
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)

        return queries


# MLP Connector

class MLPProjector(nn.Module):
    """
    Multi-layer perceptron to project VideoMAEv2 features to Qwen embedding space.
    Uses GELU activation and layer normalization for stable training.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 4096,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.projector = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for stable training"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_tokens, input_dim]
        Returns:
            [batch_size, num_tokens, output_dim]
        """
        return self.projector(x)


# VideoMAEv2-Qwen Fusion Model

class VideoMAEQwenFusion(nn.Module):
    """
    Fusion model combining VideoMAEv2 encoder with Qwen2.5-VL for video understanding.

    Architecture:
        VideoMAEv2 (frozen) -> Perceiver Resampler (trainable) -> MLP Projector (trainable) -> Qwen2.5-VL (frozen)

    Training approach:
        - VideoMAE encoder: Frozen (pre-trained)
        - Perceiver Resampler: Trainable (compresses 1568 tokens -> 64 tokens)
        - MLP Projector: Trainable (projects to Qwen embedding space)
        - Qwen2.5-VL: Frozen (pre-trained)
    """

    def __init__(
        self,
        videomae_model_path: str,
        qwen_model_path: str,
        num_query_tokens: int = 64,
        resampler_num_heads: int = 8,
        resampler_num_layers: int = 2,
        mlp_hidden_dim: int = 4096,
        mlp_num_layers: int = 3,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        logger.info(f"Loading VideoMAEv2 from {videomae_model_path}")
        self.video_encoder = VideoMAEModel.from_pretrained(
            videomae_model_path,
            torch_dtype=torch.float32
        )
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        self.video_encoder.eval()

        logger.info(f"Loading Qwen model from {qwen_model_path}")
        QwenModelClass = get_qwen_model_class(qwen_model_path)
        logger.info(f"Using model class: {QwenModelClass.__name__}")

        self.language_model = QwenModelClass.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None, 
            trust_remote_code=True 
        )

        if hasattr(self.language_model, 'gradient_checkpointing_enable'):
            self.language_model.gradient_checkpointing_enable()

        # Freeze Qwen 
        for param in self.language_model.parameters():
            param.requires_grad = False
        self.language_model.eval()

        # Get embedding dimensions
        videomae_dim = self.video_encoder.config.hidden_size
        qwen_dim = self.language_model.get_input_embeddings().embedding_dim

        logger.info(f"VideoMAE hidden dim: {videomae_dim}")
        logger.info(f"Qwen embedding dim: {qwen_dim}")

        # Perceiver Resampler
        self.resampler = PerceiverResampler(
            dim=videomae_dim,
            num_queries=num_query_tokens,
            num_heads=resampler_num_heads,
            num_layers=resampler_num_layers,
            ff_dim=videomae_dim * 2,
            dropout=mlp_dropout
        )
        logger.info(f"Perceiver Resampler: {1568} tokens -> {num_query_tokens} tokens")

        # MLP Projector 
        self.mlp_projector = MLPProjector(
            input_dim=videomae_dim,
            output_dim=qwen_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout
        )


        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        # Special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames using frozen VideoMAEv2 encoder

        Args:
            pixel_values: [batch_size, num_frames, channels, height, width]
        Returns:
            video_features: [batch_size, num_patches, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.video_encoder(pixel_values=pixel_values)
            video_features = outputs.last_hidden_state  # [B, num_patches, 768]

        return video_features

    def contrastive_loss(self, vid_embeds: torch.Tensor, txt_embeds: torch.Tensor, temperature: float = 0.07):
        """
        Compute bidirectional contrastive loss between video and text embeddings

        Args:
            vid_embeds: [B, D] - normalized video embeddings
            txt_embeds: [B, D] - normalized text embeddings
            temperature: temperature parameter for scaling logits

        Returns:
            Contrastive loss value
        """
        # vid_embeds: [B, D], txt_embeds: [B, D] (already normalized)
        logits = (vid_embeds @ txt_embeds.T) / temperature  # [B, B]
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_i2t + loss_t2i)

    def contrastive_accuracy(self, vid_embeds: torch.Tensor, txt_embeds: torch.Tensor):
        """
        Compute video-text retrieval accuracy

        Args:
            vid_embeds: [B, D] - normalized video embeddings
            txt_embeds: [B, D] - normalized text embeddings

        Returns:
            Dictionary with v2t_acc (video-to-text) and t2v_acc (text-to-video) accuracies
        """
        similarity = vid_embeds @ txt_embeds.T

        # Video-to-text retrieval
        v2t_predictions = similarity.argmax(dim=1)  # [B]
        targets = torch.arange(similarity.size(0), device=similarity.device)
        v2t_acc = (v2t_predictions == targets).float().mean()

        # Text-to-video retrieval
        t2v_predictions = similarity.argmax(dim=0)  # [B]
        t2v_acc = (t2v_predictions == targets).float().mean()

        return {
            'v2t_acc': v2t_acc, 
            't2v_acc': t2v_acc, 
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        contrastive_alpha: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete fusion model

        Args:
            pixel_values: [batch_size, num_frames, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for language modeling loss
            contrastive_alpha: weight for contrastive loss (default: 0.1)

        Returns:
            Dictionary containing loss, logits, lm_loss, nce_loss, and metrics
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # 1. Encode video features
        video_features = self.encode_video(pixel_values)  # [B, 1568, 768]

        # 2. Compress with Perceiver Resampler
        video_features_compressed = self.resampler(video_features)  # [B, num_query_tokens, 768]

        # 3. Project to Qwen embedding space
        video_embeds = self.mlp_projector(video_features_compressed)  # [B, num_query_tokens, qwen_dim]

        # Convert to bfloat16 to match Qwen's dtype
        video_embeds = video_embeds.to(torch.bfloat16)

        # 4. Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # [B, seq_len, qwen_dim]

        # Concatenate video embeddings (prefix) with text embeddings
        inputs_embeds = torch.cat([video_embeds, text_embeds], dim=1)  # [B, num_patches+seq_len, qwen_dim]

        # 5. Extend attention mask for video tokens
        num_video_tokens = video_embeds.shape[1]
        video_attention_mask = torch.ones(
            batch_size, num_video_tokens,
            dtype=attention_mask.dtype,
            device=device
        )
        extended_attention_mask = torch.cat([video_attention_mask, attention_mask], dim=1)

        # 6. Extend labels (video tokens are masked out with -100)
        if labels is not None:
            video_labels = torch.full(
                (batch_size, num_video_tokens),
                -100,
                dtype=labels.dtype,
                device=device
            )
            extended_labels = torch.cat([video_labels, labels], dim=1)
        else:
            extended_labels = None

        # 7. Forward through Qwen
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            output_hidden_states=True, 
            return_dict=True
        )

        lm_loss = outputs.loss

        # 8. Pool video/text embeddings for contrastive alignment
        num_video_tokens = video_embeds.shape[1]
        hidden = outputs.hidden_states[-1]  # [B, V+T, D]
        pooled_video = F.normalize(hidden[:, :num_video_tokens].mean(dim=1), dim=-1)
        pooled_text = F.normalize(hidden[:, num_video_tokens:].mean(dim=1), dim=-1)

        # 9. Compute contrastive loss
        nce_loss = self.contrastive_loss(pooled_video, pooled_text)
        total_loss = lm_loss + contrastive_alpha * nce_loss

        # 10. Compute accuracy metrics
        with torch.no_grad():
            # Token-level accuracy (language modeling)
            predictions = outputs.logits.argmax(dim=-1)  # [B, seq_len]

            # Only compute accuracy on text tokens (not video tokens)
            text_predictions = predictions[:, num_video_tokens:]  # [B, text_len]
            text_labels = extended_labels[:, num_video_tokens:]  # [B, text_len]

            # Mask out padding tokens (-100)
            valid_mask = (text_labels != -100)

            if valid_mask.sum() > 0:
                correct = (text_predictions[valid_mask] == text_labels[valid_mask]).float()
                accuracy = correct.mean()
                perplexity = torch.exp(lm_loss)
            else:
                accuracy = torch.tensor(0.0, device=device)
                perplexity = torch.tensor(0.0, device=device)

            # Contrastive retrieval accuracy
            retrieval_metrics = self.contrastive_accuracy(pooled_video, pooled_text)

        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "lm_loss": lm_loss,
            "nce_loss": nce_loss,
            "accuracy": accuracy,
            "perplexity": perplexity,
            "v2t_retrieval_acc": retrieval_metrics['v2t_acc'],
            "t2v_retrieval_acc": retrieval_metrics['t2v_acc'],
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1,
    ) -> str:
        """
        Generate text response given video and prompt

        Args:
            pixel_values: [1, num_frames, 3, 224, 224]
            prompt: Text prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search

        Returns:
            Generated text string
        """
        self.eval()
        device = pixel_values.device

        # Encode video with resampler
        video_features = self.encode_video(pixel_values)  # [1, 1568, 768]
        video_features_compressed = self.resampler(video_features)  # [1, num_query_tokens, 768]
        video_embeds = self.mlp_projector(video_features_compressed).to(torch.bfloat16)  # [1, num_query_tokens, qwen_dim]

        # Tokenize prompt
        text_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(text_inputs.input_ids)

        # Combine embeddings
        inputs_embeds = torch.cat([video_embeds, text_embeds], dim=1)

        # Extend attention mask
        num_video_tokens = video_embeds.shape[1]
        video_attention_mask = torch.ones(1, num_video_tokens, dtype=torch.long, device=device)
        attention_mask = torch.cat([video_attention_mask, text_inputs.attention_mask], dim=1)

        # Store prompt length (video tokens + prompt tokens) to exclude from output
        num_prompt_tokens = num_video_tokens + text_inputs.input_ids.shape[1]

        # Generate
        output_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=(temperature > 0),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the generated part (exclude prompt)
        # output_ids contains: [video tokens (implicit)] + [prompt tokens] + [generated tokens]
        # Since we used inputs_embeds, output_ids only contains the generated sequence
        # We need to skip the prompt tokens
        generated_ids = output_ids[0, num_prompt_tokens:]  # Skip video + prompt tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text


# Dataset

class VideoTextDataset(Dataset):
    """
    Dataset for video-text pairs with instruction-following structure

    Expected data format (JSON):
    [
        {
            "video_path": "path/to/video_frames_dir",
            "frames": ["frame_0001.jpg", "frame_0002.jpg", ...],  # 16 frames
            "caption": "Description of the video (ground truth to generate)"
        },
        ...
    ]

    Training structure (teacher forcing):
        input_ids:  [PROMPT tokens] + [CAPTION tokens] + [PAD]
        labels:     [-100 ...     ] + [CAPTION tokens] + [-100]
                    no loss           loss here           no loss on padding

    At inference:
        input: [VIDEO] + [PROMPT] -> model generates CAPTION
    """

    def __init__(
        self,
        data_path: str,
        video_processor: VideoMAEImageProcessor,
        tokenizer: AutoTokenizer,
        num_frames: int = 16,
        max_text_length: int = 256,
        instruction_prompt: str = "Document the experimental protocol observed in this video, including the actions performed and materials used.",
    ):
        super().__init__()

        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_text_length = max_text_length
        self.instruction_prompt = instruction_prompt

        logger.info(f"Loaded {len(self.data)} video-text pairs from {data_path}")
        logger.info(f"Using instruction prompt: '{self.instruction_prompt}'")

    def __len__(self) -> int:
        return len(self.data)

    def load_video_frames(self, video_path: str, frame_names: List[str]) -> List[Image.Image]:
        """Load video frames from disk"""
        frames = []
        for frame_name in frame_names[:self.num_frames]:
            frame_path = os.path.join(video_path, frame_name)
            try:
                img = Image.open(frame_path).convert('RGB')
                frames.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {frame_path}: {e}")
                # Create blank frame as fallback
                frames.append(Image.new('RGB', (224, 224), color='black'))

        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), color='black'))

        return frames[:self.num_frames]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        frames = self.load_video_frames(item['video_path'], item['frames'])
        video_inputs = self.video_processor(frames, return_tensors="pt")
        pixel_values = video_inputs['pixel_values'].squeeze(0)  # [num_frames, 3, 224, 224]
        caption = item['caption']

        prompt_tokens = self.tokenizer(
            self.instruction_prompt,
            add_special_tokens=True,
            return_tensors="pt"
        )

        caption_tokens = self.tokenizer(
            caption,
            add_special_tokens=False,  
            return_tensors="pt"
        )

        prompt_ids = prompt_tokens['input_ids'].squeeze(0)
        caption_ids = caption_tokens['input_ids'].squeeze(0)

        input_ids = torch.cat([prompt_ids, caption_ids], dim=0)

        # Truncate if too long
        if len(input_ids) > self.max_text_length:
            # Keep all prompt, truncate caption
            caption_max_len = self.max_text_length - len(prompt_ids)
            caption_ids = caption_ids[:caption_max_len]
            input_ids = torch.cat([prompt_ids, caption_ids], dim=0)

        # Pad to max_text_length
        pad_length = self.max_text_length - len(input_ids)
        if pad_length > 0:
            pad_token_id = self.tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id, dtype=torch.long)], dim=0)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Create labels: -100 for prompt tokens, real token_ids for caption tokens
        labels = input_ids.clone()
        # Mask prompt and padding tokens with -100
        labels[:len(prompt_ids)] = -100
        labels[attention_mask == 0] = -100

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader"""
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


# PyTorch Lightning Module

class VideoMAEQwenLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training VideoMAEQwenFusion model
    Training connector layers (Perceiver Resampler + MLP Projector) only
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create model
        self.model = VideoMAEQwenFusion(
            videomae_model_path=config['model']['videomae_model_path'],
            qwen_model_path=config['model']['qwen_model_path'],
            num_query_tokens=config['model'].get('num_query_tokens', 64),
            resampler_num_heads=config['model'].get('resampler_num_heads', 8),
            resampler_num_layers=config['model'].get('resampler_num_layers', 2),
            mlp_hidden_dim=config['model']['mlp_hidden_dim'],
            mlp_num_layers=config['model']['mlp_num_layers'],
            mlp_dropout=config['model']['mlp_dropout'],
        )

        # Get training config
        self.learning_rate = float(config['training']['learning_rate'])
        self.warmup_steps = int(config['training']['warmup_steps'])
        self.weight_decay = float(config['training']['weight_decay'])
        # Contrastive learning weight (default: 0.1)
        self.contrastive_alpha = float(config['training'].get('contrastive_alpha', 0.1))

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch, contrastive_alpha=self.contrastive_alpha)
        loss = outputs['loss']

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/lm_loss', outputs['lm_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/nce_loss', outputs['nce_loss'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/accuracy', outputs['accuracy'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/perplexity', outputs['perplexity'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/v2t_retrieval_acc', outputs['v2t_retrieval_acc'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/t2v_retrieval_acc', outputs['t2v_retrieval_acc'], prog_bar=False, on_step=True, on_epoch=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch, contrastive_alpha=self.contrastive_alpha)
        loss = outputs['loss']

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/lm_loss', outputs['lm_loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/nce_loss', outputs['nce_loss'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('val/accuracy', outputs['accuracy'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/perplexity', outputs['perplexity'], prog_bar=False, on_step=False, on_epoch=True)
        self.log('val/v2t_retrieval_acc', outputs['v2t_retrieval_acc'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/t2v_retrieval_acc', outputs['t2v_retrieval_acc'], prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Train Perceiver Resampler + MLP Projector only
        params = []
        params.extend(list(self.model.resampler.parameters()))
        params.extend(list(self.model.mlp_projector.parameters()))

        # Filter out non-trainable parameters
        params = [p for p in params if p.requires_grad]

        logger.info(f"Training {len(params)} parameter groups (Perceiver Resampler + MLP Projector)")

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


# Training Script

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]):
    """Setup logging (WandB or TensorBoard)"""
    log_dir = config['output']['log_dir']

    if config['logging']['use_wandb']:
        logger_instance = WandbLogger(
            project=config['logging']['wandb_project'],
            name=config['logging']['experiment_name'],
            save_dir=log_dir,
            log_model=False,
        )
    else:
        logger_instance = TensorBoardLogger(
            save_dir=log_dir,
            name="training",
        )

    return logger_instance


def train(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None
):
    """
    Train the VideoMAE-Qwen fusion model

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint to resume from (optional)
    """
    logger.info(f"{'='*80}")
    logger.info(f"Starting Training")
    logger.info(f"{'='*80}")

    pl.seed_everything(config['system']['seed'])

    videomae_processor = VideoMAEImageProcessor.from_pretrained(
        config['model']['videomae_model_path']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['qwen_model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    instruction_prompt = config['data'].get(
        'instruction_prompt',
        "Document the experimental protocol observed in this video, including the actions performed and materials used."
    )

    train_dataset = VideoTextDataset(
        data_path=config['data']['train_annotation_file'],
        video_processor=videomae_processor,
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        max_text_length=config['data']['max_text_length'],
        instruction_prompt=instruction_prompt,
    )

    val_dataset = VideoTextDataset(
        data_path=config['data']['val_annotation_file'],
        video_processor=videomae_processor,
        tokenizer=tokenizer,
        num_frames=config['data']['num_frames'],
        max_text_length=config['data']['max_text_length'],
        instruction_prompt=instruction_prompt,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = VideoMAEQwenLightningModule(config)

    # Load checkpoint if provided 
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['output']['checkpoint_dir'],
        filename='videomae-qwen-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback, lr_monitor]

    if config['training']['early_stopping']['enabled']:
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=config['training']['early_stopping']['patience'],
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    logger_instance = setup_logging(config)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator=config['system']['accelerator'],
        devices=config['system']['devices'],
        precision=config['system']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
        val_check_interval=config['training']['val_check_interval'],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        callbacks=callbacks,
        logger=logger_instance,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info(f"Training completed!")
    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="VideoMAEv2-Qwen2.5-VL Fusion Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)

    # Train model
    best_checkpoint = train(config, checkpoint_path=args.resume)
    logger.info(f"\nTraining Complete. Best checkpoint: {best_checkpoint}")


if __name__ == '__main__':
    main()
