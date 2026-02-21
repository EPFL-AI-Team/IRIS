"""
Prompt-Based VideoMAE-v2 Action Prediction Injection for Qwen2.5-VL
Evaluates the strategy where VideoMAE predictions are injected as textual context.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from scripts.evaluation.evaluate_videomae_qwen import (
    MetricsCalculator,
    EvaluationMetrics,
    plot_metrics_comparison,
    plot_radar_chart,
    save_qualitative_results,
    create_summary_report,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoMAEActionPredictor:
    """
    Wrapper for VideoMAE fine-tuned on action classification
    Produces top-k action predictions with probabilities
    """

    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        device: str = "cuda"
    ):
        logger.info(f"Loading VideoMAE action predictor from {model_path}")

        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

        self.processor = VideoMAEImageProcessor.from_pretrained(model_path)

        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)

        self.id2label = {v: k for k, v in class_mapping.items()}
        self.num_classes = len(self.id2label)

        self.device = device

        logger.info(f"Loaded VideoMAE with {self.num_classes} action classes")

    @torch.no_grad()
    def predict_topk(
        self,
        frames: List[Image.Image],
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Predict top-k actions with probabilities

        Args:
            frames: List of PIL images
            k: Number of top predictions to return

        Returns:
            List of (action_name, probability) tuples, sorted by probability
        """
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  

        probs = F.softmax(logits, dim=-1)[0]  

        top_k_probs, top_k_indices = torch.topk(probs, k=min(k, self.num_classes))

        # Convert to (action, prob) tuples
        predictions = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            idx = idx.item()
            prob = prob.item()
            action_name = self.id2label.get(idx, f"unknown_{idx}")
            predictions.append((action_name, prob))

        return predictions


class PromptTemplateGenerator:
    """Generate prompts with VideoMAE predictions injected as evidence"""

    def __init__(
        self,
        template_style: str = "structured_json",
        include_uncertainty: bool = True
    ):
        self.template_style = template_style
        self.include_uncertainty = include_uncertainty

    def format_predictions(
        self,
        predictions: List[Tuple[str, float]],
        format_style: str = "numbered"
    ) -> str:
        """
        Format top-k predictions as text

        Args:
            predictions: List of (action, probability) tuples
            format_style: "numbered", "bullet", or "inline"

        Returns:
            Formatted string
        """
        if format_style == "numbered":
            lines = []
            for i, (action, prob) in enumerate(predictions, 1):
                lines.append(f"{i}) {action} (confidence: {prob:.2%})")
            return "\n".join(lines)

        elif format_style == "bullet":
            lines = []
            for action, prob in predictions:
                lines.append(f"- {action} (p={prob:.3f})")
            return "\n".join(lines)

        elif format_style == "inline":
            items = [f"{action} ({prob:.2%})" for action, prob in predictions]
            return ", ".join(items)

        else:
            raise ValueError(f"Unknown format_style: {format_style}")

    def generate_prompt(
        self,
        predictions: List[Tuple[str, float]],
        segment_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Generate full prompt with VideoMAE predictions injected

        Args:
            predictions: Top-k action predictions
            segment_metadata: Optional metadata (start_time, end_time, etc.)

        Returns:
            Complete prompt string
        """
        if self.template_style == "structured_json":
            return self._generate_structured_json_prompt(predictions, segment_metadata)
        elif self.template_style == "natural_language":
            return self._generate_natural_language_prompt(predictions, segment_metadata)
        else:
            raise ValueError(f"Unknown template_style: {self.template_style}")

    def _generate_structured_json_prompt(
        self,
        predictions: List[Tuple[str, float]],
        segment_metadata: Dict[str, Any]
    ) -> str:
        """Generate structured JSON output prompt"""

        metadata_str = ""
        if segment_metadata:
            metadata_str = "Segment metadata:\n"
            if 'start_time' in segment_metadata:
                metadata_str += f"- start: {segment_metadata['start_time']:.1f}s, end: {segment_metadata['end_time']:.1f}s\n"
            if 'duration' in segment_metadata:
                metadata_str += f"- duration: {segment_metadata['duration']:.1f}s\n"
            metadata_str += "\n"

        predictions_str = self.format_predictions(predictions, format_style="numbered")

        uncertainty_instructions = ""
        if self.include_uncertainty:
            uncertainty_instructions = """
If the evidence is insufficient, inconsistent, or the confidence scores are low,
output "uncertain": true and explain why in the notes field.
"""

        prompt = f"""SYSTEM: You are an assistant documenting laboratory procedures based on video analysis.

USER:
{metadata_str}Action recognizer (VideoMAE-v2) predictions (top-{len(predictions)}):
{predictions_str}

Task:
Based on the action recognizer predictions above, produce a structured description of the laboratory step.
Output a JSON object with the following fields:
- step_id: sequential identifier
- time: timestamp or time range
- action: the primary action being performed
- manipulated_object: object being manipulated (if applicable)
- affected_object: object being affected by the action (if applicable)
- notes: additional observations or context
{uncertainty_instructions}
Respond ONLY with the JSON object, no additional text."""

        return prompt

    def _generate_natural_language_prompt(
        self,
        predictions: List[Tuple[str, float]],
        segment_metadata: Dict[str, Any]
    ) -> str:
        """Generate natural language description prompt"""

        predictions_str = self.format_predictions(predictions, format_style="inline")

        prompt = f"""SYSTEM: You are an assistant documenting laboratory procedures.

USER:
An action recognition system (VideoMAE-v2) analyzed this video segment and predicted the following actions: {predictions_str}.

Based on these predictions, describe what experimental protocol step is being performed.
Include details about the actions, materials, and objects involved.
Be concise but comprehensive."""

        return prompt


class PromptBasedEvalDataset(Dataset):
    """
    Dataset for prompt-based evaluation
    Loads video frames and metadata for action prediction + prompting
    """

    def __init__(
        self,
        data_path: str,
        video_processor: VideoMAEImageProcessor,
        num_frames: int = 16,
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.video_processor = video_processor
        self.num_frames = num_frames

        logger.info(f"Loaded {len(self.data)} samples for prompt-based evaluation")

    def __len__(self):
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
                frames.append(Image.new('RGB', (224, 224), color='black'))

        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), color='black'))

        return frames[:self.num_frames]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        frames = self.load_video_frames(item['video_path'], item['frames'])

        metadata = {
            'video_path': item['video_path'],
            'start_time': item.get('start_time', 0.0),
            'end_time': item.get('end_time', 0.0),
            'duration': item.get('duration', 0.0),
        }

        return {
            'frames': frames,
            'ground_truth': item['caption'],
            'metadata': metadata,
        }


class QwenWithPromptInjection:
    """Qwen2.5-VL model that receives VideoMAE predictions in prompt"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        logger.info(f"Loading Qwen model from {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to("cuda")
        self.model.eval()

        logger.info("Qwen model loaded successfully")

    @torch.no_grad()
    def generate(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from video frames with injected prompt

        Args:
            frames: List of PIL images
            prompt: Complete prompt (includes VideoMAE predictions)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated text
        """
        # Prepare inputs for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        # For Qwen2.5-VL, we pass frames as a video, not as images
        inputs = self.processor(
            text=[text],
            images=None,
            videos=[frames],  # Pass frames as a video
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)


        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
        )


        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text


def evaluate_prompt_based(
    action_predictor: VideoMAEActionPredictor,
    qwen_model: QwenWithPromptInjection,
    prompt_generator: PromptTemplateGenerator,
    dataset: PromptBasedEvalDataset,
    top_k: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Evaluate prompt-based injection strategy

    Returns:
        predictions, references, metadata_list
    """
    predictions = []
    references = []
    metadata_list = []

    for i in tqdm(range(len(dataset)), desc="Evaluating prompt-based injection"):
        sample = dataset[i]

        frames = sample['frames']
        ground_truth = sample['ground_truth']
        metadata = sample['metadata']

        topk_predictions = action_predictor.predict_topk(frames, k=top_k)

        prompt = prompt_generator.generate_prompt(
            predictions=topk_predictions,
            segment_metadata=metadata
        )

        generated_text = qwen_model.generate(
            frames=frames,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        predictions.append(generated_text)
        references.append(ground_truth)

        metadata_list.append({
            'video_path': metadata['video_path'],
            'videomae_predictions': topk_predictions,
            'prompt': prompt,
        })

    return predictions, references, metadata_list


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prompt-based VideoMAE injection strategy"
    )
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test annotations JSON')
    parser.add_argument('--videomae_model', type=str, required=True,
                       help='Path to fine-tuned VideoMAE model')
    parser.add_argument('--class_mapping', type=str, required=True,
                       help='Path to class_mapping.json')
    parser.add_argument('--qwen_model', type=str,
                       default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='Path to Qwen model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top-k predictions to inject')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--template_style', type=str,
                       choices=['structured_json', 'natural_language'],
                       default='structured_json',
                       help='Prompt template style')

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("\n  PROMPT-BASED VIDEOMAE INJECTION EVALUATION")

    logger.info("\n1. Loading VideoMAE action predictor...")
    action_predictor = VideoMAEActionPredictor(
        model_path=args.videomae_model,
        class_mapping_path=args.class_mapping,
        device=args.device
    )

    logger.info("\n2. Loading Qwen model...")
    qwen_model = QwenWithPromptInjection(model_path=args.qwen_model)

    logger.info("\n3. Setting up prompt generator...")
    prompt_generator = PromptTemplateGenerator(
        template_style=args.template_style,
        include_uncertainty=True
    )

    logger.info("\n4. Loading test dataset...")
    video_processor = VideoMAEImageProcessor.from_pretrained(args.videomae_model)
    dataset = PromptBasedEvalDataset(
        data_path=args.test_data,
        video_processor=video_processor,
        num_frames=args.num_frames,
    )

    # Evaluate
    logger.info(f"\n  EVALUATING WITH TOP-{args.top_k} PREDICTIONS")

    predictions, references, metadata_list = evaluate_prompt_based(
        action_predictor=action_predictor,
        qwen_model=qwen_model,
        prompt_generator=prompt_generator,
        dataset=dataset,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics_calculator = MetricsCalculator()
    metrics = metrics_calculator.compute_metrics(predictions, references)

    logger.info("\nPrompt-Based Injection Metrics:")
    for k, v in metrics.to_dict().items():
        logger.info(f"  {k}: {v:.4f}")


    logger.info("\nSaving results...")
    results = {
        'config': {
            'videomae_model': args.videomae_model,
            'qwen_model': args.qwen_model,
            'top_k': args.top_k,
            'template_style': args.template_style,
            'num_frames': args.num_frames,
        },
        'predictions': predictions,
        'references': references,
        'metadata': metadata_list,
        'metrics': metrics.to_dict()
    }

    with open(os.path.join(args.output_dir, 'prompt_based_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save qualitative examples
    with open(os.path.join(args.output_dir, 'prompt_based_examples.txt'), 'w', encoding='utf-8') as f:
        f.write("\n  PROMPT-BASED INJECTION QUALITATIVE EXAMPLES\n")

        for i in range(min(20, len(predictions))):
            f.write(f"Sample {i+1}\n")
            f.write(f"Video: {metadata_list[i]['video_path']}\n\n")

            f.write("VideoMAE Top-k Predictions:\n")
            for j, (action, prob) in enumerate(metadata_list[i]['videomae_predictions'], 1):
                f.write(f"  {j}) {action} (p={prob:.3f})\n")
            f.write("\n")

            f.write(f"Ground Truth:\n{references[i]}\n\n")
            f.write(f"Generated:\n{predictions[i]}\n\n")

    logger.info("\n  EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
