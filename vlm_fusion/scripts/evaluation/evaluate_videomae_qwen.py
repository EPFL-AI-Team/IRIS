"""
VideoMAE-Qwen2.5-VL Evaluation Script
Compares fine-tuned model with baseline Qwen2.5-VL-3B-Instruct
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from transformers import (
    VideoMAEImageProcessor,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from train_videomae_qwen_contrastive_new_resampler import (
        VideoMAEQwenFusion,
        PerceiverResampler,
        MLPProjector
    )
except ImportError:
    logger.error("Cannot import model classes. Make sure train_videomae_qwen_contrastive_new_resampler.py is in PYTHONPATH")
    sys.exit(1)


class VideoTextEvalDataset(Dataset):
    """
    Dataset for evaluation on video-text pairs
    """

    def __init__(
        self,
        data_path: str,
        video_processor: VideoMAEImageProcessor,
        tokenizer: AutoTokenizer,
        num_frames: int = 16,
        instruction_prompt: str = "Document the experimental protocol observed in this video, including the actions performed and materials used.",
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.instruction_prompt = instruction_prompt

        logger.info(f"Loaded {len(self.data)} test samples from {data_path}")

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

        video_inputs = self.video_processor(frames, return_tensors="pt")
        pixel_values = video_inputs['pixel_values'].squeeze(0)

        return {
            'pixel_values': pixel_values,
            'ground_truth': item['caption'],
            'video_path': item['video_path'],
            'prompt': self.instruction_prompt,
        }


class QwenBaselineModel:
    """Wrapper for baseline Qwen2.5-VL-3B-Instruct model (without VideoMAE)"""

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        logger.info(f"Loading baseline Qwen model from {model_path}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to("cuda")
        self.model.eval()

        logger.info("Baseline model loaded successfully")

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
        Generate text from video frames using Qwen2.5-VL directly

        Args:
            frames: List of PIL images
            prompt: Text prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated text
        """
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

        inputs = self.processor(
            text=[text],
            images=None,
            videos=[frames], 
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



@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    meteor: float
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    bert_score_f1: float
    avg_length: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'BLEU-1': self.bleu1,
            'BLEU-2': self.bleu2,
            'BLEU-3': self.bleu3,
            'BLEU-4': self.bleu4,
            'METEOR': self.meteor,
            'ROUGE-1': self.rouge1_f,
            'ROUGE-2': self.rouge2_f,
            'ROUGE-L': self.rougeL_f,
            'BERTScore-F1': self.bert_score_f1,
            'Avg Length': self.avg_length,
        }


class MetricsCalculator:
    """Calculate various NLP metrics for generated text"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> EvaluationMetrics:
        """
        Compute all metrics

        Args:
            predictions: List of generated texts
            references: List of ground truth texts

        Returns:
            EvaluationMetrics object
        """
        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []
        meteor_scores = []
        rouge1_f_scores = []
        rouge2_f_scores = []
        rougeL_f_scores = []
        lengths = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            # BLEU scores
            bleu1_scores.append(sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=self.smoothing.method1
            ))
            bleu2_scores.append(sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=self.smoothing.method1
            ))
            bleu3_scores.append(sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=self.smoothing.method1
            ))
            bleu4_scores.append(sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing.method1
            ))

            # METEOR
            try:
                meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
            except:
                meteor_scores.append(0.0)

            # ROUGE
            rouge_results = self.rouge_scorer.score(ref, pred)
            rouge1_f_scores.append(rouge_results['rouge1'].fmeasure)
            rouge2_f_scores.append(rouge_results['rouge2'].fmeasure)
            rougeL_f_scores.append(rouge_results['rougeL'].fmeasure)

            # Length
            lengths.append(len(pred_tokens))

        # BERTScore (batch computation for efficiency)
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        bert_f1 = F1.mean().item()

        return EvaluationMetrics(
            bleu1=np.mean(bleu1_scores),
            bleu2=np.mean(bleu2_scores),
            bleu3=np.mean(bleu3_scores),
            bleu4=np.mean(bleu4_scores),
            meteor=np.mean(meteor_scores),
            rouge1_f=np.mean(rouge1_f_scores),
            rouge2_f=np.mean(rouge2_f_scores),
            rougeL_f=np.mean(rougeL_f_scores),
            bert_score_f1=bert_f1,
            avg_length=np.mean(lengths),
        )




def load_finetuned_model(
    checkpoint_path: str,
    videomae_model_path: str,
    qwen_model_path: str,
    device: str = "cuda"
) -> VideoMAEQwenFusion:
    """Load fine-tuned VideoMAE-Qwen model from checkpoint"""

    logger.info(f"Loading fine-tuned model from {checkpoint_path}")

    model = VideoMAEQwenFusion(
        videomae_model_path=videomae_model_path,
        qwen_model_path=qwen_model_path,
        num_query_tokens=64,
        resampler_num_heads=8,
        resampler_num_layers=2,
        mlp_hidden_dim=4096,
        mlp_num_layers=3,
        mlp_dropout=0.1,
        use_lora=False,  
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info("Fine-tuned model loaded successfully")
    return model


def evaluate_model(
    model: VideoMAEQwenFusion,
    dataset: VideoTextEvalDataset,
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Evaluate fine-tuned model on dataset

    Returns:
        predictions, references, video_paths
    """
    predictions = []
    references = []
    video_paths = []

    model.eval()

    for i in tqdm(range(len(dataset)), desc="Evaluating fine-tuned model"):
        sample = dataset[i]

        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
        prompt = sample['prompt']
        ground_truth = sample['ground_truth']

        with torch.no_grad():
            generated_text = model.generate(
                pixel_values=pixel_values,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        predictions.append(generated_text)
        references.append(ground_truth)
        video_paths.append(sample['video_path'])

    return predictions, references, video_paths


def evaluate_baseline(
    baseline_model: QwenBaselineModel,
    dataset: VideoTextEvalDataset,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Evaluate baseline Qwen model on dataset

    Returns:
        predictions, references, video_paths
    """
    predictions = []
    references = []
    video_paths = []

    for i in tqdm(range(len(dataset)), desc="Evaluating baseline model"):
        sample = dataset[i]

        # Convert pixel_values to PIL images
        # pixel_values: [num_frames, 3, 224, 224]
        pixel_values = sample['pixel_values']
        frames = []
        for j in range(pixel_values.shape[0]):
            frame = pixel_values[j].permute(1, 2, 0).numpy()
            # Assuming ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame * std + mean) * 255
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(frame))

        prompt = sample['prompt']
        ground_truth = sample['ground_truth']

        generated_text = baseline_model.generate(
            frames=frames,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        predictions.append(generated_text)
        references.append(ground_truth)
        video_paths.append(sample['video_path'])

    return predictions, references, video_paths



def plot_metrics_comparison(
    finetuned_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    output_path: str
):
    """Create bar chart comparing metrics"""

    metrics_dict_ft = finetuned_metrics.to_dict()
    metrics_dict_bl = baseline_metrics.to_dict()

    metrics_dict_ft.pop('Avg Length', None)
    metrics_dict_bl.pop('Avg Length', None)

    metric_names = list(metrics_dict_ft.keys())
    ft_values = list(metrics_dict_ft.values())
    bl_values = list(metrics_dict_bl.values())

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, ft_values, width, label='VideoMAE-Qwen (Fine-tuned)',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, bl_values, width, label='Qwen2.5-VL (Baseline)',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Metrics comparison saved to {output_path}")
    plt.close()


def plot_radar_chart(
    finetuned_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    output_path: str
):
    """Create radar chart comparing key metrics"""

    key_metrics = ['BLEU-4', 'METEOR', 'ROUGE-L', 'BERTScore-F1']

    ft_dict = finetuned_metrics.to_dict()
    bl_dict = baseline_metrics.to_dict()

    ft_values = [ft_dict[m] for m in key_metrics]
    bl_values = [bl_dict[m] for m in key_metrics]

    num_vars = len(key_metrics)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    ft_values += ft_values[:1]
    bl_values += bl_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, ft_values, 'o-', linewidth=2, label='VideoMAE-Qwen', color='#2ecc71')
    ax.fill(angles, ft_values, alpha=0.25, color='#2ecc71')

    ax.plot(angles, bl_values, 'o-', linewidth=2, label='Qwen Baseline', color='#e74c3c')
    ax.fill(angles, bl_values, alpha=0.25, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(key_metrics, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title('Multi-Metric Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Radar chart saved to {output_path}")
    plt.close()


def save_qualitative_results(
    ft_predictions: List[str],
    bl_predictions: List[str],
    references: List[str],
    video_paths: List[str],
    output_path: str,
    num_samples: int = 20
):
    """Save qualitative comparison of generations"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n  QUALITATIVE EVALUATION: VideoMAE-Qwen vs Qwen Baseline\n")

        for i in range(min(num_samples, len(references))):
            f.write(f"Sample {i+1}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Video: {video_paths[i]}\n\n")

            f.write(f"Ground Truth:\n{references[i]}\n\n")

            f.write(f"VideoMAE-Qwen (Fine-tuned):\n{ft_predictions[i]}\n\n")

            f.write(f"Qwen Baseline:\n{bl_predictions[i]}\n\n")

            f.write("=" * 100 + "\n\n")

    logger.info(f"Qualitative results saved to {output_path}")


def create_summary_report(
    finetuned_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    output_path: str
):
    """Create comprehensive text report"""

    ft_dict = finetuned_metrics.to_dict()
    bl_dict = baseline_metrics.to_dict()

    with open(output_path, 'w') as f:
        f.write("\n  EVALUATION REPORT: VideoMAE-Qwen2.5-VL vs Qwen2.5-VL Baseline\n")

        f.write("\n QUANTITATIVE RESULTS\n")

        f.write(f"{'Metric':<20} {'Fine-tuned':<15} {'Baseline':<15} {'Improvement':<15}\n")

        for metric in ft_dict.keys():
            ft_val = ft_dict[metric]
            bl_val = bl_dict[metric]

            if metric == 'Avg Length':
                improvement = ft_val - bl_val
                f.write(f"{metric:<20} {ft_val:<15.2f} {bl_val:<15.2f} {improvement:+.2f}\n")
            else:
                improvement = ((ft_val - bl_val) / bl_val * 100) if bl_val > 0 else 0
                f.write(f"{metric:<20} {ft_val:<15.4f} {bl_val:<15.4f} {improvement:+.2f}%\n")

        f.write("\n KEY FINDINGS\n")

        improvements = []
        for metric in ft_dict.keys():
            if metric != 'Avg Length':
                ft_val = ft_dict[metric]
                bl_val = bl_dict[metric]
                if bl_val > 0:
                    improvements.append((ft_val - bl_val) / bl_val * 100)

        avg_improvement = np.mean(improvements)

        f.write(f"1. Average performance improvement: {avg_improvement:+.2f}%\n\n")

        best_metric = max(ft_dict.items(), key=lambda x: x[1] if x[0] != 'Avg Length' else 0)
        f.write(f"2. Best performing metric: {best_metric[0]} = {best_metric[1]:.4f}\n\n")

        better_count = sum(1 for m in ft_dict.keys()
                          if m != 'Avg Length' and ft_dict[m] > bl_dict[m])
        total_metrics = len([m for m in ft_dict.keys() if m != 'Avg Length'])

        f.write(f"3. Fine-tuned model outperforms baseline on {better_count}/{total_metrics} metrics\n\n")

    logger.info(f"Summary report saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate VideoMAE-Qwen2.5-VL model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test annotations JSON')
    parser.add_argument('--videomae_model', type=str,
                       default='/home/benlamri/models/videomae-base-distilled',
                       help='Path to VideoMAE model')
    parser.add_argument('--qwen_model', type=str,
                       default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='Path to Qwen model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample')
    parser.add_argument('--max_new_tokens', type=int, default=250,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline evaluation (only evaluate fine-tuned model)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("\n STARTING EVALUATION")

    videomae_processor = VideoMAEImageProcessor.from_pretrained(args.videomae_model)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)

    test_dataset = VideoTextEvalDataset(
        data_path=args.test_data,
        video_processor=videomae_processor,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
    )

    finetuned_model = load_finetuned_model(
        checkpoint_path=args.checkpoint,
        videomae_model_path=args.videomae_model,
        qwen_model_path=args.qwen_model,
        device=args.device
    )

    logger.info("\n  EVALUATING FINE-TUNED MODEL")
    ft_predictions, ft_references, ft_video_paths = evaluate_model(
        model=finetuned_model,
        dataset=test_dataset,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    metrics_calculator = MetricsCalculator()
    ft_metrics = metrics_calculator.compute_metrics(ft_predictions, ft_references)

    logger.info("\nFine-tuned Model Metrics:")
    for k, v in ft_metrics.to_dict().items():
        logger.info(f"  {k}: {v:.4f}")

    if not args.skip_baseline:
        logger.info("\n  EVALUATING BASELINE MODEL")

        baseline_model = QwenBaselineModel(model_path=args.qwen_model)
        bl_predictions, bl_references, bl_video_paths = evaluate_baseline(
            baseline_model=baseline_model,
            dataset=test_dataset,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        bl_metrics = metrics_calculator.compute_metrics(bl_predictions, bl_references)

        logger.info("\nBaseline Model Metrics:")
        for k, v in bl_metrics.to_dict().items():
            logger.info(f"  {k}: {v:.4f}")

        logger.info("\n  GENERATING VISUALIZATIONS")

        plot_metrics_comparison(
            ft_metrics, bl_metrics,
            os.path.join(args.output_dir, 'metrics_comparison.png')
        )

        plot_radar_chart(
            ft_metrics, bl_metrics,
            os.path.join(args.output_dir, 'radar_chart.png')
        )

        save_qualitative_results(
            ft_predictions, bl_predictions, ft_references, ft_video_paths,
            os.path.join(args.output_dir, 'qualitative_results.txt')
        )

        create_summary_report(
            ft_metrics, bl_metrics,
            os.path.join(args.output_dir, 'evaluation_report.txt')
        )

    results = {
        'finetuned': {
            'predictions': ft_predictions,
            'references': ft_references,
            'video_paths': ft_video_paths,
            'metrics': ft_metrics.to_dict()
        }
    }

    if not args.skip_baseline:
        results['baseline'] = {
            'predictions': bl_predictions,
            'references': bl_references,
            'video_paths': bl_video_paths,
            'metrics': bl_metrics.to_dict()
        }

    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n  EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
