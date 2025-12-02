import os
import argparse
import shutil
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_lora(lora1_path, lora2_path, output_path, weight1=0.5):
    """
    Merge two LoRA models with weighted interpolation.
    
    Args:
        lora1_path: Path to first LoRA adapter
        lora2_path: Path to second LoRA adapter
        output_path: Output path for merged model
        weight1: Weight for first model (0-1), second gets (1-weight1)
    """
    logger.info(f"Merging LoRA models with ratio {weight1:.2f}:{1-weight1:.2f}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy config from first model
    config_source = Path(lora1_path).parent / "adapter_config.json"
    config_dest = Path(output_path) / "adapter_config.json"
    shutil.copy(config_source, config_dest)
    
    # Load models
    logger.info(f"Loading LoRA 1: {lora1_path}")
    lora1_dict = load_file(lora1_path)
    
    logger.info(f"Loading LoRA 2: {lora2_path}")
    lora2_dict = load_file(lora2_path)
    
    # Verify keys match
    assert lora1_dict.keys() == lora2_dict.keys(), "LoRA models have different keys!"
    
    # Merge weights
    weight2 = 1 - weight1
    merged_dict = {
        key: weight1 * lora1_dict[key] + weight2 * lora2_dict[key]
        for key in lora1_dict
    }
    
    # Save merged model
    output_file = Path(output_path) / "adapter_model.safetensors"
    save_file(merged_dict, str(output_file))
    
    logger.info(f"Saved merged model to: {output_path}")


def batch_merge(lora1_path, lora2_path, output_dir, ratios=None, name_template=None):
    """
    Create multiple merged models with different ratios.
    
    Args:
        lora1_path: Path to first LoRA adapter
        lora2_path: Path to second LoRA adapter
        output_dir: Base output directory
        ratios: List of ratios for model 1 (default: 0.1 to 0.9)
        name_template: Template for output names (default: soup_{ratio})
    """
    if ratios is None:
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    if name_template is None:
        name_template = "soup_{ratio}"
    
    logger.info("=" * 60)
    logger.info("BATCH LORA MERGING")
    logger.info("=" * 60)
    logger.info(f"Model 1: {lora1_path}")
    logger.info(f"Model 2: {lora2_path}")
    logger.info(f"Ratios: {ratios}")
    
    for ratio in ratios:
        output_name = name_template.format(ratio=ratio)
        output_path = Path(output_dir) / output_name
        
        logger.info(f"\nMerging with ratio {ratio}...")
        merge_lora(lora1_path, lora2_path, str(output_path), weight1=ratio)
    
    logger.info("\n" + "=" * 60)
    logger.info("BATCH MERGING COMPLETE!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Merge two LoRA models")
    
    parser.add_argument(
        "--lora1",
        type=str,
        required=True,
        help="Path to first LoRA adapter (adapter_model.safetensors)"
    )
    parser.add_argument(
        "--lora2",
        type=str,
        required=True,
        help="Path to second LoRA adapter (adapter_model.safetensors)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model(s)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Weight for first model (0-1). If not set, creates multiple ratios"
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="List of ratios for batch merging"
    )
    parser.add_argument(
        "--name-template",
        type=str,
        default="soup_{ratio}",
        help="Template for output names in batch mode"
    )
    
    args = parser.parse_args()
    
    if args.ratio is not None:
        # Single merge
        merge_lora(args.lora1, args.lora2, args.output, args.ratio)
    else:
        # Batch merge
        batch_merge(
            args.lora1,
            args.lora2,
            args.output,
            args.ratios,
            args.name_template
        )


if __name__ == "__main__":
    main()