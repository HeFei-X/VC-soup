
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward model evaluation."""
    model1_path: str
    model2_path: str
    dataset_dir: str = "./RLHF/dataset"
    output_dir: str = "./outputs/reward_evaluation"
    gpu_id: str = "0"
    seed: int = 60
    torch_dtype: torch.dtype = torch.bfloat16


class RewardModelEvaluator:
    """Evaluates consistency between two reward models on preference data."""
    
    def __init__(self, config: RewardModelConfig):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: RewardModelConfig containing model paths and settings
        """
        self.config = config
        
        # Set GPU and seed
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
        set_seed(config.seed)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load models
        logger.info(f"Loading reward models...")
        self.rm1, self.rm1_tokenizer = self._load_model(config.model1_path)
        self.rm2, self.rm2_tokenizer = self._load_model(config.model2_path)
        logger.info("Models loaded successfully")
    
    def _load_model(self, model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load a reward model and its tokenizer.
        
        Args:
            model_path: Path or identifier for the model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=self.config.torch_dtype,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def compute_reward(
        self, 
        query: str, 
        response: str,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer
    ) -> torch.Tensor:
        """
        Compute reward score for a query-response pair.
        
        Args:
            query: Input prompt (currently unused, kept for compatibility)
            response: Response text to evaluate
            model: Reward model
            tokenizer: Corresponding tokenizer
            
        Returns:
            Reward score as a tensor
        """
        # Note: Currently only using response, can be modified to include query
        inputs = tokenizer(response, return_tensors="pt", truncation=True).to("cuda")
        
        with torch.no_grad():
            reward = model(**inputs).logits[0].item()
        
        return torch.tensor(reward)
    
    def evaluate_dataset(self, dataset_tag: str) -> pd.DataFrame:
        """
        Evaluate reward model consistency on a specific dataset.
        
        Args:
            dataset_tag: Tag identifying the dataset (e.g., 'helpful', 'harm')
            
        Returns:
            DataFrame containing evaluation results
        """
        logger.info(f"Evaluating dataset: {dataset_tag}")
        
        # Load dataset
        dataset_path = Path(self.config.dataset_dir) / f"{dataset_tag}_train_formed.jsonl"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        dataset = load_dataset("json", data_files=str(dataset_path))['train']
        logger.info(f"Dataset size: {len(dataset)}")
        
        # Evaluate
        results = []
        consistent_count = 0
        
        with torch.no_grad():
            for example in tqdm(dataset, desc=f"Evaluating {dataset_tag}"):
                prompt = example['prompt']
                chosen = example['chosen']
                rejected = example['rejected']
                
                # Compute rewards with both models
                reward1_chosen = self.compute_reward(prompt, chosen, self.rm1, self.rm1_tokenizer)
                reward1_reject = self.compute_reward(prompt, rejected, self.rm1, self.rm1_tokenizer)
                reward2_chosen = self.compute_reward(prompt, chosen, self.rm2, self.rm2_tokenizer)
                reward2_reject = self.compute_reward(prompt, rejected, self.rm2, self.rm2_tokenizer)
                
                # Calculate gaps (margin between chosen and rejected)
                gap1 = reward1_chosen - reward1_reject
                gap2 = reward2_chosen - reward2_reject
                
                # Record results
                record = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reward1_chosen": float(reward1_chosen),
                    "reward1_reject": float(reward1_reject),
                    "reward2_chosen": float(reward2_chosen),
                    "reward2_reject": float(reward2_reject),
                    "gap1": float(gap1),
                    "gap2": float(gap2),
                    "consistent": gap1 * gap2 > 0
                }
                results.append(record)
                
                # Track consistency (both models agree on preference direction)
                if gap1 * gap2 > 0:
                    consistent_count += 1
        
        # Calculate and log consistency rate
        consistency_rate = consistent_count / len(dataset)
        logger.info(f"Consistency Rate for {dataset_tag}: {consistency_rate:.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        output_path = Path(self.config.output_dir) / f"{dataset_tag}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Results saved to: {output_path}")
        
        return df
    
    def evaluate_multiple_datasets(self, dataset_tags: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Evaluate multiple datasets.
        
        Args:
            dataset_tags: List of dataset tags to evaluate
            
        Returns:
            Dictionary mapping dataset tags to result DataFrames
        """
        results = {}
        for tag in dataset_tags:
            try:
                results[tag] = self.evaluate_dataset(tag)
            except Exception as e:
                logger.error(f"Error evaluating dataset {tag}: {e}")
        
        return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate consistency between two reward models"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path or identifier for the first reward model"
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path or identifier for the second reward model"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["helpful", "harm"],
        help="Dataset tags to evaluate (default: helpful harm)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./RLHF/dataset",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/reward_evaluation",
        help="Directory for output files"
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=60,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = RewardModelConfig(
        model1_path=args.model1,
        model2_path=args.model2,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        seed=args.seed
    )
    
    # Run evaluation
    evaluator = RewardModelEvaluator(config)
    evaluator.evaluate_multiple_datasets(args.datasets)


if __name__ == "__main__":
    main()