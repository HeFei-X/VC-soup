
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RewardNormalizer:
    """Normalizes reward scores and computes consistency metrics."""
    
    def __init__(
        self,
        input_file: str,
        output_dir: str = "outputs/reward_analysis",
        is_cost_model: bool = False
    ):
        """
        Initialize the normalizer.
        
        Args:
            input_file: Path to CSV file with raw reward scores
            output_dir: Directory for output files and plots
            is_cost_model: If True, inverts reward2 scores (for cost models)
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.is_cost_model = is_cost_model
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract dataset tag from filename
        self.tag = self.input_file.stem
        
        logger.info(f"Initializing normalizer for: {self.tag}")
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Load data and perform initial preprocessing.
        
        Returns:
            Preprocessed DataFrame
        """
        df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(df)} records from {self.input_file}")
        
        # Handle cost model (invert reward2 scores)
        if self.is_cost_model:
            logger.info("Inverting reward2 scores (cost model mode)")
            df["reward2_chosen"] = -df["reward2_chosen"]
            df["reward2_reject"] = -df["reward2_reject"]
        
        # Standardize column names
        if "reject" in df.columns and "rejected" not in df.columns:
            df = df.rename(columns={"reject": "rejected"})
        
        # Remove invalid entries
        initial_count = len(df)
        df = df.dropna(subset=["chosen", "rejected"])
        df = df[df["chosen"].astype(str).str.strip() != ""]
        df = df[df["rejected"].astype(str).str.strip() != ""]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid records")
        
        logger.info(f"Valid records after preprocessing: {len(df)}")
        return df
    
    def normalize_rewards(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize reward scores using Z-score normalization.
        
        Args:
            df: DataFrame with raw reward scores
            
        Returns:
            DataFrame with normalized scores
        """
        df = df.copy()
        
        # Normalize reward1 (Z-score normalization)
        r1_all = pd.concat([df["reward1_chosen"], df["reward1_reject"]])
        r1_mean = r1_all.mean()
        r1_std = r1_all.std()
        
        df["reward1_chosen"] = (df["reward1_chosen"] - r1_mean) / (r1_std + 1e-12)
        df["reward1_reject"] = (df["reward1_reject"] - r1_mean) / (r1_std + 1e-12)
        
        logger.info(f"Reward1 - Mean: {r1_mean:.4f}, Std: {r1_std:.4f}")
        
        # Normalize reward2
        r2_all = pd.concat([df["reward2_chosen"], df["reward2_reject"]])
        r2_mean = r2_all.mean()
        r2_std = r2_all.std()
        
        df["reward2_chosen"] = (df["reward2_chosen"] - r2_mean) / (r2_std + 1e-12)
        df["reward2_reject"] = (df["reward2_reject"] - r2_mean) / (r2_std + 1e-12)
        
        logger.info(f"Reward2 - Mean: {r2_mean:.4f}, Std: {r2_std:.4f}")
        
        return df
    
    def compute_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute reward gaps and consistency metric.
        
        The consistency metric is computed as:
        Consistency = (gap1 + gap2) / (sqrt(gap1² + gap2²) * sqrt(2))
        
        This measures the alignment between two reward models, ranging from -1 to 1.
        
        Args:
            df: DataFrame with normalized rewards
            
        Returns:
            DataFrame with gap and consistency columns
        """
        df = df.copy()
        
        # Compute reward gaps
        df["gap1"] = df["reward1_chosen"] - df["reward1_reject"]
        df["gap2"] = df["reward2_chosen"] - df["reward2_reject"]
        
        # Compute consistency metric
        numerator = df["gap1"] + df["gap2"]
        denominator = np.sqrt(df["gap1"]**2 + df["gap2"]**2) * np.sqrt(2) + 1e-12
        df["Consistency"] = numerator / denominator
        
        logger.info(f"Consistency - Mean: {df['Consistency'].mean():.4f}, "
                   f"Std: {df['Consistency'].std():.4f}")
        
        return df
    
    def generate_visualizations(self, df: pd.DataFrame) -> None:
        """
        Generate and save visualization plots.
        
        Args:
            df: DataFrame with consistency metrics
        """
        # 1. Consistency distribution histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df["Consistency"], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        plt.title("Consistency Distribution")
        plt.xlabel("Consistency")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{self.tag}_consistency_dist.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved consistency distribution: {output_path}")
        
        # 2. Gap scatter plot with color-coded signs
        plt.figure(figsize=(6, 4))
        
        # Separate data by sign combinations
        mask_pos = (df["gap1"] > 0) & (df["gap2"] > 0)
        mask_neg = (df["gap1"] < 0) & (df["gap2"] < 0)
        mask_other = ~(mask_pos | mask_neg)
        
        # Plot each group
        plt.scatter(
            df.loc[mask_pos, "gap1"],
            df.loc[mask_pos, "gap2"],
            s=12, alpha=0.7, color="#2ecc71", label="Both positive"
        )
        
        plt.scatter(
            df.loc[mask_neg, "gap1"],
            df.loc[mask_neg, "gap2"],
            s=12, alpha=0.7, color="#e74c3c", label="Both negative"
        )
        
        plt.scatter(
            df.loc[mask_other, "gap1"],
            df.loc[mask_other, "gap2"],
            s=10, alpha=0.5, color="#7f8c8d", label="Mixed sign"
        )
        
        plt.xlabel("Gap1 (Model 1)")
        plt.ylabel("Gap2 (Model 2)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        
        output_path = self.output_dir / f"{self.tag}_gap_scatter.png"
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved gap scatter plot: {output_path}")
    
    def compute_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute and log consistency statistics.
        
        Args:
            df: DataFrame with consistency metrics
            
        Returns:
            Dictionary of statistics
        """
        total = len(df)
        high_consistency = (df["Consistency"] > 0.7).sum()
        low_consistency = (df["Consistency"] < -0.7).sum()
        
        stats = {
            "total_samples": total,
            "high_consistency_count": high_consistency,
            "high_consistency_ratio": high_consistency / total,
            "low_consistency_count": low_consistency,
            "low_consistency_ratio": low_consistency / total,
            "mean_consistency": df["Consistency"].mean(),
            "std_consistency": df["Consistency"].std(),
            "median_consistency": df["Consistency"].median()
        }
        
        logger.info("=" * 60)
        logger.info("CONSISTENCY STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total}")
        logger.info(f"High consistency (>0.7): {high_consistency} "
                   f"({stats['high_consistency_ratio']*100:.2f}%)")
        logger.info(f"Low consistency (<-0.7): {low_consistency} "
                   f"({stats['low_consistency_ratio']*100:.2f}%)")
        logger.info(f"Mean consistency: {stats['mean_consistency']:.4f}")
        logger.info(f"Median consistency: {stats['median_consistency']:.4f}")
        logger.info(f"Std consistency: {stats['std_consistency']:.4f}")
        logger.info("=" * 60)
        
        return stats
    
    def save_results(self, df: pd.DataFrame) -> Path:
        """
        Save normalized results to CSV.
        
        Args:
            df: DataFrame with all computed metrics
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{self.tag}_normed.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved normalized data: {output_path}")
        return output_path
    
    def process(self) -> Tuple[pd.DataFrame, dict]:
        """
        Run the complete normalization and analysis pipeline.
        
        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
        """
        # Load and preprocess
        df = self.load_and_preprocess()
        
        # Normalize rewards
        df = self.normalize_rewards(df)
        
        # Compute consistency
        df = self.compute_consistency(df)
        
        # Generate visualizations
        self.generate_visualizations(df)
        
        # Compute statistics
        stats = self.compute_statistics(df)
        
        # Save results
        self.save_results(df)
        
        return df, stats


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Normalize reward scores and analyze consistency"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with raw reward scores"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reward_analysis",
        help="Directory for output files and plots"
    )
    parser.add_argument(
        "--cost-model",
        action="store_true",
        help="Enable if reward2 is from a cost model (will invert scores)"
    )
    
    args = parser.parse_args()
    
    # Run normalization and analysis
    normalizer = RewardNormalizer(
        input_file=args.input,
        output_dir=args.output_dir,
        is_cost_model=args.cost_model
    )
    
    df, stats = normalizer.process()
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()