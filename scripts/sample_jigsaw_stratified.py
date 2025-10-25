"""
Stratified Random Sampling for Jigsaw Dataset
Sample 10,000 instances from test set while preserving label distribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import chi2_contingency
import json
from datetime import datetime

# Configuration
RANDOM_SEED = 42
SAMPLE_SIZE = 10000
DATA_DIR = Path("data/jigsaw")
OUTPUT_DIR = Path("results/llm_annotations_7b/jigsaw")
LABEL_COL = "label"

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

def load_full_test_set():
    """Load complete Jigsaw test set"""
    test_path = DATA_DIR / "test.csv"
    df = pd.read_csv(test_path)
    print(f"[OK] Loaded full test set: {len(df):,} samples")
    return df

def check_label_distribution(df):
    """Check and report label distribution"""
    print("\n=== Label Distribution (Full Test Set) ===")
    dist = df[LABEL_COL].value_counts().sort_index()
    prop = df[LABEL_COL].value_counts(normalize=True).sort_index()
    
    for label, count in dist.items():
        label_name = "non-toxic" if label == 0 else "toxic"
        print(f"  {label} ({label_name}): {count:,} ({prop[label]:.4%})")
    
    return dist, prop

def stratified_sampling(df):
    """
    Perform stratified random sampling using StratifiedShuffleSplit
    
    Returns:
        sampled_df: DataFrame with sampled instances
        sample_indices: Original indices of sampled instances
    """
    print(f"\n=== Stratified Sampling (n={SAMPLE_SIZE:,}) ===")
    
    # Use StratifiedShuffleSplit to get exact sample size
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=SAMPLE_SIZE,
        random_state=RANDOM_SEED
    )
    
    # Get indices
    for train_idx, sample_idx in splitter.split(df, df[LABEL_COL]):
        sample_indices = sample_idx
    
    # Extract sampled data
    sampled_df = df.iloc[sample_indices].copy()
    sampled_df['original_index'] = sample_indices
    
    print(f"[OK] Sampled {len(sampled_df):,} instances")
    
    return sampled_df, sample_indices

def validate_distribution(full_df, sampled_df):
    """
    Validate sampling quality using statistical tests
    
    Returns:
        validation_report: Dict with validation metrics
    """
    print("\n=== Distribution Validation ===")
    
    # Get distributions
    full_dist = full_df[LABEL_COL].value_counts().sort_index()
    sample_dist = sampled_df[LABEL_COL].value_counts().sort_index()
    
    full_prop = full_df[LABEL_COL].value_counts(normalize=True).sort_index()
    sample_prop = sampled_df[LABEL_COL].value_counts(normalize=True).sort_index()
    
    # Calculate absolute proportion differences
    prop_diffs = {}
    print("\nProportion Comparison:")
    for label in full_prop.index:
        diff = abs(full_prop[label] - sample_prop[label])
        prop_diffs[label] = diff
        label_name = "non-toxic" if label == 0 else "toxic"
        print(f"  {label} ({label_name}):")
        print(f"    Full:   {full_prop[label]:.4%} ({full_dist[label]:,})")
        print(f"    Sample: {sample_prop[label]:.4%} ({sample_dist[label]:,})")
        print(f"    |Δ|:    {diff:.4%}")
    
    # Chi-square test
    observed = sample_dist.values
    # Expected proportions from full dataset
    expected = full_prop.values * len(sampled_df)
    
    chi2_stat, p_value = chi2_contingency([observed, expected])[:2]
    
    print(f"\n[OK] Chi-square test:")
    print(f"  Chi-square statistic: {chi2_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Validation criteria
    max_prop_diff = max(prop_diffs.values())
    prop_diff_pass = max_prop_diff <= 0.01  # ≤1% difference
    
    # Note: For large samples (97k vs 10k), chi-square is very sensitive
    # We focus on practical significance (proportion differences)
    print(f"\n[OK] Validation Results:")
    print(f"  Max proportion difference: {max_prop_diff:.4%} (threshold: ≤1.0%)")
    print(f"  Proportion test: {'PASS' if prop_diff_pass else 'FAIL'}")
    
    if p_value < 0.05:
        print(f"  Chi-square p-value: {p_value:.4f} (< 0.05)")
        print(f"  Note: Low p-value expected for large sample sizes")
        print(f"        Practical significance (proportion diff) is more important")
    
    validation_report = {
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "max_proportion_difference": float(max_prop_diff),
        "proportion_test_passed": bool(int(prop_diff_pass)),
        "full_distribution": {int(k): int(v) for k, v in full_dist.items()},
        "sample_distribution": {int(k): int(v) for k, v in sample_dist.items()},
        "full_proportions": {int(k): float(v) for k, v in full_prop.items()},
        "sample_proportions": {int(k): float(v) for k, v in sample_prop.items()},
        "proportion_differences": {int(k): float(v) for k, v in prop_diffs.items()}
    }
    
    return validation_report

def visualize_comparison(full_df, sampled_df, output_dir):
    """Create visualization comparing distributions"""
    print("\n=== Creating Visualization ===")
    
    full_dist = full_df[LABEL_COL].value_counts().sort_index()
    sample_dist = sampled_df[LABEL_COL].value_counts().sort_index()
    
    full_prop = full_df[LABEL_COL].value_counts(normalize=True).sort_index()
    sample_prop = sampled_df[LABEL_COL].value_counts(normalize=True).sort_index()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute counts
    x = np.arange(len(full_dist))
    width = 0.35
    
    axes[0].bar(x - width/2, full_dist.values, width, label=f'Full (n={len(full_df):,})', alpha=0.8)
    axes[0].bar(x + width/2, sample_dist.values, width, label=f'Sample (n={len(sampled_df):,})', alpha=0.8)
    axes[0].set_xlabel('Label (0=non-toxic, 1=toxic)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Absolute Distribution Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['0 (non-toxic)', '1 (toxic)'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Proportions
    axes[1].bar(x - width/2, full_prop.values * 100, width, label='Full', alpha=0.8)
    axes[1].bar(x + width/2, sample_prop.values * 100, width, label='Sample', alpha=0.8)
    axes[1].set_xlabel('Label (0=non-toxic, 1=toxic)')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Proportion Distribution Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['0 (non-toxic)', '1 (toxic)'])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (full_val, sample_val) in enumerate(zip(full_prop.values * 100, sample_prop.values * 100)):
        axes[1].text(i - width/2, full_val + 0.5, f'{full_val:.2f}%', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, sample_val + 0.5, f'{sample_val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "jigsaw_sampling_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization to {output_path}")
    plt.close()

def save_sampled_data(sampled_df, output_dir):
    """Save sampled data and index list"""
    print("\n=== Saving Sampled Data ===")
    
    # Save full sampled data (for LLM annotation)
    sample_csv_path = output_dir / "jigsaw_sample10k.csv"
    sampled_df[['text', 'label']].to_csv(sample_csv_path, index=False)
    print(f"[OK] Saved sampled data to {sample_csv_path}")
    
    # Save index list for reproducibility
    index_csv_path = output_dir / "jigsaw_sample10k_ids.csv"
    pd.DataFrame({
        'original_index': sampled_df['original_index'].values,
        'label': sampled_df['label'].values
    }).to_csv(index_csv_path, index=False)
    print(f"[OK] Saved sample indices to {index_csv_path}")
    
    return sample_csv_path, index_csv_path

def generate_sampling_report(full_df, sampled_df, validation_report, output_dir):
    """Generate comprehensive sampling report"""
    print("\n=== Generating Sampling Report ===")
    
    report = {
        "sampling_metadata": {
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "sample_size": SAMPLE_SIZE,
            "full_dataset_size": len(full_df),
            "sampling_method": "StratifiedShuffleSplit",
            "stratify_column": LABEL_COL
        },
        "validation": validation_report,
        "quality_assessment": {
            "distribution_preserved": validation_report["proportion_test_passed"],
            "max_proportion_deviation": validation_report["max_proportion_difference"],
            "recommendation": "APPROVED for LLM annotation" if validation_report["proportion_test_passed"] else "REVIEW REQUIRED"
        }
    }
    
    # Save report
    report_path = output_dir / "jigsaw_sampling_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved sampling report to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLING SUMMARY")
    print("="*60)
    print(f"Full dataset size: {len(full_df):,}")
    print(f"Sample size: {len(sampled_df):,}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Max proportion difference: {validation_report['max_proportion_difference']:.4%}")
    print(f"Quality: {report['quality_assessment']['recommendation']}")
    print("="*60)
    
    return report

def main():
    """Main execution flow"""
    print("="*60)
    print("Jigsaw Stratified Random Sampling")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load full test set
    full_df = load_full_test_set()
    
    # Step 2: Check label distribution
    check_label_distribution(full_df)
    
    # Step 3: Perform stratified sampling
    sampled_df, sample_indices = stratified_sampling(full_df)
    
    # Step 4: Check sampled distribution
    print("\n=== Label Distribution (Sampled Data) ===")
    sampled_dist = sampled_df[LABEL_COL].value_counts().sort_index()
    sampled_prop = sampled_df[LABEL_COL].value_counts(normalize=True).sort_index()
    for label, count in sampled_dist.items():
        label_name = "non-toxic" if label == 0 else "toxic"
        print(f"  {label} ({label_name}): {count:,} ({sampled_prop[label]:.4%})")
    
    # Step 5: Validate distribution
    validation_report = validate_distribution(full_df, sampled_df)
    
    # Step 6: Visualize comparison
    visualize_comparison(full_df, sampled_df, OUTPUT_DIR)
    
    # Step 7: Save sampled data
    save_sampled_data(sampled_df, OUTPUT_DIR)
    
    # Step 8: Generate report
    generate_sampling_report(full_df, sampled_df, validation_report, OUTPUT_DIR)
    
    print("\n[OK] Sampling process completed successfully!")
    print(f"\nNext step: Run LLM annotation on {OUTPUT_DIR / 'jigsaw_sample10k.csv'}")

if __name__ == "__main__":
    main()

