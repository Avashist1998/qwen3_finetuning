#!/usr/bin/env python3
"""
Compare multiple trained models by loading their evaluation metrics.
This script helps identify the best performing model and detect overfitting.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_metrics_from_checkpoint(checkpoint_path: str) -> Dict:
    """Load training and evaluation metrics from a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path)
    
    metrics = {
        "checkpoint": checkpoint_path,
        "train_loss": None,
        "eval_loss": None,
        "eval_mse": None,
        "eval_rmse": None,
        "eval_mae": None,
        "eval_r2_score": None,
        "train_samples": None,
        "eval_samples": None,
        "overfitting_ratio": None,
    }
    
    # Try to load trainer_state.json (contains training history)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            
            # Get the last logged metrics
            if "log_history" in trainer_state:
                log_history = trainer_state["log_history"]
                
                # Find the last training loss
                for entry in reversed(log_history):
                    if "loss" in entry and metrics["train_loss"] is None:
                        metrics["train_loss"] = entry["loss"]
                
                # Find the last evaluation metrics
                for entry in reversed(log_history):
                    if "eval_loss" in entry:
                        metrics["eval_loss"] = entry.get("eval_loss")
                        metrics["eval_mse"] = entry.get("eval_mse")
                        metrics["eval_rmse"] = entry.get("eval_rmse")
                        metrics["eval_mae"] = entry.get("eval_mae")
                        metrics["eval_r2_score"] = entry.get("eval_r2_score")
                        break
    
    # Try to load eval_results.json
    eval_results_path = checkpoint_dir / "eval_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
            metrics["eval_loss"] = eval_results.get("eval_loss", metrics["eval_loss"])
            metrics["eval_mse"] = eval_results.get("eval_mse", metrics["eval_mse"])
            metrics["eval_rmse"] = eval_results.get("eval_rmse", metrics["eval_rmse"])
            metrics["eval_mae"] = eval_results.get("eval_mae", metrics["eval_mae"])
            metrics["eval_r2_score"] = eval_results.get("eval_r2_score", metrics["eval_r2_score"])
    
    # Try to load train_results.json
    train_results_path = checkpoint_dir / "train_results.json"
    if train_results_path.exists():
        with open(train_results_path, 'r') as f:
            train_results = json.load(f)
            metrics["train_loss"] = train_results.get("train_loss", metrics["train_loss"])
            metrics["train_samples"] = train_results.get("train_samples", metrics["train_samples"])
            metrics["eval_samples"] = train_results.get("eval_samples", metrics["eval_samples"])
    
    # Calculate overfitting ratio
    if metrics["train_loss"] is not None and metrics["eval_loss"] is not None:
        metrics["overfitting_ratio"] = metrics["eval_loss"] / metrics["train_loss"]
    
    return metrics


def find_all_checkpoints(base_dir: str = "peft_lab_outputs") -> List[str]:
    """Find all checkpoint directories in the base directory."""
    base_path = Path(base_dir)
    checkpoints = []
    
    if not base_path.exists():
        print(f"Warning: Directory {base_dir} does not exist")
        return checkpoints
    
    # Look for directories that contain either trainer_state.json or adapter_config.json
    for item in base_path.iterdir():
        if item.is_dir():
            if (item / "trainer_state.json").exists() or (item / "adapter_config.json").exists():
                checkpoints.append(str(item))
            
            # Also check subdirectories (like checkpoint-XXX)
            for subitem in item.iterdir():
                if subitem.is_dir():
                    if (subitem / "trainer_state.json").exists():
                        checkpoints.append(str(subitem))
    
    return sorted(checkpoints)


def print_comparison_table(metrics_list: List[Dict]):
    """Print a formatted comparison table of all models."""
    
    if not metrics_list:
        print("No metrics found to compare.")
        return
    
    # Convert to DataFrame for easy formatting
    df = pd.DataFrame(metrics_list)
    
    # Select columns to display
    display_columns = [
        "checkpoint", "train_loss", "eval_loss", "eval_mse", 
        "eval_rmse", "eval_mae", "eval_r2_score", "overfitting_ratio"
    ]
    
    # Filter to only include columns that exist
    display_columns = [col for col in display_columns if col in df.columns]
    df_display = df[display_columns]
    
    # Format floating point numbers
    float_columns = df_display.select_dtypes(include=['float64', 'float32']).columns
    for col in float_columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    
    # Shorten checkpoint paths for readability
    df_display["checkpoint"] = df_display["checkpoint"].apply(
        lambda x: x.replace("peft_lab_outputs/", "") if isinstance(x, str) else x
    )
    
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print(df_display.to_string(index=False))
    print("="*120)
    
    # Find and highlight the best model
    if "eval_mse" in df.columns:
        best_mse_idx = pd.to_numeric(df["eval_mse"], errors='coerce').idxmin()
        if pd.notna(best_mse_idx):
            print(f"\n🏆 Best MSE: {df.loc[best_mse_idx, 'checkpoint']}")
            print(f"   MSE: {df.loc[best_mse_idx, 'eval_mse']}")
    
    if "eval_r2_score" in df.columns:
        best_r2_idx = pd.to_numeric(df["eval_r2_score"], errors='coerce').idxmax()
        if pd.notna(best_r2_idx):
            print(f"\n🏆 Best R² Score: {df.loc[best_r2_idx, 'checkpoint']}")
            print(f"   R²: {df.loc[best_r2_idx, 'eval_r2_score']}")
    
    if "eval_loss" in df.columns:
        best_loss_idx = pd.to_numeric(df["eval_loss"], errors='coerce').idxmin()
        if pd.notna(best_loss_idx):
            print(f"\n🏆 Best Eval Loss: {df.loc[best_loss_idx, 'checkpoint']}")
            print(f"   Loss: {df.loc[best_loss_idx, 'eval_loss']}")
    
    if "overfitting_ratio" in df.columns:
        # Find models with good generalization (ratio close to 1.0)
        df["overfitting_ratio_numeric"] = pd.to_numeric(df["overfitting_ratio"], errors='coerce')
        best_gen = df.nsmallest(3, 'overfitting_ratio_numeric')
        print("\n📊 Best Generalization (lowest overfitting ratio):")
        for idx, row in best_gen.iterrows():
            ratio = row['overfitting_ratio_numeric']
            if pd.notna(ratio):
                status = "✅" if ratio < 1.1 else "⚠️" if ratio < 1.2 else "❌"
                print(f"   {status} {row['checkpoint']}: {ratio:.3f}")


def export_to_csv(metrics_list: List[Dict], output_file: str = "model_comparison.csv"):
    """Export comparison metrics to a CSV file."""
    df = pd.DataFrame(metrics_list)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Metrics exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple trained model checkpoints"
    )
    parser.add_argument(
        "--checkpoints", 
        type=str, 
        nargs="+", 
        default=None,
        help="Specific checkpoint directories to compare (space-separated)"
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="peft_lab_outputs",
        help="Base directory containing checkpoints (default: peft_lab_outputs)"
    )
    parser.add_argument(
        "--export_csv", 
        type=str, 
        default=None,
        help="Export comparison to CSV file"
    )
    
    args = parser.parse_args()
    
    # Find checkpoints to compare
    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        print(f"Scanning {args.base_dir} for checkpoints...")
        checkpoints = find_all_checkpoints(args.base_dir)
        print(f"Found {len(checkpoints)} checkpoint(s)")
    
    if not checkpoints:
        print("No checkpoints found to compare.")
        return
    
    # Load metrics from all checkpoints
    metrics_list = []
    for checkpoint in checkpoints:
        print(f"Loading metrics from {checkpoint}...")
        metrics = load_metrics_from_checkpoint(checkpoint)
        if metrics["eval_loss"] is not None:  # Only include if we have evaluation metrics
            metrics_list.append(metrics)
        else:
            print(f"  ⚠️  No evaluation metrics found for {checkpoint}")
    
    if not metrics_list:
        print("\nNo evaluation metrics found in any checkpoint.")
        print("Make sure your training runs have completed with evaluation enabled.")
        return
    
    # Print comparison table
    print_comparison_table(metrics_list)
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(metrics_list, args.export_csv)


if __name__ == "__main__":
    main()

