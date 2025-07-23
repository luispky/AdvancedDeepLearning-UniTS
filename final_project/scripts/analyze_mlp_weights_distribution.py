#!/usr/bin/env python3
"""
Analyze MLP weight distributions for initial, baseline, and coherence models.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def extract_experiment_info_from_path(models_path: Path) -> Dict[str, Any]:
    """Extract experiment information from the models directory path."""
    # Expected path format: saved_models/mlp/{dataset}/hs{hidden_sizes}_ep{epochs}
    path_parts = models_path.parts

    # Find the dataset and config parts
    if len(path_parts) < 4:
        raise ValueError(f"❌ Invalid models path structure: {models_path}")

    # Extract dataset name
    dataset_name = path_parts[-2]

    # Extract config from directory name (e.g., "hs128_ep5")
    config_dir = path_parts[-1]
    config_match = re.match(r"hs(.+)_ep(\d+)", config_dir)

    if not config_match:
        raise ValueError(f"❌ Invalid config directory format: {config_dir}")

    hidden_sizes_str = config_match.group(1)
    num_epochs = int(config_match.group(2))

    # Parse hidden sizes (e.g., "128" or "128-64")
    hidden_sizes = [int(size) for size in hidden_sizes_str.split("-")]

    return {
        "dataset": dataset_name,
        "hidden_sizes": hidden_sizes,
        "num_epochs": num_epochs,
        "config_dir": config_dir,
    }


def find_model_files(models_path: Path) -> Dict[str, Path]:
    """Find all model files in the given directory."""
    model_files = {}

    for model_file in models_path.glob("*.pth"):
        if "initial" in model_file.name:
            model_files["initial"] = model_file
        elif "baseline" in model_file.name:
            model_files["baseline"] = model_file
        elif "coherence" in model_file.name:
            model_files["coherence"] = model_file

    return model_files


def load_experiment_config(models_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    config_path = models_path / "experiment_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Experiment config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"📋 Loaded experiment config from: {config_path}")
    return config


def load_model_from_file(
    model_path: Path,
    model_config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load a model from file path using saved configuration."""
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # Create model architecture using saved config
    from src.mlp import MLPClassifier

    model = MLPClassifier(**model_config).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded {model_path.name}")

    return model


def extract_all_weights(model: nn.Module) -> np.ndarray:
    """Extract all weights from the model as a flattened array."""
    weights = [
        param.data.cpu().numpy().flatten()
        for param in model.parameters()
        if param.dim() > 1  # Only weight matrices, not biases
    ]

    return np.concatenate(weights)


def create_weight_distribution_plot(
    initial_weights: np.ndarray,
    baseline_weights: np.ndarray,
    coherence_weights: np.ndarray,
    dataset_name: str,
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float = 0.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    save: bool = True,
    plots_base_dir: Optional[Path] = None,
) -> None:
    """Create 1x3 subplot showing weight distributions for all three models."""
    print("📊 Creating weight distribution plot...")

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot data
    weights_data = [
        ("Initial", initial_weights),
        ("Baseline", baseline_weights),
        ("Coherence", coherence_weights),
    ]

    # Shared y-axis label
    shared_ylabel = "Density"

    for idx, (model_type, weights) in enumerate(weights_data):
        ax = axes[idx]

        # Create histogram
        ax.hist(weights, bins=50, density=True, alpha=0.7, color=f"C{idx}")

        # Set labels
        ax.set_xlabel("Weight Values")
        if idx == 0:  # Only first subplot gets y-label
            ax.set_ylabel(shared_ylabel, fontweight="bold")

        # Set title
        if model_type == "Coherence" and (alpha > 0 or beta > 0):
            title = f"{model_type}\n(α={alpha}, β={beta})"
        else:
            title = model_type
        ax.set_title(title, fontweight="bold")

        # Add statistics
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        ax.text(
            0.04,
            0.95,
            f"μ={mean_val:.3f}\nσ={std_val:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=14,
        )

        ax.grid(True, alpha=0.3)

    # Create suptitle
    hidden_str = "-".join(map(str, hidden_sizes))
    suptitle = f"Weight Distributions - {dataset_name.upper()} Dataset"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.95)

    plt.tight_layout()

    if save:
        # Determine plots directory
        if plots_base_dir is None:
            plots_base_dir = Path(__file__).parent.parent / "plots/mlp"

        # Create plots directory structure
        plots_config_dir = (
            plots_base_dir / dataset_name / f"hs{hidden_str}_ep{num_epochs}"
        )
        plots_config_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        if coherence_weight > 0:
            plot_filename = f"weights_distribution_cw{coherence_weight:.1f}_a{alpha:.1f}_b{beta:.1f}.png"
        else:
            plot_filename = "weights_distribution_baseline.png"

        plot_path = plots_config_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Plot saved to: {plot_path}")
    else:
        plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze MLP weight distributions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "models_path",
        type=str,
        help="Path to the experiment models directory",
    )
    parser.add_argument("--save", action="store_true")

    return parser.parse_args()


def run_analysis(models_path: Path, device: torch.device, save: bool) -> None:
    """Run the complete weight distribution analysis."""
    # Extract experiment information
    exp_info = extract_experiment_info_from_path(models_path)
    dataset_name = exp_info["dataset"]
    hidden_sizes = exp_info["hidden_sizes"]
    num_epochs = exp_info["num_epochs"]

    print("📋 Experiment info:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Architecture: {hidden_sizes}")
    print(f"   Epochs: {num_epochs}")

    # Find model files
    model_files = find_model_files(models_path)
    print(f"\n📥 Found {len(model_files)} model files:")
    for model_type, model_path in model_files.items():
        print(f"   {model_type}: {model_path.name}")

    # Check if all required models are found
    required_models = ["initial", "baseline", "coherence"]
    if missing_models := [
        model for model in required_models if model not in model_files
    ]:
        raise FileNotFoundError(f"❌ Missing models: {missing_models}")

    # Load experiment configuration
    experiment_config = load_experiment_config(models_path)
    model_config = experiment_config["model_config"]
    loss_config = experiment_config["loss_config"]

    # Load all models
    print("\n📥 Loading models...")
    models = {
        model_type: load_model_from_file(model_files[model_type], model_config, device)
        for model_type in required_models
    }

    # Extract weights
    print("\n⚙️  Extracting weights...")
    weights = {
        model_type: extract_all_weights(models[model_type])
        for model_type in required_models
    }

    # Print statistics
    print("📊 Weight statistics:")
    for model_type in required_models:
        weight_data = weights[model_type]
        print(
            f"   {model_type.capitalize()}: {len(weight_data)} weights, μ={np.mean(weight_data):.4f}, σ={np.std(weight_data):.4f}"
        )

    # Extract coherence parameters from config
    coherence_weight = loss_config.get("coherence_weight", 0.0)
    alpha = loss_config.get("alpha", 0.0)
    beta = loss_config.get("beta", 0.0)

    # Create and save plot
    create_weight_distribution_plot(
        weights["initial"],
        weights["baseline"],
        weights["coherence"],
        dataset_name,
        hidden_sizes,
        num_epochs,
        coherence_weight,
        alpha,
        beta,
        save=save,
    )

    print("\n✅ Analysis completed successfully!")


def main():
    """Main entry point."""
    args = parse_arguments()

    print("🔍 MLP Weight Distribution Analysis")
    print("=" * 50)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    try:
        # Parse models path
        models_path = Path(args.models_path)
        if not models_path.exists():
            raise FileNotFoundError(f"❌ Models directory not found: {models_path}")

        print(f"📁 Analyzing models in: {models_path}")

        # Run analysis
        run_analysis(models_path, device, args.save)

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
