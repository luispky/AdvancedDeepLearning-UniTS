from pathlib import Path
from typing import List, Dict, Any
import argparse
import yaml
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.low_coherence_mlp import (
    setup_device,
    setup_dataloaders,
    create_model_and_summary,
    setup_loss_and_optimizer,
    train_model,
    evaluate_model,
    get_weights_coherence_stats,
)
from utils import seed_everything


# Global parameters
PARENT_DIR = Path(__file__).parent.parent
PLOTS_BASE_DIR = PARENT_DIR / "plots/mlp"
MODELS_BASE_DIR = PARENT_DIR / "saved_models/mlp"
RESULTS_BASE_DIR = PARENT_DIR / "results/mlp"
CONFIG_DIR = PARENT_DIR / "configs"


def get_dataset_dirs(dataset_name: str) -> Dict[str, Path]:
    """Get dataset-specific directories for plots, results, and models."""
    return {
        "plots": PLOTS_BASE_DIR / dataset_name,
        "results": RESULTS_BASE_DIR / dataset_name,
        "models_base": MODELS_BASE_DIR / dataset_name,
    }


def get_config_subdir(hidden_sizes: List[int], num_epochs: int) -> str:
    """Get configuration subdirectory name for architecture and epochs."""
    hidden_str = format_hidden_sizes(hidden_sizes)
    return f"hs{hidden_str}_ep{num_epochs}"


def get_model_config_dir(
    dataset_name: str, hidden_sizes: List[int], num_epochs: int
) -> Path:
    """Get model configuration-specific directory for saved models."""
    config_subdir = get_config_subdir(hidden_sizes, num_epochs)
    return MODELS_BASE_DIR / dataset_name / config_subdir


def get_plots_config_dir(
    dataset_name: str, hidden_sizes: List[int], num_epochs: int
) -> Path:
    """Get plots configuration-specific directory."""
    config_subdir = get_config_subdir(hidden_sizes, num_epochs)
    return PLOTS_BASE_DIR / dataset_name / config_subdir


def get_results_config_dir(
    dataset_name: str, hidden_sizes: List[int], num_epochs: int
) -> Path:
    """Get results configuration-specific directory."""
    config_subdir = get_config_subdir(hidden_sizes, num_epochs)
    return RESULTS_BASE_DIR / dataset_name / config_subdir


def model_exists(
    model_name: str, dataset_name: str, hidden_sizes: List[int], num_epochs: int
) -> bool:
    """Check if a model file already exists in the dataset/config-specific directory."""
    model_config_dir = get_model_config_dir(dataset_name, hidden_sizes, num_epochs)
    model_path = model_config_dir / model_name
    return model_path.exists()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        print(f"📋 Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"❌ Invalid YAML configuration: {e}")


def format_hidden_sizes(hidden_sizes: List[int]) -> str:
    """Format hidden sizes list for file naming."""
    return "-".join(map(str, hidden_sizes))


def build_param_string(coherence_weight: float, alpha: float, beta: float) -> str:
    """Build parameter string for coherence experiments."""
    params = [f"cw{coherence_weight:.1f}"]
    if alpha > 0:
        params.append(f"a{alpha:.1f}")
    if beta > 0:
        params.append(f"b{beta:.1f}")
    return "_".join(params)


def create_experiment_name(
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float,
    experiment_type: str,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> str:
    """
    Create experiment name without dataset for consistent file naming.

    Args:
        hidden_sizes: List of hidden layer sizes
        num_epochs: Number of training epochs
        coherence_weight: Coherence loss weight (0.0 for baseline)
        experiment_type: 'baseline' or 'coherence'
        alpha: Equiangularity regularization weight
        beta: Tightness regularization weight

    Returns:
        Formatted experiment name (e.g., 'coherence_hs128_ep10_cw0.1_a0.2_b0.2')
    """
    hidden_str = format_hidden_sizes(hidden_sizes)
    base_name = f"{experiment_type}_hs{hidden_str}_ep{num_epochs}"

    if experiment_type == "coherence":
        param_str = build_param_string(coherence_weight, alpha, beta)
        return f"{base_name}_{param_str}"
    else:
        return base_name


def create_model_name(
    coherence_weight: float,
    experiment_type: str = None,
    trained: bool = False,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> str:
    """
    Create simplified model names without architecture/epoch information.

    Args:
        coherence_weight: Coherence loss weight
        experiment_type: 'baseline' or 'coherence' (for trained models)
        trained: If True, includes experiment type; if False, just base name
        alpha: Equiangularity regularization weight
        beta: Tightness regularization weight

    Returns:
        Formatted model name (e.g., 'coherence_model.pth' or 'initial_model.pth')
    """
    if trained and experiment_type:
        if experiment_type == "coherence":
            param_str = build_param_string(coherence_weight, alpha, beta)
            return f"coherence_{param_str}_model.pth"
        else:
            return "baseline_model.pth"
    else:
        # For initialized models, use minimal naming
        return "initial_model.pth"


def save_model(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    hidden_sizes: List[int],
    num_epochs: int,
) -> Path:
    """Save the model in the appropriate dataset and config-specific directory."""
    model_config_dir = get_model_config_dir(dataset_name, hidden_sizes, num_epochs)
    model_config_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_config_dir / model_name

    # Only save if model doesn't already exist (for initialized models)
    if not model_path.exists():
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to: {model_path}")
    else:
        print(f"⏭️  Model already exists, skipping: {model_path}")

    return model_path


def save_experiment_config(
    model_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    dataset_name: str,
    hidden_sizes: List[int],
    num_epochs: int,
) -> Path:
    """Save experiment configuration (model + loss) to JSON file."""
    model_config_dir = get_model_config_dir(dataset_name, hidden_sizes, num_epochs)
    config_path = model_config_dir / "experiment_config.json"

    # Only save if config doesn't already exist
    if not config_path.exists():
        experiment_config = {
            "model_config": model_config,
            "loss_config": loss_config,
            "dataset": dataset_name,
            "hidden_sizes": hidden_sizes,
            "num_epochs": num_epochs,
        }

        with open(config_path, "w") as f:
            json.dump(experiment_config, f, indent=2)
        print(f"📋 Experiment config saved to: {config_path}")
    else:
        print(f"⏭️  Experiment config already exists, skipping: {config_path}")

    return config_path


def save_plot(
    plot_filename: str,
    dataset_name: str,
    hidden_sizes: List[int],
    num_epochs: int,
    plot_type: str = "plot",
) -> None:
    """Save plots in the dataset and config-specific directory."""
    plots_config_dir = get_plots_config_dir(dataset_name, hidden_sizes, num_epochs)
    plots_config_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_config_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 {plot_type.title()} saved to: {plot_path}")


def create_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    dataset: str,
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float,
    experiment_type: str,
    alpha: float = 0.0,
    beta: float = 0.0,
    save: bool = False,
):
    """Create training curves with dual y-axes for loss and accuracy."""
    experiment_name = create_experiment_name(
        hidden_sizes,
        num_epochs,
        coherence_weight,
        experiment_type,
        alpha,
        beta,
    )
    print(f"📈 Creating training curves for {experiment_name}...")

    fig, loss_ax = plt.subplots(figsize=(8, 6))

    # Plot losses on primary y-axis
    loss_color = "tab:blue"
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("Loss", color=loss_color, fontweight="bold")
    loss_ax.plot(
        train_losses, label="Train Loss", color="blue", linewidth=2, marker="o"
    )
    loss_ax.plot(val_losses, label="Val Loss", color="orange", linewidth=2, marker="o")
    loss_ax.tick_params(axis="y", labelcolor=loss_color)
    loss_ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.6))
    loss_ax.grid(True, alpha=0.3)

    # Plot accuracies on secondary y-axis
    acc_ax = loss_ax.twinx()
    acc_color = "tab:green"
    acc_ax.set_ylabel("Accuracy", color=acc_color, fontweight="bold")
    acc_ax.plot(
        train_accuracies,
        label="Train Acc",
        color="green",
        linewidth=2,
        marker="s",
        linestyle="--",
    )
    acc_ax.plot(
        val_accuracies,
        label="Val Acc",
        color="red",
        linewidth=2,
        marker="s",
        linestyle="--",
    )
    acc_ax.tick_params(axis="y", labelcolor=acc_color)
    acc_ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.4))

    # Add title with final performance
    final_train_acc = train_accuracies[-1] if train_accuracies else 0
    final_val_acc = val_accuracies[-1] if val_accuracies else 0

    plt.title(
        f"Training Progress - {experiment_name}\n"
        f"Final: Train {final_train_acc:.2f} | Val {final_val_acc:.2f}",
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        plot_filename = f"{experiment_name}_training_curves.png"
        save_plot(plot_filename, dataset, hidden_sizes, num_epochs, "training curves")
    else:
        plt.show()


def create_coherence_plots(
    weights_coherence_stats: Dict[str, Any],
    final_stats: Dict[str, List[float]],
    dataset: str,
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float,
    experiment_type: str,
    alpha: float = 0.0,
    beta: float = 0.0,
    save: bool = False,
):
    """
    Create per-layer coherence plots showing coherence, tightness, equiangularity with bounds.
    """
    experiment_name = create_experiment_name(
        hidden_sizes,
        num_epochs,
        coherence_weight,
        experiment_type,
        alpha,
        beta,
    )
    num_layers = weights_coherence_stats["coherence"].shape[0]
    epochs = weights_coherence_stats["coherence"].shape[1]
    epoch_range = range(epochs)

    print(f"📈 Creating coherence plots for {num_layers} layers...")

    # Create subplots: 1 row x num_layers columns
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))

    # Handle single layer case (axes not in array)
    if num_layers == 1:
        axes = [axes]

    # Shared Y-axis label (will be set on the leftmost subplot)
    shared_ylabel = "Frame Properties"

    # Constants used in all subplots (hoisted out of loop)
    color_coherence = "tab:red"

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        # Extract training data for this layer
        coherence_values = weights_coherence_stats["coherence"][layer_idx, :]
        tightness_values = weights_coherence_stats["tightness"][layer_idx, :]
        equiangularity_values = weights_coherence_stats["equiangularity"][layer_idx, :]

        # Extract final bounds for this layer
        welch_bound = final_stats["welch_bound"][layer_idx]
        upper_bound = final_stats["upper_bound"][layer_idx]

        # Plot coherence on primary y-axis
        ax.set_xlabel("Epoch")

        # Set Y-axis label only on first subplot (shared label)
        if layer_idx == 0:
            ax.set_ylabel(shared_ylabel, color="black", fontweight="bold")

        line1 = ax.plot(
            epoch_range,
            coherence_values,
            color=color_coherence,
            linewidth=2,
            label="Coherence",
            marker="o",
            markersize=3,
        )
        ax.tick_params(axis="y", labelcolor=color_coherence)
        ax.grid(True, alpha=0.3)

        # Add Welch bound and upper bound lines
        bound_lines = []
        if welch_bound > 0:
            line_welch = ax.axhline(
                y=welch_bound,
                color="green",
                linestyle="--",
                linewidth=2,
                label="Welch Bound",
            )
            bound_lines.append(line_welch)

        if upper_bound > 0:
            line_upper = ax.axhline(
                y=upper_bound,
                color="orange",
                linestyle="-.",
                linewidth=2,
                label="Upper Bound",
            )
            bound_lines.append(line_upper)
        line2 = ax.plot(
            epoch_range,
            equiangularity_values,
            color="brown",
            linewidth=2,
            linestyle=":",
            label="Equiangularity",
            marker="^",
            markersize=3,
        )

        # Plot tightness and equiangularity on secondary y-axis
        ax2 = ax.twinx()
        line3 = ax2.plot(
            epoch_range,
            tightness_values,
            color="purple",
            linewidth=2,
            linestyle=":",
            label="Tightness",
            marker="s",
            markersize=3,
        )

        # Y-ticks are automatically set per layer based on data range
        ax2.tick_params(axis="y", labelcolor="purple")

        # Create legend only on the last subplot (axis 0, num_layers - 1)
        if layer_idx == num_layers - 1:
            lines = line1 + line2 + line3
            labels = [line.get_label() for line in lines]
            for bound_line in bound_lines:
                lines.append(bound_line)
                labels.append(bound_line.get_label())

            ax.legend(
                lines,
                labels,
                loc="center right",
                fontsize="small",
                bbox_to_anchor=(1.0, 0.5),
            )

        # Set title for each subplot (layer index)
        ax.set_title(f"Layer {layer_idx + 1}", fontweight="bold")

    # Main title
    fig.suptitle(
        f"Per-Layer Frame Properties - {experiment_name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        plot_filename = f"{experiment_name}_coherence_analysis.png"
        save_plot(
            plot_filename, dataset, hidden_sizes, num_epochs, "coherence analysis"
        )
    else:
        plt.show()


def create_loss_decomposition_plot(
    train_cross_entropy: List[float],
    train_coherence: List[float],
    dataset: str,
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float,
    experiment_type: str,
    alpha: float = 0.0,
    beta: float = 0.0,
    save: bool = False,
):
    """Create loss decomposition plot for coherence experiments with dual y-axes."""
    experiment_name = create_experiment_name(
        hidden_sizes,
        num_epochs,
        coherence_weight,
        experiment_type,
        alpha,
        beta,
    )
    print(f"📊 Creating loss decomposition plot for {experiment_name}...")

    fig, ce_ax = plt.subplots(figsize=(8, 6))

    # Plot cross entropy loss on primary y-axis
    ce_color = "tab:blue"
    ce_ax.set_xlabel("Epochs")
    ce_ax.set_ylabel("Cross Entropy Loss", color=ce_color, fontweight="bold")
    ce_ax.plot(
        train_cross_entropy,
        label="Cross Entropy",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ce_ax.tick_params(axis="y", labelcolor=ce_color)
    ce_ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ce_ax.grid(True, alpha=0.3)

    # Plot coherence loss on secondary y-axis
    coh_ax = ce_ax.twinx()
    coh_color = "tab:red"
    coh_ax.set_ylabel("Coherence Loss", color=coh_color, fontweight="bold")
    coh_ax.plot(
        train_coherence,
        label="Coherence",
        color="red",
        linewidth=2,
        marker="s",
        linestyle="--",
        markersize=4,
    )
    coh_ax.tick_params(axis="y", labelcolor=coh_color)
    coh_ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

    # Add title with final values
    final_ce = train_cross_entropy[-1] if train_cross_entropy else 0
    final_coh = train_coherence[-1] if train_coherence else 0

    plt.title(
        f"Loss Decomposition - {experiment_name}\n"
        f"Final: CE {final_ce:.4f} | Coherence {final_coh:.4f}",
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        plot_filename = f"{experiment_name}_loss_decomposition.png"
        save_plot(
            plot_filename, dataset, hidden_sizes, num_epochs, "loss decomposition"
        )
    else:
        plt.show()


def save_experiment_results(
    dataset: str,
    hidden_sizes: List[int],
    num_epochs: int,
    coherence_weight: float,
    experiment_type: str,
    alpha: float,
    beta: float,
    final_test_loss: float,
    final_test_accuracy: float,
    final_coherence_per_layer: List[float],
    final_tightness_per_layer: List[float],
    final_equiangularity_per_layer: List[float],
    final_welch_bounds: List[float],
    final_upper_bounds: List[float],
    best_model_info: Dict[str, Any] = None,
) -> None:
    """
    Save comprehensive experiment results to JSON in dataset and config-specific directory.
    """
    # Get dataset and config-specific results directory
    results_config_dir = get_results_config_dir(dataset, hidden_sizes, num_epochs)
    results_config_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = create_experiment_name(
        hidden_sizes, num_epochs, coherence_weight, experiment_type, alpha, beta
    )
    print("💾 Saving experiment results...")

    # Prepare core experiment data
    results_data = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "hidden_sizes": hidden_sizes,
        "num_epochs": num_epochs,
        "coherence_weight": coherence_weight,
        "alpha": alpha,
        "beta": beta,
        "experiment_type": experiment_type,
        "num_layers": len(final_coherence_per_layer),
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
    }

    # Add best model information if available
    if best_model_info:
        results_data.update(
            {
                "best_epoch": best_model_info["best_epoch"],
                "best_val_accuracy": best_model_info["best_val_accuracy"],
            }
        )

    # Add per-layer coherence analysis
    for layer_idx, (
        coherence,
        tightness,
        equiangularity,
        welch_bound,
        upper_bound,
    ) in enumerate(
        zip(
            final_coherence_per_layer,
            final_tightness_per_layer,
            final_equiangularity_per_layer,
            final_welch_bounds,
            final_upper_bounds,
        )
    ):
        layer_name = f"layer_{layer_idx + 1}"
        results_data.update(
            {
                f"{layer_name}_coherence": coherence,
                f"{layer_name}_tightness": tightness,
                f"{layer_name}_equiangularity": equiangularity,
                f"{layer_name}_welch_bound": welch_bound,
                f"{layer_name}_upper_bound": upper_bound,
            }
        )

    # Save results as JSON
    json_filename = f"{experiment_name}_results.json"
    json_path = results_config_dir / json_filename

    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"📄 Results saved to: {json_path}")

    print(f"✅ Results successfully saved for {experiment_name}")


def configure_baseline_experiment(config: Dict[str, Any]) -> None:
    """Configure baseline experiment settings."""
    config["loss_config"]["type"] = "cross_entropy"
    config["loss_config"]["coherence_weight"] = 0.0
    config["experiment_name"] = f"{config['dataset']}_baseline"
    print("📊 Configured for BASELINE experiment (CrossEntropy loss only)")


def configure_coherence_experiment(config: Dict[str, Any]) -> None:
    """Configure coherence experiment settings."""
    config["loss_config"]["type"] = "cross_entropy_with_coherence"
    config["experiment_name"] = f"{config['dataset']}_coherence"
    coh_weight = config["loss_config"]["coherence_weight"]
    alpha = config["loss_config"].get("alpha", 0.0)
    beta = config["loss_config"].get("beta", 0.0)
    print("🔗 Configured for COHERENCE experiment:")
    print(f"   💪 Coherence weight: {coh_weight}")
    print(f"   📐 Alpha (equiangularity): {alpha}")
    print(f"   🎯 Beta (tightness): {beta}")


def select_experiment_config(experiment_type: str) -> Dict[str, Any]:
    """
    Load and configure experiment from YAML file.

    Args:
        experiment_type: "baseline" or "coherence"

    Returns:
        Configuration dictionary with proper loss type set
    """
    print("📋 Loading experiment configuration...")

    # Load base configuration from YAML
    config_file = CONFIG_DIR / "mlp_coherence.yaml"
    config = load_yaml_config(config_file)

    # Validate configuration
    required_keys = [
        "dataset",
        "model_config",
        "loss_config",
        "batch_size",
        "num_epochs",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"❌ Missing required configuration key: {key}")

    # Configure experiment type using switch-like mapping
    experiment_configs = {
        "baseline": configure_baseline_experiment,
        "coherence": configure_coherence_experiment,
    }

    if experiment_type not in experiment_configs:
        available_experiments = list(experiment_configs.keys())
        raise ValueError(
            f"❌ Unknown experiment type: {experiment_type}. Available: {available_experiments}"
        )

    experiment_configs[experiment_type](config)
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("experiment", type=str, choices=["baseline", "coherence"])
    parser.add_argument(
        "--save", action="store_true", help="Save plots and model instead of displaying"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def run_experiment(
    experiment_type: str, save_outputs: bool = False, seed: int = 42
) -> None:
    """Execute a single training experiment with model existence checking."""
    print("\n" + "=" * 60)
    print(f"🎯 Starting {experiment_type.upper()} Experiment with seed {seed}")
    print("=" * 60)

    seed_everything(seed)

    # Load configuration
    config = select_experiment_config(experiment_type)
    dataset_name = config["dataset"]
    hidden_sizes = config["model_config"]["hidden_sizes"]
    num_epochs = config["num_epochs"]
    coherence_weight = config["loss_config"]["coherence_weight"]
    alpha = config["loss_config"].get("alpha", 0.0)
    beta = config["loss_config"].get("beta", 0.0)

    # Check if trained model already exists
    trained_model_name = create_model_name(
        coherence_weight,
        experiment_type,
        trained=True,
        alpha=alpha,
        beta=beta,
    )

    if (
        model_exists(trained_model_name, dataset_name, hidden_sizes, num_epochs)
        and save_outputs
    ):
        print(f"🔍 Trained model already exists: {trained_model_name}")
        print("⏭️  Skipping training and loading existing model...")

        # Setup device and model for evaluation only
        device = setup_device()
        model = create_model_and_summary(config["model_config"], device)

        # Load existing trained model
        model_config_dir = get_model_config_dir(dataset_name, hidden_sizes, num_epochs)
        model_path = model_config_dir / trained_model_name
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from: {model_path}")

        # Setup test data for evaluation
        _, _, test_loader = setup_dataloaders(
            batch_size=config["batch_size"],
            dataset_name=dataset_name,
            train_proportion=config["train_proportion"],
        )

        # Setup loss function for evaluation
        loss_function, _ = setup_loss_and_optimizer(model, config)

        # Evaluate model
        final_loss, final_accuracy = evaluate_model(
            model, test_loader, loss_function, device
        )
        print(
            f"🎯 Model performance - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%"
        )

        # No training data available
        weights_coherence_stats = None
        loss_components = {}

    else:
        # Setup components for training
        device = setup_device()

        train_loader, val_loader, test_loader = setup_dataloaders(
            batch_size=config["batch_size"],
            dataset_name=dataset_name,
            train_proportion=config["train_proportion"],
        )

        model = create_model_and_summary(config["model_config"], device)

        # Save initialized model and config (only once per architecture)
        print("\n💾 Saving initialized model and config...")
        init_model_name = create_model_name(
            0.0,
            trained=False,
            alpha=0.0,
            beta=0.0,
        )
        save_model(model, init_model_name, dataset_name, hidden_sizes, num_epochs)
        save_experiment_config(
            config["model_config"],
            config["loss_config"],
            dataset_name,
            hidden_sizes,
            num_epochs,
        )

        # Setup loss and optimizer
        loss_function, optimizer = setup_loss_and_optimizer(model, config)

        # Initial evaluation
        print("\n📊 Initial model evaluation...")
        initial_loss, initial_accuracy = evaluate_model(
            model, test_loader, loss_function, device
        )
        print(
            f"🎯 Initial - Loss: {initial_loss:.4f}, Accuracy: {initial_accuracy:.2f}%"
        )

        # Execute training
        training_results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            num_epochs=config["num_epochs"],
        )

        # Unpack training results
        (
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            weights_coherence_stats,
            loss_components,
            best_model_info,
        ) = training_results

        # Final evaluation with best model
        print("\n📊 Final evaluation (best model)...")
        final_loss, final_accuracy = evaluate_model(
            model, test_loader, loss_function, device
        )

        # Display performance summary
        best_epoch = best_model_info["best_epoch"]
        best_val_acc = best_model_info["best_val_accuracy"]
        print(
            f"🎯 Final Test - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%"
        )
        print(
            f"🏆 Best model from epoch {best_epoch + 1} (Val acc: {best_val_acc:.2f}%)"
        )

        # Create visualizations
        print("\n📊 Creating visualizations...")
        create_training_plots(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            dataset_name,
            hidden_sizes,
            num_epochs,
            coherence_weight,
            experiment_type,
            alpha,
            beta,
            save=save_outputs,
        )

        # Create loss decomposition plot for coherence experiments
        if experiment_type == "coherence" and loss_components:
            create_loss_decomposition_plot(
                loss_components["train_cross_entropy"],
                loss_components["train_coherence"],
                dataset_name,
                hidden_sizes,
                num_epochs,
                coherence_weight,
                experiment_type,
                alpha,
                beta,
                save=save_outputs,
            )

        # Save best trained model if requested
        if save_outputs:
            print(f"\n💾 Saving best model (epoch {best_epoch + 1})...")
            save_model(
                model, trained_model_name, dataset_name, hidden_sizes, num_epochs
            )

        # Compute final coherence statistics and create plots
        final_coherence_stats = get_weights_coherence_stats(model, only_essential=False)
        create_coherence_plots(
            weights_coherence_stats,
            final_coherence_stats,
            dataset_name,
            hidden_sizes,
            num_epochs,
            coherence_weight,
            experiment_type,
            alpha,
            beta,
            save=save_outputs,
        )

        # Save experimental results if requested
        if save_outputs:
            print("\n📊 Saving experiment results...")
            save_experiment_results(
                dataset_name,
                hidden_sizes,
                num_epochs,
                coherence_weight,
                experiment_type,
                alpha,
                beta,
                final_loss,
                final_accuracy,
                final_coherence_stats["coherence"],
                final_coherence_stats["tightness"],
                final_coherence_stats["equiangularity"],
                final_coherence_stats["welch_bound"],
                final_coherence_stats["upper_bound"],
                best_model_info,
            )

    print(f"\n✅ {experiment_type.upper()} experiment completed successfully!")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.save:
        print("💾 Save mode: Plots and models will be saved to disk")

    try:
        run_experiment(args.experiment, save_outputs=args.save, seed=args.seed)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
