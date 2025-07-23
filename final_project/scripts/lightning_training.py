import json
import yaml
import torch
import argparse
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torchsummary import summary
from rich import print

from src.lightning_model import Classifier
from src.mlp import MLPClassifier
from src.cnn import CNNClassifier
from src.gcnn_model import build_gcnn_model
from src.datamodule import ImageDataModule

# Global constants
PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = PARENT_DIR / "data"
CHECKPOINT_DIR = PARENT_DIR / "saved_models" / "lightning"
LOGS_DIR = PARENT_DIR / "logs"
CONFIGS_DIR = PARENT_DIR / "configs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)



def resolve_model_config_path(model_name):
    """Resolve model name to configuration file path."""
    valid_models = ["mlp", "cnn", "gcnn"]

    if model_name not in valid_models:
        raise ValueError(
            f"❌ Invalid model '{model_name}'. Available models: {valid_models}"
        )

    config_path = CONFIGS_DIR / f"{model_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Configuration file not found: {config_path}")

    return config_path


def load_configuration(config_path):
    """Load and validate configuration from YAML file."""
    print(f"📋 Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = [
            "dataset",
            "datasets",
            "model",
            "data",
            "training",
            "optimizer",
        ]
        missing_sections = [
            section for section in required_sections if section not in config
        ]

        if missing_sections:
            raise ValueError(
                f"❌ Missing required sections in config: {missing_sections}"
            )

        # Validate dataset configuration exists
        selected_dataset = config["dataset"]
        if selected_dataset not in config["datasets"]:
            available_datasets = list(config["datasets"].keys())
            raise ValueError(
                f"❌ Dataset '{selected_dataset}' not found. Available: {available_datasets}"
            )

        print("✅ Configuration loaded successfully")
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"❌ Invalid YAML format in config file: {e}")


def extract_dataset_config(config):
    """Extract dataset configuration for the selected dataset."""
    selected_dataset = config["dataset"]
    dataset_config = config["datasets"][selected_dataset]

    print(f"📊 Using dataset: {selected_dataset}")
    print(f"   Classes: {dataset_config['num_classes']}")

    return dataset_config


def build_mlp_model(dataset_config, model_config):
    """Build MLP model with specified configuration."""
    return MLPClassifier(
        input_size=dataset_config["input_size"],
        num_classes=dataset_config["num_classes"],
        hidden_sizes=model_config["hidden_sizes"],
        dropout_rate=model_config["dropout_rate"],
        use_batch_norm=model_config["use_batch_norm"],
        activation=model_config["activation"],
    )


def build_cnn_model(dataset_config, model_config):
    """Build CNN model with specified configuration."""
    return CNNClassifier(
        in_channels=dataset_config["in_channels"],
        num_classes=dataset_config["num_classes"],
        kernel_size=model_config["kernel_size"],
        num_hidden=model_config["num_hidden"],
        hidden_channels=model_config["hidden_channels"],
        padding=model_config["padding"],
        stride=model_config["stride"],
    )


def create_model_from_config(config):
    """Factory function to create the appropriate model based on configuration."""
    dataset_config = extract_dataset_config(config)
    model_config = config["model"]
    model_type = model_config["type"]

    if model_type == "mlp":
        model = build_mlp_model(dataset_config, model_config)
    elif model_type == "cnn":
        model = build_cnn_model(dataset_config, model_config)
    elif model_type == "gcnn":
        model = build_gcnn_model(dataset_config, model_config)
    else:
        raise ValueError(f"❌ Unknown model type: {model_type}")

    return model


def setup_data_module(config, seed):
    """Set up Lightning data module from configuration."""
    data_config = config["data"]
    dataset_name = config["dataset"]

    print("📁 Setting up data module...")
    print(f"   Dataset: {dataset_name}")

    if data_config["random_rotation_train"] or data_config["random_rotation_test"]:
        print("   🔄 Data augmentation: Rotation enabled")

    return ImageDataModule(
        dataset_name=dataset_name,
        data_dir=str(DATA_DIR),
        batch_size=config["data"]["batch_size"],
        train_proportion=data_config["train_proportion"],
        random_rotation_train=data_config["random_rotation_train"],
        random_rotation_test=data_config["random_rotation_test"],
        num_workers=2,
        seed=seed,
    )


def configure_training_components(config, experiment_name):
    """Configure training callbacks and loggers."""
    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        mode="max",
        monitor="val_accuracy",
        filename=experiment_name,
        dirpath=str(CHECKPOINT_DIR),
    )

    # Set up logging
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(LOGS_DIR),
        name=experiment_name,
        default_hp_metric=False,
    )

    csv_logger = CSVLogger(
        save_dir=str(LOGS_DIR),
        name=experiment_name,
        version=tensorboard_logger.version,
    )

    return checkpoint_callback, [tensorboard_logger, csv_logger]


def create_trainer(config, checkpoint_callback, loggers):
    """Create Lightning trainer with specified configuration."""
    # Disable deterministic mode for GCNN models due to non-deterministic operations
    model_type = config["model"]["type"]
    deterministic = model_type != "gcnn"  # GCNN uses grid_sampler_3d which is non-deterministic
    
    return Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=config["training"]["max_epochs"],
        deterministic=deterministic,
        callbacks=[checkpoint_callback],
        logger=loggers,
        log_every_n_steps=config["training"]["log_every_n_steps"],
    )


def extract_checkpoint_epoch(checkpoint_path):
    """Extract current epoch from checkpoint with error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = checkpoint.get("epoch", 0)
        # Ensure epoch is an integer
        return int(epoch) if epoch is not None else 0
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"⚠️  Warning: Could not read checkpoint epoch: {e}")
        return 0


def show_model_architecture(model, config):
    """Display model summary with correct input shape based on model type."""
    dataset_config = extract_dataset_config(config)
    model_type = config["model"]["type"]

    print("🏗️  Model Architecture Summary:")

    if model_type == "mlp":
        input_shape = (dataset_config["input_size"],)
    else:  # CNN or GCNN
        input_shape = tuple(dataset_config["input_size"])

    # Always use CPU for summary to avoid device mismatch issues
    model_cpu = model.cpu()
    try:
        summary(model_cpu, input_size=input_shape)
    except Exception as e:
        print(f"⚠️  Could not display model summary: {e}")
        print("   Continuing with training...")
    finally:
        # Move model back to GPU if available
        if torch.cuda.is_available():
            model.to('cuda')


def save_experiment_results(
    config, experiment_name, test_results, model_checkpoint_path
):
    """Save experiment results to JSON file."""
    results = {
        "experiment_name": experiment_name,
        "config": config,
        "test_results": test_results,
        "checkpoint_path": model_checkpoint_path,
    }

    results_file_path = LOGS_DIR / f"{experiment_name}_results.json"

    try:
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved: {results_file_path}")
        return results_file_path
    except Exception as e:
        print(f"⚠️  Warning: Could not save results: {e}")
        return None


def handle_checkpoint_and_training(
    config, experiment_name, base_model, trainer, checkpoint_callback
):
    """Handle checkpoint loading and training logic."""
    checkpoint_file_path = CHECKPOINT_DIR / f"{experiment_name}.ckpt"

    if checkpoint_file_path.exists():
        current_epoch = extract_checkpoint_epoch(str(checkpoint_file_path))
        target_epochs = config["training"]["max_epochs"]

        print(f"🔍 Found existing checkpoint: {checkpoint_file_path.name}")
        print(f"   Current epoch: {current_epoch} | Target: {target_epochs}")

        if current_epoch >= target_epochs:
            print("🎯 Target epochs reached. Loading model for testing...")
            trained_model = Classifier.load_from_checkpoint(
                str(checkpoint_file_path), model=base_model
            )
            # Use same deterministic logic for test trainer
            model_type = config["model"]["type"]
            deterministic = model_type != "gcnn"
            test_trainer = Trainer(accelerator="auto", devices="auto", logger=False, deterministic=deterministic)
            return trained_model, test_trainer, str(checkpoint_file_path)
        else:
            print(f"🔄 Resuming training from epoch {current_epoch}...")
            training_model = Classifier(
                base_model,
                optimizer_config=config["optimizer"],
            )
            return training_model, trainer, None
    else:
        print("🆕 No checkpoint found. Starting fresh training...")

        print(f"\n{'=' * 60}")
        show_model_architecture(base_model, config)
        print(f"{'=' * 60}")

        new_model = Classifier(
            base_model,
            optimizer_config=config["optimizer"],
        )
        return new_model, trainer, None


def execute_training_pipeline(model_name, seed):
    """Execute the complete training pipeline for the specified model."""
    print("🚀 Starting Training Pipeline")
    print(f"🎯 Model: {model_name.upper()}")
    print(f"🌱 Seed: {seed}")

    # Load configuration
    config_path = resolve_model_config_path(model_name)
    config = load_configuration(config_path)
    experiment_name = f"{config['dataset']}_{config['model']['type']}"

    # Initialize environment
    print("🌱 Initializing environment...")
    seed_everything(seed, workers=True)

    # Create necessary directories
    print("📂 Creating directories...")
    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Set up data module
    data_module = setup_data_module(config, seed)

    # Create model
    base_model = create_model_from_config(config)

    # Configure training components
    checkpoint_callback, loggers = configure_training_components(
        config, experiment_name
    )
    trainer = create_trainer(config, checkpoint_callback, loggers)

    # Display experiment info
    print(f"\n{'=' * 60}")
    print(f"🧪 EXPERIMENT: {experiment_name.upper()}")
    print(
        f"📊 Train/Val Split: {config['data']['train_proportion']:.1%}/{(1 - config['data']['train_proportion']):.1%}"
    )
    print(
        f"⚡ Optimizer: {config['optimizer']['type'].upper()} (lr={config['optimizer']['learning_rate']})"
    )
    print(f"{'=' * 60}")

    # Handle training or loading
    model, final_trainer, existing_checkpoint_path = handle_checkpoint_and_training(
        config, experiment_name, base_model, trainer, checkpoint_callback
    )

    # Execute training if needed
    if existing_checkpoint_path is None:
        checkpoint_file_path = CHECKPOINT_DIR / f"{experiment_name}.ckpt"

        print("🏋️  Starting training...")

        if checkpoint_file_path.exists():
            trainer.fit(model, data_module, ckpt_path=str(checkpoint_file_path))
        else:
            trainer.fit(model, data_module)

        best_model_path = checkpoint_callback.best_model_path
        print("✅ Training completed!")
    else:
        best_model_path = existing_checkpoint_path
        print("⏭️  Skipping training - using existing model")

    # Run testing
    print("🧪 Running testing...")
    test_results = final_trainer.test(model, data_module, verbose=False)[0]
    print(f"🎯 Test Accuracy: {test_results['test_accuracy']:.4f}")

    # Save results
    save_experiment_results(config, experiment_name, test_results, best_model_path)

    # Display final results
    print(f"\n{'=' * 60}")
    print("🎉 EXPERIMENT COMPLETED")
    print(f"{'=' * 60}")
    print(f"📈 Monitor training: tensorboard --logdir={LOGS_DIR}")
    print(f"{'=' * 60}")

    return model, test_results


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="🚀 Train deep learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python lightning_training.py --model mlp     # Train MLP model
            python lightning_training.py --model cnn     # Train CNN model
            python lightning_training.py --model gcnn    # Train GCNN model
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "gcnn"],
        help="Model type to train (mlp, cnn, or gcnn)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    try:
        execute_training_pipeline(args.model, args.seed)
        print("🏁 Training completed successfully!")
        return 0
    except Exception as error:
        print(f"❌ Error: {error}")
        return 1


if __name__ == "__main__":
    exit(main())
