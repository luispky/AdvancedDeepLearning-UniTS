from typing import Tuple, List, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import random_split
from src.mlp import MLPClassifier
from src.minimal_coherence_optimization import (
    surrogate_coherence_loss,
    compute_frame_properties,
)

PARENT_DIR = Path(__file__).parent.parent
DATA_DIR = PARENT_DIR / "data"


class CoherenceLoss(nn.Module):
    """
    Flexible loss function that can use either:
    1. Standard CrossEntropy loss
    2. CrossEntropy + Coherence regularization on MLP weights

    For coherence: each linear layer with weights W (dim_out, dim_in), we compute coherence loss
    on X = W where columns are treated as frame vectors because in_features > out_features.
    """

    def __init__(
        self,
        loss_type: str = "cross_entropy",
        coherence_weight: float = 0.0,
        lambda_softmax: float = 20.0,
        alpha: float = 0.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.loss_type = loss_type
        self.coherence_weight = coherence_weight
        self.lambda_softmax = lambda_softmax
        self.alpha = alpha
        self.beta = beta

        self.use_coherence = (
            loss_type == "cross_entropy_with_coherence" and coherence_weight > 0
        )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        """
        Compute combined loss = CrossEntropy + coherence_weight * Σ CoherenceLoss(W_i)

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            model: The MLP model to extract weights from

        Returns:
            Combined loss value
        """
        # Standard cross-entropy loss
        cross_entropy_loss = self.cross_entropy(predictions, targets)

        if not self.use_coherence:
            return cross_entropy_loss

        # Compute coherence loss for each linear layer
        layer_coherence_losses = []
        for layer_module in model.modules():
            if isinstance(layer_module, nn.Linear):
                weight_matrix = layer_module.weight  # (out_features, in_features)
                # Here, in_features > out_features
                frame_matrix = weight_matrix  # columns are frame vectors
                # (vector_space_dim, num_frame_vectors)
                # num_frame_vectors > vector_space_dim

                if frame_matrix.shape[1] > 1:  # Need at least 2 vectors for coherence
                    # Create a copy to avoid inplace operations on model parameters
                    # frame_matrix_copy still tracks the gradient of frame_matrix
                    # but avoids inplace operations on frame_matrix which is performed
                    # when we call surrogate_coherence_loss
                    frame_matrix_copy = frame_matrix.clone()
                    layer_coh_loss = surrogate_coherence_loss(
                        frame_matrix_copy,
                        lambda_softmax=self.lambda_softmax,
                        alpha=self.alpha,
                        beta=self.beta,
                    )
                    layer_coherence_losses.append(layer_coh_loss)

        total_coherence_loss = (
            sum(layer_coherence_losses)
            if layer_coherence_losses
            else torch.tensor(0.0, device=cross_entropy_loss.device)
        )

        total_loss = cross_entropy_loss + self.coherence_weight * total_coherence_loss

        return total_loss, cross_entropy_loss, total_coherence_loss


def create_datasets(dataset_class, transform_config: dict, train_proportion: float):
    """Create train, validation, and test datasets."""
    train_dataset_complete = dataset_class(
        root=DATA_DIR, train=True, transform=transform_config, download=True
    )

    # Calculate split sizes based on train_proportion
    total_train_size = len(train_dataset_complete)
    train_size = int(total_train_size * train_proportion)
    val_size = total_train_size - train_size

    # Split training data into train and validation
    train_data, val_data = random_split(
        train_dataset_complete,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = dataset_class(
        root=DATA_DIR, train=False, transform=transform_config, download=True
    )
    return train_data, val_data, test_dataset


def setup_dataloaders(
    batch_size: int = 64,
    dataset_name: str = "mnist",
    train_proportion: float = 0.9,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Setup train, validation, and test data loaders for specified dataset.

    Args:
        batch_size: Batch size for data loaders
        dataset_name: Name of dataset ("mnist", "cifar10", "cifar100", "fashion_mnist")
        train_proportion: Proportion of training data to use for training (rest for validation)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"📊 Setting up {dataset_name.upper()} dataset with batch size {batch_size}")

    # Dataset normalization parameters
    normalize_params = {
        "mnist": {"mean": (0.1307,), "std": (0.3081,)},
        "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
        "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
        "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    }

    if dataset_name not in normalize_params:
        raise ValueError(f"❌ Dataset {dataset_name} not supported")

    # Create transform
    transform_config = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(**normalize_params[dataset_name])]
    )

    # Dataset mapping
    dataset_classes = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "fashion_mnist": datasets.FashionMNIST,
    }

    train_dataset, val_dataset, test_dataset = create_datasets(
        dataset_classes[dataset_name], transform_config, train_proportion
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(
        f"✅ Data loaders ready - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} samples"
    )
    return train_loader, val_loader, test_loader


def calculate_accuracy(model_outputs: torch.Tensor, true_labels: torch.Tensor) -> int:
    """Calculate number of correct predictions."""
    _, predicted_labels = model_outputs.max(dim=1)
    return predicted_labels.eq(true_labels).sum().item()


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, loss_function, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on given dataset.

    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs, batch_targets = (
                batch_inputs.to(device),
                batch_targets.to(device),
            )
            model_outputs = model(batch_inputs)

            loss_output = loss_function(model_outputs, batch_targets, model)

            # Handle both single loss and tuple (total_loss, ce_loss, coh_loss) returns
            batch_loss = (
                loss_output[0] if isinstance(loss_output, tuple) else loss_output
            )

            total_loss += batch_loss.item()
            correct_predictions += calculate_accuracy(model_outputs, batch_targets)
            total_samples += batch_targets.size(dim=0)

    average_loss = total_loss / len(data_loader)
    accuracy_percentage = correct_predictions / total_samples
    return average_loss, accuracy_percentage


def count_linear_layers(model: nn.Module) -> int:
    """
    Counts the number of nn.Linear layers in a PyTorch model.

    Parameters:
        model (nn.Module): The PyTorch model to analyze.

    Returns:
        int: The number of nn.Linear layers in the model.
    """
    return len([module for module in model.modules() if isinstance(module, nn.Linear)])


def get_weights_coherence_stats(
    model: nn.Module, only_essential: bool = True
) -> Dict[str, List[float]]:
    """
    Get the coherence statistics of the weights of the model.

    Args:
        model: PyTorch model with Linear layers
        only_essential: If True, compute only coherence, tightness, equiangularity

    Returns:
        Dictionary with lists of properties for each linear layer
    """
    frame_properties = {
        "coherence": [],
        "tightness": [],
        "equiangularity": [],
    }

    # Add additional properties if requested
    if not only_essential:
        frame_properties.update(
            {
                "welch_bound": [],
                "upper_bound": [],
            }
        )

    for layer_module in model.modules():
        if isinstance(layer_module, nn.Linear):
            weight_matrix = layer_module.weight  # (out_features, in_features)
            frame_properties_dict = compute_frame_properties(
                weight_matrix, only_essential=only_essential
            )

            # Add essential properties
            for key in ["coherence", "tightness", "equiangularity"]:
                frame_properties[key].append(frame_properties_dict[key])

            # Add additional properties if computed
            if not only_essential:
                for key in ["welch_bound", "upper_bound"]:
                    frame_properties[key].append(frame_properties_dict[key])

    return frame_properties


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function,
    optimizer,
    device: torch.device,
    num_epochs: int,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    Dict[str, np.ndarray],
    Dict[str, List[float]],
    Dict[str, Any],
]:
    """
    Train the model and track essential coherence metrics with best model tracking.

    Returns:
        Tuple of (train_losses, val_losses, train_accuracies, val_accuracies,
                 weights_coherence_stats, loss_components, best_model_info)
    """
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    # Training metrics tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Best model tracking
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_state = model.state_dict().copy()

    # Check if this is a coherence experiment
    is_coherence_experiment = (
        hasattr(loss_function, "use_coherence") and loss_function.use_coherence
    )

    # Initialize loss component tracking for coherence experiments
    loss_components = (
        {
            "train_cross_entropy": [],
            "train_coherence": [],
        }
        if is_coherence_experiment
        else {}
    )

    # Initialize coherence statistics tracking
    num_linear_layers = count_linear_layers(model)
    weights_coherence_stats = {
        "coherence": np.zeros((num_linear_layers, num_epochs + 1)),
        "tightness": np.zeros((num_linear_layers, num_epochs + 1)),
        "equiangularity": np.zeros((num_linear_layers, num_epochs + 1)),
    }

    # Save initial weights coherence stats
    initial_stats = get_weights_coherence_stats(model, only_essential=True)
    for key in weights_coherence_stats:
        weights_coherence_stats[key][:, 0] = initial_stats[key]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        epoch_total_loss, epoch_ce_loss, epoch_coh_loss = 0.0, 0.0, 0.0
        correct_predictions, total_samples = 0, 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = (
                batch_inputs.to(device),
                batch_targets.to(device),
            )

            # Forward pass
            optimizer.zero_grad()
            model_outputs = model(batch_inputs)
            loss_output = loss_function(model_outputs, batch_targets, model)

            # Handle both single loss and tuple returns
            if isinstance(loss_output, tuple):
                # Coherence experiment: (total_loss, ce_loss, coh_loss)
                batch_total_loss, batch_ce_loss, batch_coh_loss = loss_output
                epoch_ce_loss += batch_ce_loss.item()
                epoch_coh_loss += batch_coh_loss.item()
            else:
                # Baseline experiment: just total loss
                batch_total_loss = loss_output

            epoch_total_loss += batch_total_loss.item()

            # Backward pass
            batch_total_loss.backward()
            optimizer.step()

            # Track metrics
            correct_predictions += calculate_accuracy(model_outputs, batch_targets)
            total_samples += batch_targets.size(dim=0)

        # Calculate training metrics
        avg_train_loss = epoch_total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Track loss components for coherence experiments
        if is_coherence_experiment:
            avg_ce_loss = epoch_ce_loss / len(train_loader)
            avg_coh_loss = epoch_coh_loss / len(train_loader)
            loss_components["train_cross_entropy"].append(avg_ce_loss)
            loss_components["train_coherence"].append(avg_coh_loss)

        # Validation phase
        val_loss, val_accuracy = evaluate_model(
            model, val_loader, loss_function, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Track best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        # Track coherence statistics
        epoch_stats = get_weights_coherence_stats(model, only_essential=True)
        for key in weights_coherence_stats:
            weights_coherence_stats[key][:, epoch + 1] = epoch_stats[key]

        # Print epoch summary
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    print(
        f"\n✅ Training completed! Best model from epoch {best_epoch + 1} with {best_val_accuracy:.2f}% val accuracy"
    )

    # Prepare best model info
    best_model_info = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state_dict": best_model_state,
    }

    return (
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        weights_coherence_stats,
        loss_components,
        best_model_info,
    )


def create_model_and_summary(
    model_config: Dict[str, Any], device: torch.device
) -> nn.Module:
    """Create model and display summary."""
    print("\n🧠 Creating MLP model...")

    model = MLPClassifier(**model_config).to(device)

    # Display model summary
    print("📋 Model Architecture:")
    summary(model, (model_config["input_size"],))
    return model


def setup_loss_and_optimizer(
    model: nn.Module, config: Dict[str, Any]
) -> Tuple[nn.Module, optim.Optimizer]:
    """Setup loss function and optimizer based on configuration."""
    print("⚡ Setting up loss function and optimizer...")

    # Extract loss configuration
    loss_config = config["loss_config"]
    loss_type = loss_config["type"]

    # Setup loss function with all parameters
    loss_function = CoherenceLoss(
        loss_type=loss_type,
        coherence_weight=loss_config.get("coherence_weight", 0.0),
        lambda_softmax=loss_config.get("lambda_softmax", 50.0),
        alpha=loss_config.get("alpha", 0.0),
        beta=loss_config.get("beta", 0.0),
    )

    # Display loss configuration
    if loss_type == "cross_entropy_with_coherence":
        coherence_weight = loss_config["coherence_weight"]
        print(f"🔗 Using CrossEntropy + Coherence loss (weight: {coherence_weight})")
        print(f"   Lambda: {loss_config.get('lambda_softmax', 20.0)}")
        print(f"   Alpha: {loss_config.get('alpha', 0.0)}")
        print(f"   Beta: {loss_config.get('beta', 0.0)}")
    else:
        print("📊 Using standard CrossEntropy loss")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    print(
        f"⚡ Optimizer: AdamW (lr={config['learning_rate']}, weight_decay={config['weight_decay']})"
    )

    return loss_function, optimizer


def setup_device() -> torch.device:
    """Setup and return the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "GPU" if device.type == "cuda" else "CPU"
    print(f"🔧 Using device: {device_name} ({device})")
    return device
