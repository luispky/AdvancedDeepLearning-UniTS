# Import necessary libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
from safetensors.torch import save_file, load_file    
from typing import List, Optional, Sequence    
import logging


# Base directory for utils.py
BASE_DIR = Path(__file__).resolve().parent

# Paths for saving results
MODELS_DIR = BASE_DIR.parent / "models"
RESULTS_DIR = BASE_DIR.parent / "results"
FIGURES_DIR = BASE_DIR.parent / "figures"
LOGS_DIR = BASE_DIR.parent / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def save_model(model: torch.nn.Module, filename: str) -> None:
    """
    Parameters:
      model (torch.nn.Module): The PyTorch model to be saved.
      filename (str): The filename (without extension) to save the model's state_dict.
    """
    filepath = MODELS_DIR / f"{filename}.safetensors"
    if not filepath.exists():
        save_file(model.state_dict(), str(filepath))
        print(f"Model saved at {filepath}")
    else:
        print(f"Model already exists at {filepath}")


def load_model(model: torch.nn.Module, filename: str) -> torch.nn.Module:
    """
    Parameters:
      model (torch.nn.Module): An instance of the model architecture to be loaded.
      filename (str): The filename (without extension) from which to load the state_dict.

    Returns:
      torch.nn.Module: The model with the loaded state_dict.
    """
    filepath = MODELS_DIR / f"{filename}.safetensors"
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}")
    state_dict = load_file(str(filepath))
    model.load_state_dict(state_dict)
    print(f"Model loaded from {filepath}")
    return model


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Parameters:
    - seed (int): Random seed value.

    Returns:
    - None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(model: nn.Module, 
                   data_loader: DataLoader,
                   device: torch.device = torch.device("cpu")) -> float:
    """
    Evaluate the model on the given data loader and return the average MSE loss.
    
    Parameters:
      model       : The PyTorch model to evaluate.
      data_loader : DataLoader for the test dataset.
      device      : Torch device.
      
    Returns:
      The average MSE loss on the dataset.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def plot_train_and_test_loss(
    train_losses: List[float],
    test_losses: List[float],
    title: str,
    eval_schedule: Optional[Sequence] = None,
    save_fig: bool = False,
    filename: Optional[str] = None,
    use_log_scale: bool = False
) -> None:
    """
    Plots training and testing loss over epochs or iterations with optional scientific notation or log scale.
    
    Parameters:
    - train_losses (List[float]): Training loss values per epoch.
    - test_losses (List[float]): Testing loss values.
    - title (str): Title for the plot.
    - eval_schedule (Optional[Sequence]): Epoch indices corresponding to test loss evaluations.
    - save_fig (bool): Whether to save the plot as an image.
    - filename (Optional[str]): Filename for saving.
    - use_log_scale (bool): Whether to plot the y-axis in log scale (default: False).
    """
    
    sns.set_style("whitegrid")  # Aesthetic style

    # Handle x-axis mapping
    if eval_schedule is None:
        if len(train_losses) != len(test_losses):
            raise ValueError("If no evaluation schedule is provided, train_losses and test_losses must be the same length.")
        x_train = list(range(len(train_losses)))
        x_test = x_train  # Test losses align with train losses per epoch
        x_label = "Epochs"
    else:
        if len(eval_schedule) != len(test_losses):
            raise ValueError("eval_schedule length must match test_losses length.")
        x_train = list(range(len(train_losses)))
        x_test = eval_schedule
        x_label = "Epochs (Train) / Iterations (Test)"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training and test loss
    ax.plot(x_train, train_losses, label="Train Loss", color="blue", linestyle="--", linewidth=2)
    ax.plot(x_test, test_losses, label="Test Loss", color="orange", linestyle="--", linewidth=2)

    # Labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    # Option 1: Scientific notation for better readability
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 2))  # Use scientific notation for values below 1e-1 or above 1e2
    ax.yaxis.set_major_formatter(formatter)

    # Option 2: Use logarithmic scale if requested
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Log Loss (log scale)", fontsize=12)

    plt.tight_layout()

    # Save or show
    if save_fig:
        if filename is None:
            filename = "train_test_loss"
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {filepath}")
        plt.close()
    else:
        plt.show()


def setup_logging(log_file: str, log_level: int = logging.INFO) -> None:
    """
    Configures logging to output messages to both a file and the console.

    Parameters:
      log_file (str): The file path where log messages will be saved.
      log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Remove any existing logging handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with both a file handler and a stream (console) handler.
    logging.basicConfig(
        level=log_level,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up. All logs will be saved to '%s'.", log_file)
