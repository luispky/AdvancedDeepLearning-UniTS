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
    Plots training and testing loss over epochs or iterations, switching to log scale if needed.
    
    Parameters:
    - train_losses (List[float]): Training loss values per epoch.
    - test_losses (List[float]): Testing loss values.
    - title (str): Title for the plot.
    - eval_schedule (Optional[Sequence]): Epoch indices corresponding to test loss evaluations.
    - save_fig (bool): Whether to save the plot as an image.
    - filename (Optional[str]): Filename for saving.
    - use_log_scale (bool): Whether to force log scale (if False, log scale is chosen automatically).
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

    # Compute loss range for log scale decision
    all_losses = np.array(train_losses + test_losses)
    min_loss = np.min(all_losses[np.nonzero(all_losses)])  # Avoid zero values
    max_loss = np.max(all_losses)
    loss_ratio = max_loss / min_loss if min_loss > 0 else np.inf

    # Automatically use log scale if ratio is too high
    auto_log_scale = loss_ratio > 1e2
    if auto_log_scale:
        use_log_scale = True  # Override to force log scale

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

    # Option 1: Use scientific notation for better readability
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-1, 2))  # Use scientific notation for values below 1e-1 or above 1e2
    ax.yaxis.set_major_formatter(formatter)

    # Option 2: Use logarithmic scale if required
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


class EarlyStopping:
    def __init__(self, patience=10, threshold=0.001):
        """
        Implements early stopping to prevent overfitting.

        :param patience: Number of iterations without improvement before stopping.
        :param threshold: Minimum change in test loss to be considered an improvement.
        """
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.best_iteration = 0  # Track when the best model was found

    def __call__(self, val_loss, model, iteration):
        """
        Checks if training should stop.

        :param val_loss: Current validation/test loss.
        :param model: The current model.
        :param iteration: The current iteration number.
        :return: True if early stopping should be triggered, else False.
        """
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Save the best model
            self.best_iteration = iteration  # Store best iteration
        else:
            self.counter += 1

        return self.counter >= self.patience  # Stop if patience limit is reached

    def restore_best_model(self, model):
        """
        Restores the model to the best recorded state.
        """
        if self.best_model_state and self.counter >= self.patience:
            model.load_state_dict(self.best_model_state)
            logging.info(f"Restored best model (Iteration {self.best_iteration}) with Test Loss: {self.best_loss:.4f}")


def get_early_stopping_patience(total_epochs: int) -> int:
    """
    Given the total number of training epochs, computes a reasonable early stopping patience.
    
        total_epochs (int): The total number of training epochs.
        
    Returns:
        int: The early stopping patience (number of evaluations to wait for improvement).
    """
    # log_total = np.log1\0(total_epochs)
    # patience = np.pow(10, 0.333 * log_total + 0.334)
    # return max(1, int(round(patience)))
    return max(1, int(np.ceil(0.25*total_epochs*1e-1)))


def get_lr_scheduler_patience(total_epochs: int) -> int:
    """
    Given the total number of training epochs, computes a reasonable learning rate scheduler patience.
    
    Based on the desired values:
        - 100 epochs  -> 2
        - 1,000 epochs -> 10
        - 10,000 epochs -> 50
        - 100,000 epochs -> 500
    
    We use a log-linear relationship:
        lr_scheduler_patience = 10^(0.8 * log10(total_epochs) - 1.3)
    
    Parameters:
        total_epochs (int): The total number of training epochs.
        
    Returns:
        int: The learning rate scheduler patience.
    """
    log_total = np.log10(total_epochs)
    patience = np.pow(10, 0.8 * log_total - 1.3)
    return max(1, int(round(patience)))
