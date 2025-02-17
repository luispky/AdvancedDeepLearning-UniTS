import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
from scipy.stats import ks_2samp
import seaborn as sns
import logging
from typing import List, Optional, Dict, Tuple

from utils import (
    evaluate_model, 
    RESULTS_DIR,
    FIGURES_DIR, 
    EarlyStopping
)

# Standard Normal initialization function for the weights and biases
def standard_normal_init_fn(layer):
    """
    Custom initialization: weights from Normal(0, 1), biases set to zero.
    
    Parameters:
        layer (nn.Module): Layer of a PyTorch model.
    """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        nn.init.normal_(layer.bias, mean=0.0, std=1.0)


class Model(nn.Module):
    """
    A simple feedforward neural network model.
    
    Parameters:
        layers_units (List[int]): Number of units in each layer.
        init_fn (Optional[Callable]): Initialization function for the weights and biases.
        
    Returns:
        torch.nn.Module: A PyTorch model.
    """
    def __init__(self, layers_units, init_fn=None):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create layers based on sizes
        for i in range(len(layers_units) - 1):
            layer = nn.Linear(layers_units[i], layers_units[i + 1])
            self.layers.append(layer)
        
        self.relu = nn.ReLU()
        
        # Apply the custom initialization if provided
        if init_fn is not None:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    init_fn(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation
        return x


# Generate test data
def generate_test_loader(teacher_model: nn.Module,
                        n_samples: int=60000,
                        input_dim: int=100,
                        batch_size: int=128
    )-> DataLoader:
    """
    Generate a DataLoader for the test dataset using the teacher model.
    
    Parameters:
        teacher_model (nn.Module): Teacher model to generate the target outputs.
        n_samples (int): Number of samples to generate.
        input_dim (int): Dimension of the input samples.
        batch_size (int): Batch size for the DataLoader.
    
    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    with torch.no_grad():
        x_test = torch.empty(n_samples, input_dim).uniform_(0, 2)
        y_test = teacher_model(x_test)
        # float32: 4 bytes
        # x_test memory = 60000 * 100 * 4 bytes = 24 MB
        # y_test memory = 60000 * 1 * 4 bytes = 0.24 MB
    # Wrap the dataset in a TensorDataset
    test_dataset = TensorDataset(x_test, y_test)
    # Create a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def log_evaluation_schedule(n_iterations: int, max_num_evaluations: int) -> List[int]:
    """
    Generate a logarithmically spaced evaluation schedule over the range of iterations.
    
    This function returns a sorted list of unique iteration numbers where evaluations should occur.
    Evaluations are denser at the beginning of training and sparser towards the end, which is often
    desirable when early training dynamics are critical.
    
    The evaluation points are generated using NumPy's geomspace, which creates values spaced evenly
    on a log scale. These float values are then rounded to the nearest integer, and duplicate values 
    are removed.
    
    Parameters:
        n_iterations (int): Total number of iterations (must be a positive integer).
        max_num_evaluations (int): Desired number of evaluation points (must be a positive integer).
    
    Returns:
        List[int]: Sorted list of evaluation iteration numbers.
    
    Raises:
        ValueError: If n_iterations or max_num_evaluations is less than 1.
    """
    # Input validation: ensure both n_iterations and max_num_evaluations are positive.
    if n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer.")
    if max_num_evaluations < 1:
        raise ValueError("max_num_evaluations must be a positive integer.")
    
    # Use np.geomspace to generate geometrically spaced values between 1 and n_iterations.
    # Note: The endpoints are included (from 1 to n_iterations).
    eval_schedule = np.geomspace(1, n_iterations, num=max_num_evaluations, endpoint=True)
    
    # Convert the generated floats to the nearest integers.
    # np.rint rounds to the nearest integer, and np.unique removes duplicates (which can happen due to rounding).
    eval_schedule_int = np.unique(np.rint(eval_schedule).astype(int))
    
    # Return the schedule as a sorted list (np.geomspace generates sorted values, but sorting ensures order).
    return eval_schedule_int.tolist()


# Training student models
def train_student(student_model,
                teacher_model,
                test_loader,
                batch_size=128,
                learning_rate=0.001,
                eval_schedule=[],
                device="cpu", 
                enable_lr_scheduler: bool = False,
                lr_factor: Optional[float] = None,
                lr_patience: Optional[int] = None,
                early_stopping_fn: Optional[EarlyStopping] = None, 
    ) -> Tuple[List[float], List[float]]:
    """
    Train a student model using the teacher model and evaluate it based on the evaluation schedule.
    
    Parameters:
        student_model (nn.Module): Student model to train.
        teacher_model (nn.Module): Teacher model to generate the target outputs.
        test_loader (DataLoader): DataLoader for the test dataset.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimization.
        eval_schedule (List[int]): List of iteration numbers for evaluation.
        device (str): Device to use for training and evaluation.
        enable_lr_scheduler (bool): Whether to enable the learning rate scheduler.
        lr_factor (Optional[float]): Factor by which to reduce the learning rate.
        lr_patience (Optional[int]): Number of epochs with no improvement after which learning rate will be reduced.
        early_stopping_fn (Optional[EarlyStopping]): If provided, early stopping will be applied.
        
    Returns:
        Tuple[List[float], List[float]]: Training and test losses.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # Initialize learning rate scheduler
    if enable_lr_scheduler and lr_factor and lr_patience:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=lr_factor,
                                                        patience=lr_patience,
                                                        )
    else:
        enable_lr_scheduler = False
    
    train_losses, test_losses = [], []
    
    student_model.train()
            
    n_iterations = max(eval_schedule)
    start_time = time.time()

    # Training loop with dynamic evaluation
    for iteration in range(1, n_iterations + 1):
        # Generate fresh batch
        inputs = torch.empty(batch_size, 100).uniform_(0, 2).to(device)
        
        targets = teacher_model(inputs).detach().to(device)
        
        outputs = student_model(inputs)
        train_loss = criterion(outputs, targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # Perform evaluation based on the schedule
        if iteration in eval_schedule:
            test_loss = evaluate_model(student_model, test_loader, device)
            test_losses.append(test_loss)

            # Step the scheduler based on the train loss.
            if enable_lr_scheduler:
                scheduler.step(test_loss)
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr <= 1e-5:  
                    logging.info(f"Learning rate reached minimum value. Stopping training.")
                    break
            
            logging.info(f"Iteration {iteration}, Train Loss: {train_loss.item():.4f},\
                    Test Loss: {test_loss:.4f}" + 
                    (f", LR: {current_lr:.6f}" if enable_lr_scheduler else ""))
            
            # Early Stopping Check
            if early_stopping_fn and early_stopping_fn(test_loss, student_model, iteration):
                logging.info(f"Early stopping triggered at Iteration {iteration}. Best Test Loss: {early_stopping_fn.best_loss:.4f}")
                break  # Stop training
    
    if enable_lr_scheduler:
        final_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Final Learning Rate: {final_lr}")
    
    # Restore best model if early stopping was used
    if early_stopping_fn:
        early_stopping_fn.restore_best_model(student_model)

    # End time tracking
    elapsed_time = (time.time() - start_time) / 60
    logging.info(f"Time taken to train student model for {iteration} iterations: {elapsed_time:.2f} minutes")
    
    return train_losses, test_losses


def distributions_statistical_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
) -> Tuple[float, float]:
    """
    
    Perform a statistical test to compare two distributions.
    
    Kolmogorov-Smirnov (KS) Test
    The two-sample KS test compares the cumulative distribution functions (CDFs)
    of two samples. The null hypothesis is that the two samples are drawn from
    the same distribution.
    
    If the p-value is less than a chosen alpha level (typically 0.05), then the null
    hypothesis is rejected, and the two samples are considered to be drawn from
    different distributions.
    
    Parameters:
        sample1 (np.ndarray): First sample for comparison.
        sample2 (np.ndarray): Second sample for comparison.
    
    Returns:
        Tuple[float, float]: KS statistic and p-value.
    """
    
    statistic, p_value = ks_2samp(sample1, sample2)
    
    return statistic, p_value


def get_global_params(model: torch.nn.Module) -> np.ndarray:
    """
    Extract the global parameters of a PyTorch model.
    
    Parameters:
    - model: PyTorch model.
    
    Returns:
    - np.ndarray: Flattened array of model parameters.
    """
    params = []
    for param in model.parameters():
        params.extend(param.detach().cpu().numpy().flatten())
    return np.array(params)


def get_layerwise_params(
    model: torch.nn.Module,
    layers_indices: Optional[List[int]] = None,
):
    """
    Extract the parameters of the specified layers in the model and returns them
    as dictionary with layer index as key and parameters as value.
    If layers_indices is None, it will extract the parameters of all layers.
    """
    layer_params = {}
    
    if not layers_indices:
        layers_indices = [idx for idx in range(len(model.layers))]
        layers_indices[-1] = -1  
    
    selected_layers = [model.layers[idx] for idx in layers_indices]
    for layer, idx in zip(selected_layers, layers_indices):
        params = []
        for param in layer.parameters():
            params.extend(param.detach().cpu().numpy().flatten())
        layer_params[idx] = np.array(params)
    return layer_params


def compute_stats(
    teacher_params: np.ndarray,
    student_params: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute the mean and standard deviation of the student parameters,
    and perform a statistical test to compare the distributions of teacher and student parameters.
    
    Parameters:
    - teacher_params: Flattened array of teacher model parameters.
    - student_params: Flattened array of student model parameters.
    
    Returns:
    - Tuple[float, float, float, float]: Mean and standard deviation of student parameters,
        KS statistic, and p-value from the statistical test.
    """
    mean_student_params = np.mean(student_params)
    std_student_params = np.std(student_params)
    
    statistic, p_value = distributions_statistical_test(teacher_params, student_params)
    
    return mean_student_params, std_student_params, statistic, p_value


def compute_layerwise_stats(
    teacher_layerwise_params: Dict,
    student_layerwise_params: Dict,
) -> Dict:
    """
    Compute the mean and standard deviation of the student parameters for each layer,
    and perform a statistical test to compare the distributions of teacher and student parameters for each layer.
    
    Parameters:`
    - teacher_layerwise_params: Dictionary of teacher model layerwise parameters.
    - student_layerwise_params: Dictionary of student model layerwise parameters.
    
    Returns:
    - Dict: Dictionary containing the mean and standard deviation of student parameters,
        KS statistic, and p-value for each layer.
    """
    layerwise_stats = {}
    
    for layer, student_params in student_layerwise_params.items():
        layerwise_stats[layer] = compute_stats(teacher_layerwise_params[layer],
                                                student_params)
        
    return layerwise_stats


def save_statistics(global_stats: Dict,
                    layerwise_stats: Dict,
                    filename=None) -> None:
    """
    Save the global and layerwise statistics to CSV files.
    
    Parameters:
    - global_stats: Dictionary containing the global statistics.
    - layerwise_stats: Dictionary containing the layerwise statistics.
    - filename: Filename to save the statistics.
    """
    if filename is None:
        raise ValueError("Please provide a filename to save the statistics.")
    
    global_stats_df = pd.DataFrame.from_dict(global_stats, orient="index", columns=["mean", "std", "statistic", "pvalue"])
    global_stats_df.reset_index(inplace=True)
    global_stats_df.rename(columns={"index": "model"}, inplace=True)
    global_stats_df.to_csv(RESULTS_DIR / f"{filename}_global_stats.csv", index=False)
    
    # Flatten the layerwise statistics
    data = []
    for model_name, layers in layerwise_stats.items():
        for layer_name, values in layers.items():
            mean, std, statistic, pvalue = values
            data.append([model_name, layer_name, mean, std, statistic, pvalue])

    layerwise_stats_df = pd.DataFrame(data, columns=["model", "layer", "mean", "std", "statistic", "pvalue"])
    layerwise_stats_df.to_csv(RESULTS_DIR / f"{filename}_layerwise_stats.csv", index=False)


def plot_global_params_models(
    global_params: Dict[str, np.ndarray], 
    title: str = "Histogram of Model Parameters", 
    save_fig: bool = False,
    filename: Optional[str] = None
) -> None:
    """
    Plots a histogram comparing the distribution of parameters across multiple models.

    Parameters:
    - global_params: Dictionary where keys are model names and values are flattened parameter arrays.
    - title: Title of the plot.
    - save_fig: Whether to save the figure.
    - filename: Filename to save the figure.
    """

    plt.figure(figsize=(8, 5))

    # Plot histograms for each model
    for model_name, params in reversed(global_params.items()):
        sns.histplot(params, label=model_name, alpha=0.3, kde=True,
                    line_kws={"linewidth": 3,
                            "linestyle": "--" if "teacher" in model_name else "-"})

    # Configure the plot
    plt.xlabel("Parameter Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend(title="Models")

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig:
        if filename is None:
            filename = "global_params_histogram"
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {filepath}")
        plt.close()  # Close the plot to avoid displaying in non-interactive environments
    else:
        plt.show()