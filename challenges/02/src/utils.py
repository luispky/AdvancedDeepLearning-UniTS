# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
import seaborn as sns
import pandas as pd
import sys

# Base directory for utils.py
BASE_DIR = Path(__file__).resolve().parent

# Paths for saving results
MODELS_DIR = BASE_DIR.parent / "models"
RESULTS_DIR = BASE_DIR.parent / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_model(model, filename):
    filepath = MODELS_DIR / f"{filename}.pt"
    torch.save(model.state_dict(), filepath)
    print(f"Model saved at {filepath}")

def plot_losses(student_metrics, eval_schedule, filename=None):
    if eval_schedule is None:
        raise ValueError("Please provide the evaluation schedule.")
    if filename is None:
        raise ValueError("Please provide a filename to save the plot.")

    fig, ax = plt.subplots(1, len(student_metrics), figsize=(15, 5))
    for i, (student_name, (train_losses, test_losses, _)) in enumerate(student_metrics.items()):
        # Ensure the right subplot is selected
        ax[i].plot(range(len(train_losses)), train_losses, label="Train Loss", color="blue", linewidth=2)
        
        # Scatter plot of test losses with eval_schedule as X-coordinates
        ax[i].plot(eval_schedule, test_losses, label="Test Loss", color="red", linewidth=2, linestyle="--")#, marker="o", markersize=4)
        
        ax[i].set_title(f"{student_name} Loss", fontsize=14)
        ax[i].set_xlabel("Iteration", fontsize=12)
        ax[i].set_ylabel("Loss", fontsize=12)
        ax[i].legend(fontsize=10)
        ax[i].grid(True, linestyle="--", alpha=0.5)  # Add grid for better readability

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{filename}.png", bbox_inches="tight")

    
def save_metrics(student_metrics, filename):
    filepath = RESULTS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(student_metrics, f)
    print(f"Metrics saved at {filepath}")

def load_metrics(filename):
    with open(RESULTS_DIR / f"{filename}.pkl", "rb") as f:
        return pickle.load(f) 

def compute_layerwise_stats(model):
    layerwise_stats = {}
    layer_counter = 1  # To normalize unnamed layers as fc1, fc2, etc.
    
    for name, param in model.named_parameters():
        # Extract the top-level layer name
        layer_name = name.split(".")[0]
        
        # Handle cases where layer names are numbers (e.g., "0", "1")
        if layer_name == "layers":  # For models using ModuleList
            sub_layer_name = name.split(".")[1]  # Extract the index within "layers"
            layer_name = f"fc{int(sub_layer_name) + 1}"  # Convert to fc1, fc2, etc.
        elif layer_name.isdigit():  # For other unnamed layers
            layer_name = f"fc{layer_counter}"
            layer_counter += 1
        
        # Initialize stats dictionary for the layer if not already present
        if layer_name not in layerwise_stats:
            layerwise_stats[layer_name] = {
                "mean_weights": None,
                "std_weights": None,
                "mean_biases": None,
                "std_biases": None
            }
        
        # Compute stats for weights
        if "weight" in name:
            layerwise_stats[layer_name]["mean_weights"] = param.data.mean().item()
            layerwise_stats[layer_name]["std_weights"] = param.data.std().item()
        
        # Compute stats for biases
        elif "bias" in name:
            if param.data.numel() > 1:  # If there are multiple biases
                layerwise_stats[layer_name]["mean_biases"] = param.data.mean().item()
                layerwise_stats[layer_name]["std_biases"] = param.data.std().item()
            else:  # If there is a single bias
                layerwise_stats[layer_name]["mean_biases"] = param.data.item()
                layerwise_stats[layer_name]["std_biases"] = 0.0
    
    return layerwise_stats

def compute_layerwise_stats_models(models):
    layerwise_stats = {}
    for model_name, model in models.items():
        layerwise_stats[model_name] = compute_layerwise_stats(model)
    return layerwise_stats

def plot_params_layerwise_stats_models(models_stats, filename=None):
    if filename is None:
        raise ValueError("Please provide a filename to save the plot.")
    
    # Prepare data
    models = list(models_stats.keys())
    layers = sorted(set(layer for model in models_stats.values() for layer in model))

    # Plot configuration
    x = np.arange(len(layers))  # Layer positions
    bar_width = 0.2
    colors = ['blue', 'orange', 'green', 'red']
    markers = ['o', 's', '^', 'D']

    # Weights plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        means = []
        stds = []
        valid_layers = []  # Keep track of valid layers with data
        for layer in layers:
            if layer in models_stats[model]:
                means.append(models_stats[model][layer]['mean_weights'])
                stds.append(models_stats[model][layer]['std_weights'])
                valid_layers.append(layer)

        # Only plot if there is valid data
        if means:
            x_offset = x[:len(valid_layers)] + i * bar_width
            ax1.errorbar(x_offset, means, yerr=stds, fmt='none', ecolor=colors[i], capsize=5)
            ax1.scatter(x_offset, means, color=colors[i], marker=markers[i], s=60, label=model)

    # Add vertical lines between x-ticks
    for tick in x[:-1]:  # Skip the last tick
        ax1.axvline(tick + 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=3)

    ax1.set_title('Weights: Mean and Standard Deviation')
    ax1.set_xlabel('Layers')
    ax1.set_ylabel('Mean ± Std')
    ax1.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Biases plot
    fig, ax2 = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        means = []
        stds = []
        valid_layers = []  # Keep track of valid layers with data
        for layer in layers:
            if layer in models_stats[model]:
                means.append(models_stats[model][layer]['mean_biases'])
                stds.append(models_stats[model][layer]['std_biases'])
                valid_layers.append(layer)

        # Only plot if there is valid data
        if means:
            x_offset = x[:len(valid_layers)] + i * bar_width
            ax2.errorbar(x_offset, means, yerr=stds, fmt='none', ecolor=colors[i], capsize=5)
            ax2.scatter(x_offset, means, color=colors[i], marker=markers[i], s=60, label=model)

    # Add vertical lines between x-ticks
    for tick in x[:-1]:  # Skip the last tick
        ax2.axvline(tick + 0.5, color='gray', linestyle='--', alpha=0.7, linewidth=3)

    ax2.set_title('Biases: Mean and Standard Deviation')
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Mean ± Std')
    ax2.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax2.set_xticklabels(layers)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filepath = RESULTS_DIR / f"{filename}.png"
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Layerwise stats plot saved at {filepath}")
    
def save_layerwise_stats_models(models_stats, filename):
    filepath = RESULTS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(models_stats, f)
    print(f"Layerwise stats saved at {filepath}")

def extract_params_global(model):
    weights_and_biases = {"weights": [], "biases": []}
    
    # Extract weights and biases from each layer
    for name, param in model.named_parameters():
        if "weight" in name:
            weights_and_biases["weights"].extend(param.data.cpu().numpy().flatten())
        elif "bias" in name:
            weights_and_biases["biases"].extend(param.data.cpu().numpy().flatten())
            
    return weights_and_biases

def extract_global_params_models(models):
    
    params_models = {}
    for model_name, model in models.items():
        params_models[model_name] = extract_params_global(model)
    return params_models

def plot_global_params_models(params_models, filename=None):
    if filename is None:
        raise ValueError("Please provide a filename to save the plot.")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot KDE for weights
    for model_name, params in params_models.items():
        sns.kdeplot(params["weights"], fill=True, label=f"{model_name}", alpha=0.3, ax=axs[0])

    # Plot KDE for biases
    for model_name, params in params_models.items():
        bias_variance = np.var(params["biases"])
        if bias_variance > 0:
            sns.kdeplot(params["biases"], fill=True, label=f"{model_name}", alpha=0.3, ax=axs[1])
        else:
            sns.histplot(params["biases"], label=f"{model_name}", alpha=0.3, kde=False, ax=axs[1], stat="density")

    axs[0].set_title("Weights Distributions")
    axs[1].set_title("Biases Distributions")
    axs[0].set_xlabel("Weight Value")
    axs[1].set_xlabel("Bias Value")
    axs[0].legend()
    axs[1].legend()
    filepath = RESULTS_DIR / f"{filename}.png"
    plt.savefig(filepath, bbox_inches="tight")
    
def save_global_params_models(params_models, filename):
    filepath = RESULTS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(params_models, f)
    print(f"Global params saved at {filepath}")
        
def compute_global_stats_models(global_params_models):
    global_stats = {}
    
    for model_name, params in global_params_models.items():
        global_stats[model_name] = {
            "mean_weights": np.mean(params["weights"]),
            "std_weights": np.std(params["weights"]),
            "mean_biases": np.mean(params["biases"]),
            "std_biases": np.std(params["biases"])
        }
        
    return global_stats

def save_global_stats_models(global_stats, filename):
    filepath = RESULTS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(global_stats, f)
    print(f"Global stats saved at {filepath}")

def plot_global_stats_models(global_stats, filename=None):
    if filename is None:
        raise ValueError("Please provide a filename to save the plot.")
    
    # Extract data from global stats
    models = list(global_stats.keys())
    mean_weights = [global_stats[model]["mean_weights"] for model in models]
    std_weights = [global_stats[model]["std_weights"] for model in models]
    mean_biases = [global_stats[model]["mean_biases"] for model in models]
    std_biases = [global_stats[model]["std_biases"] for model in models]

    x = np.arange(len(models))  # X-axis positions for models
    bar_width = 0.4
    colors = ['blue', 'orange', 'green', 'red']  # Colors for each model
    markers = ['o', 's', '^', 'D']  # Different markers for each model

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1x2 layout

    # Plot for weights
    ax1 = axes[0]
    for i, (mean, std) in enumerate(zip(mean_weights, std_weights)):
        ax1.errorbar(x[i], mean, yerr=std, fmt='none', ecolor=colors[i], capsize=5)
        ax1.scatter(x[i], mean, color=colors[i], marker=markers[i], s=60, label=models[i])  # Add mean points

    ax1.set_title("Global Statistics: Weights")
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Mean ± Std")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot for biases
    ax2 = axes[1]
    for i, (mean, std) in enumerate(zip(mean_biases, std_biases)):
        ax2.errorbar(x[i], mean, yerr=std, fmt='none', ecolor=colors[i], capsize=5)
        ax2.scatter(x[i], mean, color=colors[i], marker=markers[i], s=60, label=models[i])  # Add mean points

    ax2.set_title("Global Statistics: Biases")
    ax2.set_xlabel("Models")
    ax2.set_ylabel("Mean ± Std")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add legend to the first plot
    ax1.legend(loc="upper left")

    plt.tight_layout()
    filepath = RESULTS_DIR / f"{filename}.png"
    plt.savefig(filepath, bbox_inches="tight")
    print(f"Global stats plot saved at {filepath}")

# Function to compute absolute percentage error
def absolute_percentage_error(true_value, approx_value):
    return abs((true_value - approx_value) / true_value) * 100 if true_value != 0 else float('inf')

def compute_and_save_students_errors(models_global_stats, filename=None):
    if filename is None:
        raise ValueError("Please provide a filename to save the results.")
    
    # Teacher data for reference
    teacher = models_global_stats['Teacher']

    # Compute percentage errors for Students
    errors = {}
    for student, stats in models_global_stats.items():
        if student == 'Teacher':  # Skip the Teacher
            continue
        
        errors[student] = {
            'mean_weights_error': absolute_percentage_error(teacher['mean_weights'], stats['mean_weights']),
            'std_weights_error': absolute_percentage_error(teacher['std_weights'], stats['std_weights']),
            'mean_biases_error': absolute_percentage_error(teacher['mean_biases'], stats['mean_biases']),
            'std_biases_error': absolute_percentage_error(teacher['std_biases'], stats['std_biases'])
        }

    # Display the results
    df = pd.DataFrame(errors).T  # Convert to a DataFrame for better visualization
    df.columns = ['Mean Weights % Error', 'Std Weights % Error', 'Mean Biases % Error', 'Std Biases % Error']

    # Save the results
    filepath = RESULTS_DIR / f"{filename}.csv"
    df.to_csv(filepath)
    print(f"Errors saved at {filepath}")

def compute_and_save_stats_teacher_student(models, n_iterations, learning_rate):
    
    # Extract weights and biases statistics per layer for each model
    models_layerwise_stats = compute_layerwise_stats_models(models)
    
    # Plot the statistics
    plot_params_layerwise_stats_models(models_layerwise_stats, f"layerwise_stats_iter{n_iterations}_lr{learning_rate}")
    
    # Save layer-wise statistics for each model
    save_layerwise_stats_models(models_layerwise_stats, f"layerwise_stats_iter{n_iterations}_lr{learning_rate}")
    
    # Extract all the params from the models
    models_params = extract_global_params_models(models)
    
    # Save the params for each model
    save_global_params_models(models_params, f"params_iter{n_iterations}_lr{learning_rate}")
    
    # Plot the params for each model
    plot_global_params_models(models_params, f"params_iter{n_iterations}_lr{learning_rate}")
    
    # Compute global statistics for each model
    models_global_stats = compute_global_stats_models(models_params)
    
    # Save the global statistics for each model
    save_global_stats_models(models_global_stats, f"global_stats_iter{n_iterations}_lr{learning_rate}")
    
    # Plot the global statistics for each model
    plot_global_stats_models(models_global_stats, f"global_stats_iter{n_iterations}_lr{learning_rate}")
    
    # Compute and save the errors for each student model
    compute_and_save_students_errors(models_global_stats, f"errors_iter{n_iterations}_lr{learning_rate}")