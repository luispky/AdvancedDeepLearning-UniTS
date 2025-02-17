#!/usr/bin/env python3
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from partB_src import (
    generate_non_hierarchical_B6_polynomial,
    B6,
    ResidualNetwork,
    generate_data_loaders,
    train_model,
    evaluate_model,
    aggregated_variable_sweep, 
    plot_variable_sweep_results
)
from utils import (
    set_seed,
    plot_train_and_test_loss, 
    save_model, 
    load_model,
    BASE_DIR, 
    LOGS_DIR, 
    MODELS_DIR, 
    setup_logging, 
)


# =============================================================================
CONFIG_FILE = BASE_DIR / "config_partB.yml"

# =============================================================================
def main(
        seed: int,
        grid_points: int,
        num_trials: int,
        num_epochs: int, 
        learning_rate: float, 
        suffix: str,
        ) -> None:
    """
    Main function to perform the following tasks:
        1. Generate the non-hierarchical polynomial tildeB6.
        2. Prepare training and test datasets for both B6 and tildeB6.
        3. Instantiate and train a Residual Network model on each dataset.
        4. Save models and loss curves using a unified suffix.
        5. Evaluate each model on the test set.
        6. Conduct an aggregated variable sweep analysis.
        7. Plot the aggregated analysis results.

    Parameters:
        seed        (int): Random seed for reproducibility.
        grid_points (int): Number of grid points for the variable sweep analysis.
        num_trials  (int): Number of random input trials for aggregation.
        num_epochs  (int): Number of epochs for training.
        learning_rate (float): Initial learning rate.
        suffix      (str): A suffix string computed from key parameters. If not provided, it is computed internally.
    """
    # Set the random seed for reproducibility.
    set_seed(seed)
    
    # Set device (use GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    # -------------------------------------------------------------------------
    # Generate the non-hierarchical polynomial tildeB6.
    # -------------------------------------------------------------------------
    B6_tilde, _ = generate_non_hierarchical_B6_polynomial(seed=seed, symbolic=True)
    logging.info("Non-hierarchical polynomial tildeB6 generated.")
    
    # -------------------------------------------------------------------------
    # Data preparation setting and parameters.
    # -------------------------------------------------------------------------
    num_train = int(1e5)
    num_test  = int(6e4)
    batch_size = 20
    
    # Create a grid over [0,2] for variable sweep analysis.
    grid = np.linspace(0, 2, grid_points)
    
    # Define the two polynomials to be used.
    polynomials: Dict[str, Any] = {
        "B6": B6,
        "B6 tilde": B6_tilde,
    }
    
    # Dictionaries to store trained models and aggregated sweep results.
    models: Dict[str, nn.Module] = {}
    
    # -------------------------------------------------------------------------
    # Loop over the two polynomials (B6 and B6 tilde).
    # -------------------------------------------------------------------------
    for poly_name, polynomial in polynomials.items():
        logging.info("Processing %s polynomial", poly_name)
        
        # Generate training and test DataLoaders.
        train_loader, test_loader = generate_data_loaders(
            func=polynomial,
            num_train=num_train,
            num_test=num_test,
            batch_size=batch_size,
        )
        
        # Construct model filename and check if model already exists.
        model_filename = f"model_{poly_name}_{suffix}"
        model_filepath = MODELS_DIR / f"{model_filename}.safetensors"
        
        if model_filepath.exists():
            logging.info(f"Pre-trained model for {poly_name} found.")
            model = ResidualNetwork(input_dim=6, hidden_dim=50, output_dim=1, num_hidden=8)
            model = load_model(model, model_filename)
            model.to(device)
        else:
            # Instantiate a new residual network model.
            model = ResidualNetwork(input_dim=6, hidden_dim=50, output_dim=1, num_hidden=8)
            model.to(device)
            
            logging.info(f"Training model for {poly_name} polynomial...")
            # Train the model and obtain training and test loss curves.
            train_losses, test_losses = train_model(
                model, 
                train_loader, 
                test_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
            )
            
            # Save the trained model with a filename that includes the suffix.
            save_model(model, model_filename)
            logging.info(f"Model for {poly_name} polynomial saved.")
            
            # Plot and save the training and test loss curves.
            plot_train_and_test_loss(
                train_losses, test_losses, 
                title=f"Train and Test Loss for {poly_name}",
                filename=f"train_test_loss_{poly_name}_{suffix}",
                save_fig=True
            )
            logging.info(f"Loss curves for {poly_name} polynomial plotted and saved.")
        
        # Store the trained model.
        models[poly_name] = model
        
        # Evaluate the final model on the test set.
        final_test_loss = evaluate_model(model, test_loader, device=device)
        logging.info(f"Final Test Loss for {poly_name} polynomial: {final_test_loss:.6f}")
        
        # Perform aggregated variable sweep analysis.
        logging.info(f"Performing aggregated variable sweep analysis ({num_trials} trials) for {poly_name} polynomial...")
        aggregated_results = aggregated_variable_sweep(
            model,
            target_function=polynomial,
            grid=grid,
            num_trials=num_trials,
            device=device
        )
    
        # -------------------------------------------------------------------------
        # Plot aggregated variable sweep results.
        # -------------------------------------------------------------------------
        plot_variable_sweep_results(
            aggregated_results=aggregated_results,
            grid=grid,
            poly_name=poly_name,
            title="Aggregated Variable Sweep Analysis",
            save_fig=True,
            filename=f"aggregated_variable_sweep_{poly_name}_{suffix}"
        )
        
    logging.info("All processing completed.")

# =============================================================================
if __name__ == "__main__":
    # Load Configuration from the Global CONFIG_FILE.
    try:
        with open(CONFIG_FILE, "r") as config_file:
            config = yaml.safe_load(config_file)
    except Exception as e:
        print(f"Error loading configuration file '{CONFIG_FILE}': {e}")
        raise
    
    # Extract Parameters from the Configuration.
    seed          = config.get("seed", 42)
    grid_points   = config.get("grid_points", 100)
    num_trials    = config.get("num_trials", 5)
    num_epochs    = config.get("num_epochs", 30)
    learning_rate = config.get("learning_rate", 1e-3)
    
    suffix = (
        f"s{seed}"
        f"_lr{learning_rate}"
        f"_ntr{num_trials}"
        f"_ep{num_epochs}"
    )
    
    # Create the Log File Name Based on the Suffix.
    log_file = LOGS_DIR / f"partB_{suffix}.log" 
    
    # If the file already exists, stop to avoid doing unnecessary work.
    if log_file.exists():
        print(f"Log file '{log_file}' already exists. Exiting...")
        exit(0)
    
    # Set up logging.
    setup_logging(log_file=str(log_file), log_level=logging.INFO)
    logging.info("Configuration loaded from '%s'.", CONFIG_FILE)
    logging.info("Experiment parameters: %s", config)
    
    main(
        seed=seed,
        grid_points=grid_points,
        num_trials=num_trials,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        suffix=suffix,
    )
