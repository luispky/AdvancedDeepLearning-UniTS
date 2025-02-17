#!/usr/bin/env python3
import yaml
import logging
import torch
import numpy as np
from typing import Optional
from partA_src import (
    generate_test_loader,
    Model,
    standard_normal_init_fn,
    train_student, 
    log_evaluation_schedule,  
    get_global_params,
    get_layerwise_params, 
    compute_stats,
    compute_layerwise_stats,
    save_statistics, 
    plot_global_params_models
)
from utils import (
    save_model,
    evaluate_model, 
    set_seed, 
    plot_train_and_test_loss, 
    BASE_DIR, 
    LOGS_DIR,
    setup_logging, 
    EarlyStopping, 
    get_early_stopping_patience, 
    get_lr_scheduler_patience,
)

# =============================================================================
CONFIG_FILE = BASE_DIR / "config_partA.yml"

# =============================================================================
def main(
    seed: int,
    n_iterations: int,
    learning_rate: float,
    suffix: str,
    params_init_fn: Optional[str] = None,
    enable_lr_scheduler: bool = False,
    lr_factor: Optional[float] = None,
    lr_patience: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_threshold: Optional[float] = None,
):
    """
    Main function to train and evaluate the student models.
    
    Parameters:
        seed (int): Random seed for reproducibility.
        n_iterations (int): Number of training iterations.
        learning_rate (float): Initial learning rate.
        suffix (str): Suffix to append to the model filenames.
        params_init_fn (Optional[str]): Initialization function for the model parameters.
        enable_lr_scheduler (bool): Enable learning rate scheduler.
        lr_factor   (Optional[float]): Factor by which to reduce the learning rate.
        lr_patience (Optional[int]): Number of epochs with no improvement after which learning rate will be reduced.
        early_stopping_patience (Optional[int]): Number of epochs with no improvement after which training will be stopped.
        early_stopping_threshold (Optional[float]): Threshold for early stopping based on the validation loss.
    """
    # Set the random seed for reproducibility.
    set_seed(seed)
    
    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    batch_size = 128
    max_num_evaluations = max(1, int(n_iterations // 10))
    students_config = {
        "StudentUnderparam": [100, 10, 1],
        "StudentEqualparam": [100, 75, 50, 10, 1],
        "StudentOverparam": [100, 200, 200, 200, 100, 1],
    }
    
    # Teacher model
    teacher = Model(layers_units=[100, 75, 50, 10, 1], 
                    init_fn=standard_normal_init_fn).to(device)
    teacher.eval()  # Teacher is frozen
    logging.info("Teacher model created.")
    model_filename = f"teacher_model_s{seed}"
    save_model(teacher, model_filename)
    
    # Data Structures to store the global and layerwise parameters and statistics
    global_params = {}
    global_params["teacher"] = get_global_params(teacher)
    global_stats = {}
    layerwise_params = {}
    layerwise_params["teacher"] = get_layerwise_params(teacher)
    layerwise_stats = {}
    
    # Generate the test_dataset
    test_loader = generate_test_loader(teacher, input_dim=100)
    logging.info("Test dataset generated.")

    # Evaluation schedule, early stopping and learning rate scheduler
    eval_schedule = log_evaluation_schedule(n_iterations, max_num_evaluations=max_num_evaluations)
    early_stopping_patience = get_early_stopping_patience(total_epochs=n_iterations)
    
    # Train the student models
    for student_name, student_layer_units in students_config.items():
        logging.info(f"Training {student_name} model...")
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        student = Model(student_layer_units,
            init_fn=standard_normal_init_fn if params_init_fn == "standard_normal" else None
        ).to(device)
        model_filename = f"{student_name.lower()}_{suffix}"
        train_losses, test_losses = train_student(
            student,
            teacher,
            test_loader,
            batch_size=batch_size,
            learning_rate=learning_rate,
            eval_schedule=eval_schedule,
            device=device,
            enable_lr_scheduler=enable_lr_scheduler,
            lr_factor=lr_factor,
            lr_patience=lr_patience, 
            early_stopping_fn=early_stopping,
        )
        save_model(student, model_filename)
        
        # Evaluate the final model on the test set.
        final_test_loss = evaluate_model(student, test_loader, device=device)
        logging.info(f"Final Test Loss for {student_name} model: {final_test_loss:.6f}")
        
        # Plot and save the training and test loss curves.
        test_losses.append(final_test_loss)
        tmp = [eval_schedule[i] for i in range(len(test_losses)-1)]
        plot_train_and_test_loss(
            train_losses, 
            test_losses,
            title=f"Train and Test Loss for {student_name}",
            eval_schedule=tmp + [tmp[-1]+1],
            filename=f"train_test_loss_{model_filename}",
            save_fig=True, 
            use_log_scale=False,
        )
        
        # Compute the global and layerwise statistics
        global_params[student_name] = get_global_params(student)
        global_stats[student_name] = compute_stats(global_params["teacher"], global_params[student_name])
        logging.info("Global Statistics computed")
        
        layers_indices = None if student_name == "StudentEqualparam" else [0, -1]
        layerwise_params[student_name] = get_layerwise_params(student, layers_indices)
        layerwise_stats[student_name] = compute_layerwise_stats(layerwise_params["teacher"],
                                                                layerwise_params[student_name], 
                                                                )
        logging.info("Layerwise Statistics computed")
    
    # Save the statistics
    save_statistics(global_stats, layerwise_stats, suffix)
    
    # Plot the global parameters
    plot_global_params_models(global_params,
                            filename=f"global_params_{suffix}",
                            save_fig=True)


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
    n_iterations  = config.get("n_iterations", 100)
    learning_rate = config.get("learning_rate", 1e-2)
    params_init_fn = config.get("params_init_fn", None)
    enable_lr_scheduler = config.get("enable_lr_scheduler", False)
    lr_factor     = config.get("lr_factor", None)
    lr_patience   = config.get("lr_patience", None)
    early_stopping_patience = config.get("early_stopping_patience", None)
    early_stopping_threshold = config.get("early_stopping_threshold", None)
    
    suffix = (
        f"s{seed}"
        f"_niter{n_iterations}"
        f"_lr{learning_rate}"
        f"_{params_init_fn}"
        + (f"_lrs{enable_lr_scheduler}" if enable_lr_scheduler else "")
        + (f"_lrf{lr_factor}" if lr_factor is not None else "")
        + (f"_lrp{lr_patience}" if lr_patience is not None else "")
        + (f"_esp{early_stopping_patience}" if early_stopping_patience is not None else "")
        + (f"_est{early_stopping_threshold}" if early_stopping_threshold is not None else "")
    )

    log_file = LOGS_DIR / f"partA_{suffix}.log"

    if log_file.exists():
        print(f"Log file '{log_file}' already exists. Exiting...")
        exit(0)
        
    # Setup the logging configuration
    setup_logging(log_file=str(log_file), log_level=logging.INFO)
    logging.info("Configuration loaded from '%s'.", CONFIG_FILE)
    logging.info("Experiment parameters: %s", config)
    
    main(
        seed=seed,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        suffix=suffix,
        enable_lr_scheduler=enable_lr_scheduler,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        params_init_fn=params_init_fn,
    )
