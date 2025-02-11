#!/usr/bin/env python3
import yaml
import logging
import torch
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
    setup_logging
)

import sys


# =============================================================================
CONFIG_FILE = BASE_DIR / "config_partA.yml"

# =============================================================================
def main(
    seed: int,
    n_iterations: int,
    learning_rate: float,
    lr_factor: float,
    lr_patience: int,
    enable_lr_scheduler: bool = False, 
    suffix: str = ""
):
    """
    Main function to train and evaluate the student models.
    
    Parameters:
        seed (int): Random seed for reproducibility.
        n_iterations (int): Number of training iterations.
        learning_rate (float): Initial learning rate.
        lr_factor (float): Factor to reduce learning rate.
        lr_patience (int): Number of epochs to wait before reducing learning rate.
        enable_lr_scheduler (bool): Enable learning rate scheduler.
        suffix (str): Suffix to append to the model filenames.
    """
    # Set the random seed for reproducibility.
    set_seed(seed)
    
    # Parameters
    max_num_evaluations = n_iterations // 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    test_loader = generate_test_loader(teacher)
    logging.info("Test dataset generated.")

    # Evaluation schedule
    eval_schedule = log_evaluation_schedule(n_iterations, max_num_evaluations=max_num_evaluations)
    
    # Train the student models
    for student_name, student_layer_units in students_config.items():
        logging.info(f"Training {student_name} model...")
        student = Model(student_layer_units, init_fn=standard_normal_init_fn).to(device)
        model_filename = f"{student_name.lower()}_{suffix}"
        train_losses, test_losses = train_student(
            student,
            teacher,
            test_loader,
            learning_rate=learning_rate,
            eval_schedule=eval_schedule,
            device=device,
            enable_lr_scheduler=enable_lr_scheduler,
            lr_factor=lr_factor,
            lr_patience=lr_patience, 
        )
        save_model(student, model_filename)
        
        # Evaluate the final model on the test set.
        final_test_loss = evaluate_model(student, test_loader, device=device)
        logging.info(f"Final Test Loss for {student_name} model: {final_test_loss:.6f}")
        
        # Plot and save the training and test loss curves.
        test_losses.append(final_test_loss)
        plot_train_and_test_loss(
            train_losses, 
            test_losses,
            title=f"Train and Test Loss for {student_name}",
            eval_schedule=eval_schedule + [n_iterations+1],
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
    enable_lr_scheduler = config.get("enable_lr_scheduler", False)
    lr_factor     = config.get("lr_factor", 0.5)
    lr_patience   = config.get("lr_patience", 3)
    
    suffix = f"s{seed}_niter{n_iterations}_lr{learning_rate}" \
                f"_lrs{enable_lr_scheduler}_lrf{lr_factor}_lrp{lr_patience}"

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
        enable_lr_scheduler=enable_lr_scheduler,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        suffix=suffix
    )
    