import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from functools import partial
from pathlib import Path

# Add src directory to Python path (assumes utils are in ../src)
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils import save_model, plot_losses, save_metrics
from utils import compute_and_save_stats_teacher_student

# Teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(100, 75)
        self.fc2 = nn.Linear(75, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights and biases from the Standard Normal distribution
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Student Model
class StudentModel(nn.Module):
    def __init__(self, layer_sizes, init_fn=None):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create layers based on sizes
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
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

# Generate test set
def generate_test_set(teacher_model, n_samples=60000, input_dim=100):
    with torch.no_grad():
        x_test = torch.FloatTensor(n_samples, input_dim).uniform_(0, 2)
        y_test = teacher_model(x_test)
        # x_test.dtype = torch.float32
        # y_test.dtype = torch.float32
        # x_test.shape = torch.Size([60000, 100])
        # y_test.shape = torch.Size([60000, 1])
        # float32: 4 bytes
        # x_test memory = 60000 * 100 * 4 bytes = 23.44 MB
        # y_test memory = 60000 * 1 * 4 bytes = 0.23 MB
    return x_test, y_test

# Training student models
def train_student(student_model,
                  teacher_model,
                  test_loader,
                  batch_size=128,
                  n_iterations=1000,
                  learning_rate=0.001,
                  save_test_every=100, 
                  device="cpu"
                  ):
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    student_model.train()
    train_losses, test_losses = [], []
    
    @torch.no_grad()
    def eval_test():
        student_model.eval()
        test_loss = 0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = student_model(x_batch)
            test_loss += mse_loss(y_pred, y_batch).item()
        return test_loss / len(test_loader)
    
    # Start time tracking
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Generate fresh batch
        x_batch = torch.FloatTensor(batch_size, 100).uniform_(0, 2).to(device)
        y_batch = teacher_model(x_batch).detach()  # Use teacher to label data
        
        # Forward pass
        y_pred = student_model(x_batch)
        loss = mse_loss(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if iteration % save_test_every == 0:
            test_loss = eval_test()
            test_losses.append(test_loss)
            print(f"Iteration {iteration}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")

    # End time tracking
    elapsed_time = time.time() - start_time
    print(f"Time taken to train student model for {n_iterations} iterations: {elapsed_time:.2f} seconds")

    final_loss = eval_test()
    
    return np.array(train_losses), np.array(test_losses), final_loss   

def train_students(student_models,
                   teacher_model,
                   test_loader,
                   **kwargs):
    results = {}
    for student_name, student_model in student_models.items():
        print(f"\n Training {student_name}...")
        results[student_name] = train_student(student_model, teacher_model, test_loader, **kwargs)
        
        print(f"Final test loss: {results[student_name][-1]:.4f}\n")
        
    return results

# ---- Experiment Runner for Parallelization ----
def run_experiment(teacher,
                   test_dataloader,
                   batch_size,
                   n_iterations,
                   learning_rate,
                   save_test_every,
                   gpu_id):
    """
    Runs a single experiment (specific n_iterations and learning_rate) on a given GPU.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Running experiment: n_iterations={n_iterations}, learning_rate={learning_rate}, device={device}")

    # Define Student Models
    student_underparam = StudentModel([100, 10, 1]).to(device)
    student_equalparam = StudentModel([100, 75, 50, 10, 1]).to(device)
    student_overparam = StudentModel([100, 200, 200, 200, 100, 1]).to(device)

    # Package the student models for training
    student_models = {
        "StudentUnderparam": student_underparam,
        "StudentEqualparam": student_equalparam,
        "StudentOverparam": student_overparam,
    }

    # Train the student models
    student_results = train_students(
        student_models,
        teacher,
        test_dataloader,
        batch_size=batch_size,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        save_test_every=save_test_every,
        device=device,
    )

    # Save Results and Models
    save_metrics(student_results, f"results_iter{n_iterations}_lr{learning_rate}_gpu{gpu_id}.pkl")
    save_model(student_underparam, f"student_underparam_iter{n_iterations}_lr{learning_rate}_gpu{gpu_id}")
    save_model(student_equalparam, f"student_equalparam_iter{n_iterations}_lr{learning_rate}_gpu{gpu_id}")
    save_model(student_overparam, f"student_overparam_iter{n_iterations}_lr{learning_rate}_gpu{gpu_id}")

    print(f"Experiment completed: n_iterations={n_iterations}, learning_rate={learning_rate}, device={device}")

# ---- Main Function ----
def main():
    # ---- Parameters ----
    N_SAMPLES = 60000  # Number of samples in the test dataset
    BATCH_SIZE = 128   # Batch size for training and testing
    NUMBER_SAVED_EVALUATIONS = 100 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Create Teacher Model ----
    teacher = TeacherModel()
    teacher.eval()  # Freeze the teacher model
    save_model(teacher, "teacher_model")  # Save the teacher model

    # ---- Generate Test Dataset ----
    x_test, y_test = generate_test_set(teacher, n_samples=N_SAMPLES)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Hyperparameter Search ----
    n_iterations_list = [10000, 50000, 100000, 1000000]
    learning_rate_list = [0.01, 0.001, 0.0001]

    # Create all combinations of (n_iterations, learning_rate)
    experiments = [(n_iter, lr) for n_iter in n_iterations_list for lr in learning_rate_list]

    # ---- Multiprocessing for Parallel Experiments ----
    num_gpus = torch.cuda.device_count()  # Number of available GPUs
    print(f"Number of GPUs available: {num_gpus}")

    # Wrap the experiment function for multiprocessing
    experiment_fn = partial(
        run_experiment,
        teacher=teacher,
        test_dataloader=test_dataloader,
        batch_size=BATCH_SIZE,
        save_test_every=SAVE_TEST_EVERY,
    )

    # Use multiprocessing.Pool to parallelize experiments across GPUs
    with mp.Pool(processes=num_gpus) as pool:
        pool.starmap(
            experiment_fn,
            [(n_iter, lr, gpu_id % num_gpus) for gpu_id, (n_iter, lr) in enumerate(experiments)],
        )

if __name__ == "__main__":
    main()
