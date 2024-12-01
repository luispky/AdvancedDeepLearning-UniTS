import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

# Import from utils
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
    
# Example of a custom initialization function
def custom_init_fn(layer):
    """
    Custom initialization: weights from Normal(0, 1), biases set to zero.
    """
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        nn.init.normal_(layer.bias, mean=0.0, std=1.0)

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

def run_teacher_student_experiment(teacher,
                                   test_dataloader,
                                   batch_size,
                                   n_iterations,
                                   save_test_every,
                                   learning_rate,
                                   device
                                   ):
    
    # Start time tracking
    start_time = time.time()

    # Student models
    student_underparam = StudentModel([100, 10, 1]) # Under-parameterized
    student_equalparam = StudentModel([100, 75, 50, 10, 1]) # Equally parameterized
    student_overparam = StudentModel([100, 200, 200, 200, 100, 1]) # Over-parameterized
    
    # Train the student models
    student_models = {
        "StudentUnderparam": student_underparam,
        "StudentEqualparam": student_equalparam,
        "StudentOverparam": student_overparam
    }
    
    student_results = train_students(student_models,
                                     teacher,
                                     test_dataloader,
                                     batch_size=batch_size,
                                     n_iterations=n_iterations,  
                                     learning_rate=learning_rate,
                                     save_test_every=save_test_every, 
                                     device=device
                                     )
    
    # Save the student models
    save_model(student_underparam, f"student_underparam_iter{n_iterations}_lr{learning_rate}")
    save_model(student_equalparam, f"student_equalparam_iter{n_iterations}_lr{learning_rate}")
    save_model(student_overparam, f"student_overparam_iter{n_iterations}_lr{learning_rate}")
    
    # Save the results
    save_metrics(student_results, f"student_results_iter{n_iterations}_lr{learning_rate}")
    
    # Plot the losses
    plot_losses(student_results, n_iterations, save_test_every, f"student_results_iter{n_iterations}_lr{learning_rate}")
    
    # Models Dictionary
    models = {
        "Teacher": teacher,
        "StudentUnderparam": student_underparam,
        "StudentEqualparam": student_equalparam,
        "StudentOverparam": student_overparam
    }
    
    # Compute and save the statistics of the models parameters
    compute_and_save_stats_teacher_student(models, test_dataloader, f"teacher_student_iter{n_iterations}_lr{learning_rate}")

    # End time tracking
    elapsed_time = time.time() - start_time
    print(f"Time taken for experiment with {n_iterations} iterations and learning rate {learning_rate}: {elapsed_time:.2f} seconds")

    
def main():
    # Parameters
    N_SAMPLES = 60000
    BATCH_SIZE = 128
    SAVE_TEST_EVERY = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Teacher model
    teacher = TeacherModel()
    teacher.eval()  # Teacher is frozen
    
    # Save teacher model
    save_model(teacher, "teacher_model")
    
    sys.exit()
    
    # Generate the test_dataset
    x_test, y_test = generate_test_set(teacher, n_samples=N_SAMPLES)
    # Wrap the dataset in a TensorDataset
    test_dataset = TensorDataset(x_test, y_test)
    # Create a DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    n_iterations_list = [10000, 50000, 100000, 1000000]
    learning_rate_list = [0.01, 0.001, 0.0001]

    for n_iterations in n_iterations_list:
        for learning_rate in learning_rate_list:
            print(f"\nRunning experiment with {n_iterations} iterations and learning rate {learning_rate}...")
            run_teacher_student_experiment(teacher,
                                           test_dataloader,
                                           BATCH_SIZE,
                                           n_iterations,
                                           SAVE_TEST_EVERY,
                                           learning_rate,
                                           DEVICE
                                           )

if __name__ == "__main__":
    
    main()