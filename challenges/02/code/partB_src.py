import numpy as np
import itertools
from typing import Callable, Tuple, Optional
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Callable, Dict, Tuple, List
import logging
from utils import FIGURES_DIR


def B6(x: np.ndarray) -> float:
    """
    Evaluate the hierarchical polynomial B6 at a single input vector x.
    
    The polynomial is defined as:
    
        B6(x) = x1^6 + 15*x2*x1^4 + 20*x3*x1^3 + 45*x2^2*x1^2 +
                15*x2^3 + 60*x3*x2*x1 + 15*x4*x1^2 + 10*x3^2 +
                15*x4*x2 + 6*x5*x1 + x6
                
    Parameters:
      x : np.ndarray
          1D array of length 6 representing [x1, x2, x3, x4, x5, x6].
    
    Returns:
      float
          The computed value of B6 at x.
    """
    return (
        x[0]**6 +
        15 * x[1] * x[0]**4 +
        20 * x[2] * x[0]**3 +
        45 * x[1]**2 * x[0]**2 +
        15 * x[1]**3 +
        60 * x[2] * x[1] * x[0] +
        15 * x[3] * x[0]**2 +
        10 * x[2]**2 +
        15 * x[3] * x[1] +
        6  * x[4] * x[0] +
        x[5]
    )


def generate_non_hierarchical_B6_polynomial(
    seed: int,
    symbolic: bool = False
) -> Tuple[Callable[[np.ndarray], float], Optional[sp.Expr]]:
    """
    Generate a function representing the non-hierarchical polynomial B6 tilde,
    obtained by scrambling the monomials of B6 using unique variable permutations.
    
    The requirements enforced are:
      1. B6 tilde depends non-trivially on all six input variables.
      2. Each monomial gets a unique permutation.
      3. No two monomials can be rearranged by commutativity to become identical—
         in particular, the two terms originally corresponding to 15*x4*x2 and 6*x5*x1
         are forced to be distinct.
    
    Optionally, a symbolic expression (using sympy) is returned.
    
    Parameters:
      seed : int
          Random seed for reproducibility.
      symbolic : bool, optional
          If True, return a sympy expression for B6 tilde as the second element of the tuple.
          Otherwise, return None for the symbolic expression.
    
    Returns:
      A tuple (B6_tilde, symbolic_expr) where:
        - B6_tilde is a callable function that accepts a 1D NumPy array of length 6.
        - symbolic_expr is the sympy expression for B6 tilde if symbolic is True, else None.
    """
    # Set random seed for reproducibility.
    np.random.seed(seed)
    
    # Define the monomial structure as tuples of exponents (for x1, x2, ..., x6).
    monomials = [
        (6, 0, 0, 0, 0, 0),   # x1^6
        (4, 1, 0, 0, 0, 0),   # x1^4 * x2
        (3, 0, 1, 0, 0, 0),   # x1^3 * x3
        (2, 2, 0, 0, 0, 0),   # x1^2 * x2^2
        (0, 3, 0, 0, 0, 0),   # x2^3
        (1, 1, 1, 0, 0, 0),   # x1 * x2 * x3
        (2, 0, 0, 1, 0, 0),   # x1^2 * x4
        (0, 0, 2, 0, 0, 0),   # x3^2
        (0, 1, 0, 1, 0, 0),   # x2 * x4  --> special case 1 (15*x4*x2)
        (1, 0, 0, 0, 1, 0),   # x1 * x5  --> special case 2 (6*x5*x1)
        (0, 0, 0, 0, 0, 1),   # x6
    ]
    coefficients = [1, 15, 20, 45, 15, 60, 15, 10, 15, 6, 1]
    
    # Generate all unique permutations of the indices [0, 1, 2, 3, 4, 5], 
    # which is 6! = 720 permutations!
    # This accounts for all possible permutations of the variables in possible
    # monomials with 6 variables.
    all_perms = list(itertools.permutations(range(6)))
    np.random.shuffle(all_perms)
    
    # Choose one unique permutation for each monomial.
    chosen_perms = all_perms[:len(monomials)]
    
    # Identify the indices for the two special monomials:
    #   Special monomial 1: (0, 1, 0, 1, 0, 0) --> corresponds to 15*x4*x2
    #   Special monomial 2: (1, 0, 0, 0, 1, 0) --> corresponds to 6*x5*x1
    special_idx1, special_idx2 = None, None
    for idx, mono in enumerate(monomials):
        if mono == (0, 1, 0, 1, 0, 0):
            special_idx1 = idx
        elif mono == (1, 0, 0, 0, 1, 0):
            special_idx2 = idx
    
    # Ensure the two special monomials do not share the same permutation.
    if special_idx1 is not None and special_idx2 is not None:
        if chosen_perms[special_idx1] == chosen_perms[special_idx2]:
            # Find an alternative permutation not already used.
            for perm in all_perms:
                if perm not in chosen_perms:
                    chosen_perms[special_idx2] = perm
                    break

    # -----------------------------------------------------------------------------
    # Define the numerical function for B6 tilde.
    # -----------------------------------------------------------------------------
    def B6_tilde(x: np.ndarray) -> float:
        """
        Evaluate the non-hierarchical polynomial B6 tilde at input vector x.
        
        The function applies a unique permutation to the variables in each monomial.
        
        Parameters:
          x : np.ndarray of shape (6,)
        
        Returns:
          float: The value of B6 tilde at x.
        """
        total = 0.0
        # Iterate over each monomial.
        for coeff, exponents, perm in zip(coefficients, monomials, chosen_perms):
            # Apply the permutation: the i-th exponent is applied to x[perm[i]].
            term_val = coeff
            for i in range(6):
                term_val *= x[perm[i]] ** exponents[i]
            total += term_val
        return total

    # -----------------------------------------------------------------------------
    # If a symbolic expression is requested, build it using sympy.
    # -----------------------------------------------------------------------------
    symbolic_expr = None
    if symbolic:
        # Define symbolic variables x1, x2, ..., x6.
        sym_vars = sp.symbols('x1 x2 x3 x4 x5 x6')
        symbolic_expr = 0
        for coeff, exponents, perm in zip(coefficients, monomials, chosen_perms):
            # Create the term: use the permuted symbolic variables.
            term = coeff * sp.prod([sym_vars[perm[i]] ** exponents[i] for i in range(6)])
            symbolic_expr += term
        symbolic_expr = sp.expand(symbolic_expr)
        
        logging.info("Symbolic expression for B6 tilde:")
        logging.info(sp.latex(symbolic_expr))

    return B6_tilde, symbolic_expr


def generate_dataset(num_samples: int, 
                     func: Callable[[np.ndarray], float]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset by sampling inputs uniformly from [0,2]^6 and computing the outputs
    for the function func.
    
    Parameters:
      num_samples : int
          Number of data points to generate.
      func : Callable
          Function that computes B6 tilde on a single 1D input vector.
    
    Returns:
        A tuple (X_data, Y_data) where:
            - X_data is a NumPy array of shape (num_samples, 6) containing the input vectors.
            - Y_data is a NumPy array of shape (num_samples,) containing the corresponding outputs.
    """
    X_data = np.random.uniform(0, 2, size=(num_samples, 6))
    # Use np.apply_along_axis for clarity (alternatively, list comprehensions work too)
    Y_data = np.apply_along_axis(func, 1, X_data)
    return X_data, Y_data


def generate_data_loaders(
    func: Callable[[np.ndarray], float],
    num_train: int = int(1e5),
    num_test: int = int(6e4),
    batch_size: int = 20,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    # Generate dataset (inputs and corresponding outputs)
    X_train_np, Y_train_np = generate_dataset(num_train, func)
    X_test_np,  Y_test_np  = generate_dataset(num_test, func)
    
    # Convert datasets to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class ResidualBlock(nn.Module):
    """
    A residual block for fully-connected layers.
    This block applies a linear transformation followed by ReLU activation.
    If the input and output dimensions match, it adds a skip connection.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        # Enable skip connection only if input and output dimensions are the same.
        self.use_skip = (in_features == out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        if self.use_skip:
            out = out + x  # Residual (skip) connection
        return out


class ResidualNetwork(nn.Module):
    """
    A fully-connected residual network with:
      - 1 input layer (dimension 6 -> hidden_dim)
      - 8 hidden layers (each of dimension hidden_dim)
      - 1 output layer (hidden_dim -> 1)
    Skip connections (residual connections) are applied on hidden layers where 
    the input and output dimensions match.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 50, output_dim: int = 1, num_hidden: int = 8):
        super().__init__()
        
        # Input layer: from input_dim to hidden_dim (no skip connection, dimensions differ)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        
        # Hidden layers: first hidden layer is created by the input layer's activation.
        # The subsequent (num_hidden) layers are residual blocks.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden):
            # For these layers, in_features == out_features == hidden_dim, so skip is enabled.
            self.hidden_layers.append(ResidualBlock(hidden_dim, hidden_dim))
        
        # Output layer: from hidden_dim to output_dim (no skip connection, dimensions differ)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through input layer and activation.
        x = self.activation(self.input_layer(x))
        # Pass through the hidden residual layers.
        for layer in self.hidden_layers:
            x = layer(x)
        # Output layer (no activation as per assignment).
        x = self.output_layer(x)
        return x


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
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs: int = 30,
                learning_rate: float = 1e-3,
                device: torch.device = torch.device("cpu"),
                lr_factor: float = 0.5,
                lr_patience: int = 3,
                lr_verbose: bool = True
               ) -> Tuple[List[float], List[float]]:
    """
    Train the given model using the Adam optimizer, MSE loss, and a learning rate scheduler.
    
    The scheduler used is ReduceLROnPlateau, which reduces the learning rate by a specified factor
    if the test loss does not improve for a given number of epochs (patience).
    
    Parameters:
      model         : The PyTorch model to be trained.
      train_loader  : DataLoader for training data.
      test_loader   : DataLoader for test data.
      num_epochs    : Number of training epochs.
      learning_rate : Initial learning rate for the optimizer.
      device        : Torch device (CPU or CUDA).
      lr_factor     : Factor by which to reduce the learning rate (e.g., 0.5).
      lr_patience   : Number of epochs with no improvement after which learning rate is reduced.
      lr_verbose    : If True, prints a message each time the learning rate is reduced.
    
    Returns:
      A tuple (train_losses, test_losses) containing the MSE losses logged after each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the scheduler with customizable parameters.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=lr_factor,
                                                     patience=lr_patience,
                                                     verbose=lr_verbose)
    
    train_losses = []
    test_losses = []
    
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        
        # Iterate over training batches.
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(), batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_inputs.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluate on the test set.
        epoch_test_loss = evaluate_model(model, test_loader, device)
        test_losses.append(epoch_test_loss)
        
        # Step the scheduler based on the test loss.
        scheduler.step(epoch_test_loss)
        
        logging.info(f"Epoch [{epoch:02d}/{num_epochs}] -- Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}")
    
    return train_losses, test_losses


def aggregated_variable_sweep(model: torch.nn.Module,
                              target_function: Callable[[np.ndarray], float],
                              grid: np.ndarray,
                              num_trials: int = 5,
                              device: torch.device = torch.device("cpu")
                             ) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Perform aggregated variable sweep analysis for a given model and target function.
    
    For each input variable (index 0..5), the function performs the following:
      - Sample a random input vector x0 (from [0,2]^6).
      - For each grid value, create a batch by replacing the i-th variable of x0 with the grid values.
      - Compute model predictions (vectorized in one forward pass).
      - Compute target function evaluations (using a vectorized approach).
      - Repeat for num_trials and aggregate (compute mean and std).
    
    Parameters:
      model           : The trained PyTorch model.
      target_function : A function that accepts a 1D NumPy array (shape (6,)) and returns a float.
      grid            : 1D NumPy array of grid values over [0,2] (e.g. np.linspace(0, 2, 100)).
      num_trials      : Number of random input vectors to sample.
      device          : Torch device.
      
    Returns:
      A dictionary with keys 0 through 5 (for each input variable). For each key, the value is a dict:
          {
             'model_mean'  : np.ndarray of shape (len(grid),) – mean model prediction over trials,
             'model_std'   : np.ndarray of shape (len(grid),) – std deviation of model prediction,
             'target_mean' : np.ndarray of shape (len(grid),) – mean target function evaluation,
             'target_std'  : np.ndarray of shape (len(grid),) – std deviation of target evaluations
          }
    """
    # Initialize dictionary to store lists of results per variable per trial.
    results = {i: {'model': [], 'target': []} for i in range(6)}
    n_points = grid.shape[0]
    
    # Loop over trials.
    for trial in range(num_trials):
        # Sample a random input vector from [0,2]^6.
        x0 = np.random.uniform(0, 2, size=(6,))
        # For each variable index.
        for i in range(6):
            # Create a batch: replicate x0 into a (n_points, 6) array.
            X_batch = np.tile(x0, (n_points, 1))
            # Replace the i-th column with grid values.
            X_batch[:, i] = grid
            
            X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                y_model = model(X_batch_tensor).cpu().numpy().flatten()
            
            y_target = np.apply_along_axis(target_function, 1, X_batch)
            
            # Append the results for variable i.
            results[i]['model'].append(y_model)
            results[i]['target'].append(y_target)
    
    # Aggregate results over trials: compute mean and standard deviation.
    aggregated = {}
    for i in range(6):
        model_array = np.array(results[i]['model'])  # shape (num_trials, n_points)
        target_array = np.array(results[i]['target'])
        aggregated[i] = {
            'model_mean': model_array.mean(axis=0),
            'model_std': model_array.std(axis=0),
            'target_mean': target_array.mean(axis=0),
            'target_std': target_array.std(axis=0)
        }
    return aggregated


def plot_variable_sweep_results(aggregated_results: Dict[int, Dict[str, np.ndarray]],
                                grid: np.ndarray,
                                poly_name: str,
                                title: str,
                                save_fig: bool = False,
                                filename: str = None) -> None:
    """
    Plot aggregated_results variable sweep results for a single polynomial in a 3x2 grid.
    
    For each input variable (x1, ..., x6), a subplot is created showing:
      - The target function evaluations (mean ± std) labeled as "Target".
      - The model evaluations (mean ± std) labeled as "Model".
    
    Parameters:
        aggregated_results : Dict mapping variable index to a dict with keys 'target_mean', 
                    'target_std', 'model_mean', 'model_std'.
        grid       : 1D NumPy array of grid values.
        poly_name  : Name of the polynomial (used in the title and y-axis labels).
        title      : Overall title for the figure.
        save_fig   : If True, the figure is saved to disk.
        filename   : Filename for saving the figure (if save_fig is True).
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        # Plot target function results.
        ax.fill_between(grid,
                        aggregated_results[i]['target_mean'] - aggregated_results[i]['target_std'],
                        aggregated_results[i]['target_mean'] + aggregated_results[i]['target_std'],
                        color='green', alpha=0.2, label='Target ± std')
        ax.plot(grid, aggregated_results[i]['target_mean'], color='green', linestyle='-', label='Target')
        
        # Plot model results.
        ax.fill_between(grid,
                        aggregated_results[i]['model_mean'] - aggregated_results[i]['model_std'],
                        aggregated_results[i]['model_mean'] + aggregated_results[i]['model_std'],
                        color='orange', alpha=0.2, label='Model ± std')
        ax.plot(grid, aggregated_results[i]['model_mean'], color='orange', linestyle='--', label='Model')
        
        ax.set_xlabel('Input variable value')
        ax.set_title(f'Sweep for x[{i+1}]')
        ax.legend(loc='upper right')
    
    fig.suptitle(f"{title} - {poly_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_fig:
        if filename is None:
            filename = f"variable_sweep_{poly_name}"
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        print(f"Figure saved at: {filepath}")
        plt.close()  # Close the plot to avoid displaying it.
    else:
        plt.show()