import torch
import numpy as np
import torch.optim as optim
from typing import Dict
from tqdm import tqdm


def normalize(X: torch.Tensor) -> torch.Tensor:
    """
    Normalize each column of X to have unit L2 norm.
    Each column represents a frame vector.

    Args:
        X (Tensor): A tensor of shape (m, n) where each column is a vector.

    Returns:
        Tensor: Normalized tensor with unit-norm columns.
    """
    return X / (X.norm(dim=0, keepdim=True) + 1e-8)


def compute_gram(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix G = X^T X.

    Args:
        X (Tensor): A tensor of shape (m, n) (each column is a vector).

    Returns:
        Tensor: Gram matrix of shape (n, n).
    """
    X_norm = normalize(X)
    return X_norm.T @ X_norm


def get_upper_triangular_elements(X: torch.Tensor) -> torch.Tensor:
    """
    Extract the upper triangular elements of a square matrix (above diagonal).

    This function extracts all elements X[i,j] where i < j, which correspond
    to the unique pairwise inner products between different frame vectors.

    Args:
        X (Tensor): A square tensor of shape (n, n).

    Returns:
        Tensor: 1D tensor containing upper triangular elements.
    """
    n = X.shape[0]
    upper_indices = torch.triu_indices(n, n, offset=1)
    return X[upper_indices[0], upper_indices[1]]


def surrogate_coherence_loss(
    X: torch.Tensor, lambda_softmax: float = 20.0, alpha: float = 0.0, beta: float = 0.0
) -> torch.Tensor:
    """
    Compute a smooth surrogate for minimizing the maximum inner product between frame vectors.

    This function implements a log-sum-exp approximation to the maximum inner product:
        max_{i ≠ j} <x_i, x_j>

    The log-sum-exp surrogate is:
        L_coh(X) = (1/λ) * log(∑_{i<j} exp(λ * <x_i, x_j>))

    where λ (lambda_softmax) controls the tightness of the approximation.
    Higher λ values give tighter approximations to the true maximum.

    Optionally includes regularization terms:
        L_total(X) = L_coh(X) + α * L_equiangular(X) + β * L_tight(X)

    Where:
        - L_equiangular(X) = Var({|<x_i, x_j>| : i < j}) encourages equiangularity
        - L_tight(X) = ||XX^T - (n/m)I_m||_F^2 encourages tightness

    Args:
        X (Tensor): Frame matrix of shape (m, n) where columns are frame vectors.
        lambda_softmax (float): Temperature parameter for log-sum-exp approximation.
        alpha (float): Regularization strength for equiangularity (0 = no regularization).
        beta (float): Regularization strength for tightness (0 = no regularization).

    Returns:
        Tensor: Scalar surrogate loss value.
    """
    with torch.no_grad():
        X.copy_(normalize(X))
    G = compute_gram(X)
    upper_diag_elements = get_upper_triangular_elements(G)
    loss_coh = (1.0 / lambda_softmax) * torch.log(
        torch.sum(torch.exp(lambda_softmax * upper_diag_elements))
    )

    total_loss = loss_coh

    # Equiangularity regularization: variance of off-diagonal elements
    if alpha > 0.0:
        loss_equiangular = upper_diag_elements.abs().var()
        total_loss += alpha * loss_equiangular

    # Tightness regularization: Frobenius norm of XX^T - (n/m)I_m
    if beta > 0.0:
        m, n = X.shape
        S = X @ X.T  # Frame operator: XX^T (m x m)
        target = (n / m) * torch.eye(m, device=X.device)  # Target for tight frame
        loss_tight = torch.norm(S - target, p="fro") ** 2  # Squared Frobenius norm
        total_loss += beta * loss_tight

    return total_loss


def compute_welch_bound(m: int, n: int) -> float:
    """
    Compute the Welch bound for a frame.

    The Welch bound applies when n > m. For n <= m, returns 0 since
    orthogonal frames are possible and coherence can be made arbitrarily small.

    Args:
        m (int): Dimension of each vector.
        n (int): Number of vectors.

    Returns:
        float: Welch bound (0 if n <= m).
    """
    # Orthogonal frames possible, no lower bound on coherence
    return 0.0 if n <= m else np.sqrt((n - m) / (m * (n - 1)))


def compute_coherence(X: torch.Tensor) -> float:
    """
    Compute the coherence of a frame.
    """
    G = compute_gram(X)
    upper_diag_elements = get_upper_triangular_elements(G)
    return upper_diag_elements.abs().max().item()


def compute_frame_properties(X: torch.Tensor, only_essential: bool = False) -> Dict:
    """
    Compute various properties of the frame X.

    This function computes the following properties for a frame X (an m x n matrix
    whose columns are unit-norm vectors in R^m):

      - 'coherence': maximum coherence, μ(X) = max_{i ≠ j} |<x_i, x_j>|.
      - 'tightness': Frobenius norm ||X X^T - (n/m) I_m||_F, which is 0 for a tight frame.
      - 'equiangularity': variance of the off-diagonal elements of the Gram matrix (lower is better).

    If only_essential=False, also computes:
      - 'unique_abs_count': number of unique absolute inner products (from the upper triangular part).
      - 'unique_angles': array of unique angles (in degrees) between vectors.
      - 'welch_bound': the Welch bound for the frame.
      - 'upper_bound': an upper bound on coherence computed as √(r) * Welch bound, where r is the unique_abs_count.

    Args:
        X: Frame matrix (m x n) where columns are unit-norm vectors
        only_essential: If True, compute only coherence, tightness, equiangularity

    Returns:
        dict: A dictionary containing the requested statistics.
    """
    m, n = X.shape  # m: ambient dimension, n: number of vectors
    G = compute_gram(X)  # G is an (n x n) matrix since X is (m x n)
    upper_diag_elements = get_upper_triangular_elements(G).cpu().detach()

    # Frame operator matrix for computing tightness
    S = X @ X.T  # (m x m)
    target = (n / m) * torch.eye(m, device=X.device)

    # Essential properties (always computed)
    properties = {
        "coherence": upper_diag_elements.abs().max().item(),
        "tightness": torch.norm(S - target, p="fro").item(),
        "equiangularity": upper_diag_elements.abs().var().item(),
    }

    # Additional properties (only if requested)
    if not only_essential:
        unique_abs_count = len(
            np.unique(np.round(np.abs(upper_diag_elements.numpy()), 1))
        )
        welch_bound = compute_welch_bound(m, n)

        properties.update(
            {
                "unique_abs_count": unique_abs_count,
                "unique_angles": np.unique(
                    np.round(
                        np.rad2deg(
                            np.arccos(np.clip(upper_diag_elements.numpy(), -1, 1))
                        ),
                        1,
                    )
                ),
                "welch_bound": welch_bound,
                "upper_bound": np.sqrt(unique_abs_count) * welch_bound,
            }
        )

    return properties


# ===============================
# Optimization Function
# ===============================


def optimize_frame(
    m: int,
    n: int,
    num_iterations: int = 200,
    learning_rate: float = 0.1,
    lambda_softmax: float = 20.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    verbose: bool = True,
):
    """
    Optimize a frame consisting of n vectors in R^m (columns of X) to minimize the maximum inner product:

        L(X) = μ(X) = min_{X in R^(m x n)} max_{1 ≤ i < j ≤ n} <x_i, x_j>,
    subject to ||x_i||_2 = 1 for each vector (each column of X).

    We use the smooth surrogate loss:
         L_coh(X) = (1/λ) * log(∑_{i ≠ j} exp(λ * <x_i, x_j>)),
    and optionally add regularization terms:
         L_total(X) = L_coh(X) + α * L_equiangular(X) + β * L_tight(X)

    Where:
         - L_equiangular(X) = Var({|<x_i, x_j>| : i ≠ j, i < j}) encourages equiangularity
         - L_tight(X) = ||XX^T - (n/m)I_m||_F^2 encourages tightness

    Args:
        m (int): Dimension of each vector.
        n (int): Number of vectors.
        num_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for AdamW.
        lambda_softmax (float): Temperature parameter for the surrogate loss.
        alpha (float): Regularization parameter for equiangularity (default 0, no regularization).
        beta (float): Regularization parameter for tightness (default 0, no regularization).
        verbose (bool): If True, print loss periodically.

    Returns:
        X_init: The initial normalized frame (m x n).
        X_opt: The optimized normalized frame (m x n).
        loss_history: List of loss values.
    """
    # Initialize X with shape (m, n), where each column is a vector.
    X = torch.randn(m, n, requires_grad=True)
    optimizer = optim.AdamW([X], lr=learning_rate)
    loss_history = []
    coherence_history = []
    equiangularity_history = []
    tightness_history = []

    print(f"Optimizing {n} vectors in R^{m}")
    print(f"Initial coherence: {compute_frame_properties(X)['coherence']:.4f}")

    with torch.no_grad():
        # Makes frame vectors unit-norm.
        X_init = normalize(X.clone())

    # Initialize progress tracking
    progress_bar = tqdm(total=num_iterations, desc="Optimizing frame", unit="iter")

    for _ in range(num_iterations):
        # Forward pass and optimization step
        optimizer.zero_grad()
        loss = surrogate_coherence_loss(X, lambda_softmax, alpha, beta)
        # # Enforce unit-norm constraint on frame vectors after each step
        loss.backward()
        optimizer.step()

        # Record metrics for this iteration
        current_properties = compute_frame_properties(X)
        loss_history.append(loss.item())
        coherence_history.append(current_properties["coherence"])
        equiangularity_history.append(current_properties["equiangularity"])
        tightness_history.append(current_properties["tightness"])

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            coherence=f"{current_properties['coherence']:.4f}",
        )
        progress_bar.update(1)

    progress_bar.close()

    with torch.no_grad():
        X_opt = normalize(X.clone())

    return {
        "X_init": X_init.detach().cpu().numpy(),
        "X_opt": X_opt.detach().cpu().numpy(),
        "loss_history": loss_history,
        "coherence_history": coherence_history,
        "equiangularity_history": equiangularity_history,
        "tightness_history": tightness_history,
    }
