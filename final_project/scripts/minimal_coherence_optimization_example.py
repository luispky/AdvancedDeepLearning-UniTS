import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import seed_everything
from src.minimal_coherence_optimization import (
    optimize_frame,
    compute_frame_properties,
    compute_coherence,
    normalize,
)


# ===============================
# Plotting Functions
# ===============================
def save_plot_to_file(filename, fig=None):
    """Save plot to ../plots/ directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(parent_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {filepath}")
    return filepath


def plot_loss_and_coherence_history(
    loss_history,
    coherence_history,
    welch_bound=None,
    upper_bound=None,
    ax=None,
    figsize=(8, 5),
):
    """Plot loss and coherence history with dual y-axes, including theoretical bounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    iterations = range(len(loss_history))

    # Plot loss on left y-axis
    color_loss = "tab:blue"
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surrogate Loss", color=color_loss)
    line1 = ax.plot(
        iterations, loss_history, color=color_loss, linewidth=2, label="Loss"
    )
    ax.tick_params(axis="y", labelcolor=color_loss)
    ax.grid(True, alpha=0.3)

    # Plot coherence on right y-axis
    ax2 = ax.twinx()
    color_coherence = "tab:red"
    ax2.set_ylabel("Coherence", color=color_coherence)
    line2 = ax2.plot(
        iterations,
        coherence_history,
        color=color_coherence,
        linewidth=2,
        label="Coherence",
    )
    ax2.tick_params(axis="y", labelcolor=color_coherence)

    # Add theoretical bound lines if provided
    bound_lines = []
    if welch_bound is not None and welch_bound > 0:
        line3 = ax2.axhline(
            y=welch_bound,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Welch Bound",
        )
        bound_lines.append(line3)

    if upper_bound is not None and upper_bound > 0:
        line4 = ax2.axhline(
            y=upper_bound,
            color="orange",
            linestyle="-.",
            linewidth=2,
            label="Upper Bound",
        )
        bound_lines.append(line4)

    ax.set_title("Optimization Progress: Loss and Coherence")

    # Create combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    for bound_line in bound_lines:
        lines.append(bound_line)
        labels.append(bound_line.get_label())
    ax.legend(lines, labels, loc="upper right")

    return ax


def plot_equiangularity_and_tightness_history(
    equiangularity_history, tightness_history, ax=None, figsize=(8, 5)
):
    """Plot equiangularity and tightness history with dual y-axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    iterations = range(len(equiangularity_history))

    # Plot equiangularity on left y-axis
    color_eq = "tab:orange"
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Equiangularity (Variance)", color=color_eq)
    line1 = ax.plot(
        iterations,
        equiangularity_history,
        color=color_eq,
        linewidth=2,
        label="Equiangularity",
    )
    ax.tick_params(axis="y", labelcolor=color_eq)
    ax.grid(True, alpha=0.3)

    # Plot tightness on right y-axis
    ax2 = ax.twinx()
    color_tight = "tab:purple"
    ax2.set_ylabel("Tightness (Frobenius Norm)", color=color_tight)
    line2 = ax2.plot(
        iterations, tightness_history, color=color_tight, linewidth=2, label="Tightness"
    )
    ax2.tick_params(axis="y", labelcolor=color_tight)

    ax.set_title("Frame Quality Metrics: Equiangularity and Tightness")

    # Create combined legend
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper right")

    return ax


def create_comprehensive_plot(
    X_init,
    X_opt,
    loss_history,
    coherence_history,
    equiangularity_history,
    tightness_history,
    welch_bound=None,
    upper_bound=None,
    save_plot=False,
    figsize=(16, 12),
    seed=None,
    alpha=None,
    beta=None,
):
    """
    Create a comprehensive 2x2 plot showing frame optimization results.

    Only works for 2D frames (m=2).

    Layout:
    - (0,0): Initial frame vectors
    - (0,1): Optimized frame vectors
    - (1,0): Loss and coherence history
    - (1,1): Equiangularity and tightness history
    """
    # Check if frames are 2D
    if X_init.shape[0] != 2:
        print("Comprehensive plot only available for 2D frames (m=2)")
        return

    # Create subplots with consistent aspect ratios
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.3
    )

    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ]

    # (0,0): Initial frame vectors
    plot_2d_frame_vectors(
        X_init,
        title=f"Initial Frame\nCoherence = {compute_coherence(X_init):.4f}",
        ax=axes[0][0],
    )

    # (0,1): Optimized frame vectors
    plot_2d_frame_vectors(
        X_opt,
        title=f"Optimized Frame\nCoherence = {compute_coherence(X_opt):.4f}",
        ax=axes[0][1],
    )

    # (1,0): Loss and coherence history
    plot_loss_and_coherence_history(
        loss_history,
        coherence_history,
        welch_bound,
        upper_bound,
        ax=axes[1][0],
    )

    # (1,1): Equiangularity and tightness history
    plot_equiangularity_and_tightness_history(
        equiangularity_history,
        tightness_history,
        ax=axes[1][1],
    )

    # Ensure consistent aspect ratios for time series plots
    axes[1][0].set_aspect("auto")
    axes[1][1].set_aspect("auto")

    # Save plot if requested
    if save_plot:
        n_vectors = X_opt.shape[1]
        filename = (
            f"frame_optimization_m2_n{n_vectors}_iter{len(loss_history)}"
            f"_seed{seed}_alpha{alpha}_beta{beta}.png"
        )
        save_plot_to_file(filename)

    plt.show()
    return fig


def create_optimization_plots(
    loss_history,
    coherence_history,
    equiangularity_history,
    tightness_history,
    welch_bound=None,
    upper_bound=None,
    frame_size=None,
    save_plot=False,
    seed=None,
    alpha=None,
    beta=None,
):
    """
    Create optimization plots for non-2D frames or as a standalone visualization.

    Creates a 1x2 layout with:
    - Left: Loss and coherence history with bounds
    - Right: Equiangularity and tightness history
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Loss and coherence history
    plot_loss_and_coherence_history(
        loss_history, coherence_history, welch_bound, upper_bound, ax=axes[0]
    )

    # Right plot: Equiangularity and tightness history
    plot_equiangularity_and_tightness_history(
        equiangularity_history, tightness_history, ax=axes[1]
    )

    if frame_size:
        m, n = frame_size
        fig.suptitle(
            f"Frame Optimization Results: {n} vectors in R^{m}", fontsize=16, y=1.02
        )

    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if frame_size:
            m, n = frame_size
            filename = (
                f"frame_optimization_m{m}_n{n}_iter{len(loss_history)}"
                f"_seed{seed}_alpha{alpha}_beta{beta}.png"
            )
        else:
            filename = (
                f"frame_optimization_iter{len(loss_history)}"
                f"_seed{seed}_alpha{alpha}_beta{beta}.png"
            )
        save_plot_to_file(filename)

    plt.show()
    return fig


def plot_2d_frame_vectors(
    X: torch.Tensor, title: str = "Frame Vectors", ax=None, figsize=(6, 6)
):
    """Plot frame vectors on the unit circle."""
    X_np = normalize(X).detach().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

    # Plot vectors
    colors = ["red", "blue", "green", "purple", "orange"]
    for i in range(X_np.shape[1]):
        vec = X_np[:, i]
        ax.arrow(
            0,
            0,
            vec[0],
            vec[1],
            head_width=0.05,
            length_includes_head=True,
            color=colors[i % len(colors)],
            linewidth=2,
        )
        ax.text(
            vec[0] * 1.15,
            vec[1] * 1.15,
            f"$v_{i + 1}$",
            fontsize=12,
            color=colors[i % len(colors)],
            fontweight="bold",
        )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return ax


# ===============================
# Main
# ===============================


def main():
    """
    Main function to demonstrate frame coherence optimization.

    This function optimizes a frame to minimize coherence and provides
    comprehensive visualization of the results including initial/final frames,
    optimization progress, and frame quality metrics.
    """

    # =================================================================
    # OPTIMIZATION PARAMETERS - Modify these for different experiments
    # =================================================================

    # Frame configuration
    m = 2  # Ambient dimension (vectors in R^m)
    n = 3  # Number of frame vectors

    # NOTE: The code now handles both cases:
    # - n > m: Welch bound applies, comprehensive analysis with bounds
    # - n <= m: Orthogonal frames possible, no meaningful lower bound

    seed = 42
    seed_everything(seed=seed)

    # Optimization settings
    num_iterations = 200
    learning_rate = 0.01

    # Loss function parameters
    lambda_softmax = (
        50.0  # Temperature for log-sum-exp approximation (higher = tighter)
    )
    alpha = 0.1  # Equiangularity regularization strength (0 = no regularization)
    beta = 0.1  # Tightness regularization strength (0 = no regularization)

    # =================================================================
    # OPTIMIZATION PROCESS
    # =================================================================

    # Optimize the frame with optional regularization.
    optimization_results = optimize_frame(
        m, n, num_iterations, learning_rate, lambda_softmax, alpha, beta
    )

    X_init = torch.tensor(optimization_results["X_init"])
    X_opt = torch.tensor(optimization_results["X_opt"])
    loss_history = optimization_results["loss_history"]
    coherence_history = optimization_results["coherence_history"]
    equiangularity_history = optimization_results["equiangularity_history"]
    tightness_history = optimization_results["tightness_history"]

    # Compute final frame properties.
    stats = compute_frame_properties(X_opt)
    print("\nFinal Frame Properties")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # =================================================================
    # RESULTS SUMMARY
    # =================================================================

    initial_coherence = compute_coherence(X_init)
    final_coherence = stats["coherence"]
    welch_bound = stats["welch_bound"]
    equiangularity = stats["equiangularity"]
    tightness = stats["tightness"]
    improvement = initial_coherence - final_coherence
    gap_from_optimum = final_coherence - welch_bound

    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Frame size:           {n} vectors in R^{m}")
    print(f"Iterations:           {num_iterations}")
    print(f"Initial coherence:    {initial_coherence:.6f}")
    print(f"Final coherence:      {final_coherence:.6f}")
    print(f"Welch bound:          {welch_bound:.6f}")
    print(f"Equiangularity:       {equiangularity:.6f}")
    print(f"Tightness:            {tightness:.6f}")
    print(f"Improvement:          {improvement:.6f}")
    print(f"Gap from optimum:     {gap_from_optimum:.6f}")
    if welch_bound > 0:
        print(f"Relative gap:         {(gap_from_optimum / welch_bound) * 100:.2f}%")
    else:
        print("Relative gap:         N/A (orthogonal frame possible)")
    print("=" * 60)

    # =================================================================
    # VISUALIZATION
    # =================================================================

    if m == 2:
        # For 2D frames, create comprehensive 2x2 plot with vector visualizations
        print("Creating comprehensive 2D visualization...")
        create_comprehensive_plot(
            X_init,
            X_opt,
            loss_history,
            coherence_history,
            equiangularity_history,
            tightness_history,
            welch_bound=stats["welch_bound"],
            upper_bound=stats["upper_bound"],
            save_plot=True,
            seed=seed,
            alpha=alpha,
            beta=beta,
        )
    else:
        # For higher-dimensional frames, create optimization plots only
        print(f"Creating optimization plots for {m}D frame...")
        create_optimization_plots(
            loss_history,
            coherence_history,
            equiangularity_history,
            tightness_history,
            welch_bound=stats["welch_bound"],
            upper_bound=stats["upper_bound"],
            frame_size=(m, n),
            save_plot=True,
            seed=seed,
            alpha=alpha,
            beta=beta,
        )


if __name__ == "__main__":
    main()
