import matplotlib.pyplot as plt
from src.gcnn import CyclicGroup, DihedralGroup, LiftingKernel


def visualize_transformed_grids(
    group: CyclicGroup | DihedralGroup,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
):
    lifting_kernel = LiftingKernel(group, kernel_size, in_channels, out_channels)
    transformed_grids = lifting_kernel.transformed_grid_R2

    num_elements = group.numel()

    fig, ax = plt.subplots(1, num_elements, figsize=(12, 3), sharey=True)
    # share x and y axes
    for i in range(1, num_elements):
        ax[i].tick_params(
            left=False,
        )
        ax[i].set_ylabel("")

    for idx in range(num_elements):
        grid = transformed_grids[:, idx, :, :]
        ax[idx].scatter(grid[1, :, :], grid[0, :, :], c="r")
        ax[idx].scatter(grid[1, 0, 0], grid[0, 0, 0], marker="x", c="b", s=60)
        if idx == 0:
            ax[idx].set_title(r"$e$")
            ax[idx].set_xlabel(r"$x$")
            ax[idx].set_ylabel(r"$y$")
        else:
            ax[idx].set_title(f"$g^{idx - 1}$")

    fig.text(0.5, 0.05, "Group elements", ha="center")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_transformed_grids(
        CyclicGroup(order=4), kernel_size=7, in_channels=3, out_channels=1
    )

    visualize_transformed_grids(
        DihedralGroup(order=4), kernel_size=7, in_channels=3, out_channels=1
    )
