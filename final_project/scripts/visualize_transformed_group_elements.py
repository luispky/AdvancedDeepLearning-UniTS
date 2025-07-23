import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from src.gcnn import CyclicGroup, InterpolativeGroupKernel


def visualize_transformed_group_elements(
    group,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
):
    """
    Visualize the transformed group element grids for a given group, kernel size, and channel configuration.
    Each subplot shows the transformed grid for a specific group element in (x, y, h) space.
    """
    gk = InterpolativeGroupKernel(
        group=group,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    transformed_grid = gk.transformed_grid_R2xH  # shape: (3, |H|, |H|, ks, ks)
    num_group_elements = group.numel()

    # Flatten spatial and group grid dimensions for visualization
    grid_flat = transformed_grid.reshape(
        3,  # (x, y, h)
        num_group_elements,  # |H|
        num_group_elements * kernel_size * kernel_size,  # |H| * ks * ks
    )

    fig, ax = plt.subplots(
        1,
        num_group_elements,
        subplot_kw=dict(projection="3d"),
        figsize=(2.5 * num_group_elements, 3),
    )
    if num_group_elements == 1:
        ax = [ax]

    for group_elem in range(num_group_elements):
        ax[group_elem].scatter(
            grid_flat[1, group_elem, 1:],  # x
            grid_flat[0, group_elem, 1:],  # y
            grid_flat[2, group_elem, 1:],  # h
            c="r",
            s=8,
            alpha=0.7,
        )
        ax[group_elem].scatter(
            grid_flat[1, group_elem, 0],  # x
            grid_flat[0, group_elem, 0],  # y
            grid_flat[2, group_elem, 0],  # h
            marker="x",
            c="b",
            s=40,
        )
        ax[group_elem].set_title(f"g={group_elem}")
        ax[group_elem].set_xlabel("x")
        ax[group_elem].set_ylabel("y")
        ax[group_elem].set_zlabel("h")
        ax[group_elem].view_init(elev=20, azim=45)

    fig.text(0.5, 0.04, "Group elements", ha="center")
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    visualize_transformed_group_elements(
        group=CyclicGroup(order=4),
        kernel_size=7,
        in_channels=1,
        out_channels=1,
    )
