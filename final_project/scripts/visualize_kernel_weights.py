import matplotlib.pyplot as plt
from src.gcnn import CyclicGroup, DihedralGroup, InterpolativeLiftingKernel


def visualize_kernel_weights(
    group,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
):
    """
    Visualize the sampled kernel weights for a given group, kernel size, and channel configuration.
    Each subplot shows the kernel for a specific output channel, input channel, and group element.
    """
    ik = InterpolativeLiftingKernel(
        group=group,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    weights = ik.sample()  # shape: (out_channels, group_elements, in_channels, ks, ks)
    num_group_elements = group.numel()

    # Reduce the size of each subplot for compactness
    fig, ax = plt.subplots(
        out_channels,
        in_channels * num_group_elements,
        figsize=(1.2 * in_channels * num_group_elements, 1.2 * out_channels),
        squeeze=False,
    )

    for out_channel in range(out_channels):
        for in_channel in range(in_channels):
            for group_elem in range(num_group_elements):
                idx = in_channel * num_group_elements + group_elem
                ax[out_channel, idx].imshow(
                    weights[out_channel, group_elem, in_channel, :, :].detach().numpy(),
                    cmap="viridis",
                )
                ax[out_channel, idx].set_xticks([])
                ax[out_channel, idx].set_yticks([])
                if out_channel == 0:
                    ax[out_channel, idx].set_title(
                        f"in={in_channel}\ng={group_elem}", fontsize=7
                    )
                if group_elem == 0 and in_channel == 0:
                    ax[out_channel, idx].set_ylabel(f"out={out_channel}", fontsize=8)

    fig.text(0.5, 0.04, "Input channel / Group element", ha="center", fontsize=10)
    fig.text(0.04, 0.5, "Output channel", va="center", rotation="vertical", fontsize=10)
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    visualize_kernel_weights(
        group=CyclicGroup(order=4),
        kernel_size=7,
        in_channels=3,
        out_channels=2,
    )
    visualize_kernel_weights(
        group=DihedralGroup(order=4),
        kernel_size=7,
        in_channels=3,
        out_channels=2,
    )
