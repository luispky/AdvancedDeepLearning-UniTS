import matplotlib.pyplot as plt
from src.gcnn import CyclicGroup, DihedralGroup, InterpolativeGroupKernel
import matplotlib.patches as mpatches


def visualize_group_kernel_weights(
    group,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    out_channel_idx: int = 0,
):
    """
    Visualize the sampled group kernel weights for a given group, kernel size, and channel configuration.
    Each subplot shows the kernel for a specific input channel and group element for a fixed output channel.
    The input group dimension is folded into the spatial x dimension for visualization.
    """
    igk = InterpolativeGroupKernel(
        group=group,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    weights = (
        igk.sample()
    )  # shape: (out_channels, group_elements, in_channels, group_elements, ks, ks)
    num_group_elements = group.numel()

    # Fold the input group dimension into the spatial x dimension for visualization
    weights_t = weights.view(
        igk.out_channels,
        num_group_elements,
        igk.in_channels,
        num_group_elements * igk.kernel_size,
        igk.kernel_size,
    )

    fig, ax = plt.subplots(
        igk.in_channels,
        num_group_elements,
        figsize=(1.2 * num_group_elements, 1.2 * igk.in_channels),
        squeeze=False,
    )

    for in_channel in range(igk.in_channels):
        for group_elem in range(num_group_elements):
            ax[in_channel, group_elem].imshow(
                weights_t[out_channel_idx, group_elem, in_channel, :, :]
                .detach()
                .numpy(),
                cmap="viridis",
            )
            # Outline the spatial kernel corresponding to the first group element under canonical transformation
            rect = mpatches.Rectangle(
                (-0.5, group_elem * weights_t.shape[-1] - 0.5),
                weights_t.shape[-1],
                weights_t.shape[-1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax[in_channel, group_elem].add_patch(rect)
            ax[in_channel, group_elem].set_xticks([])
            ax[in_channel, group_elem].set_yticks([])
            if in_channel == 0:
                ax[in_channel, group_elem].set_title(f"g={group_elem}")
            if group_elem == 0:
                ax[in_channel, group_elem].set_ylabel(f"in={in_channel}")

    fig.text(0.5, 0.04, "Group elements", ha="center")
    fig.text(
        0.04,
        0.5,
        "Input channels / input group elements",
        va="center",
        rotation="vertical",
    )
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0)
    plt.show()


if __name__ == "__main__":
    visualize_group_kernel_weights(
        group=CyclicGroup(order=4),
        kernel_size=5,
        in_channels=2,
        out_channels=8,
        out_channel_idx=0,
    )
    # visualize_group_kernel_weights(
    #     group=DihedralGroup(order=4),
    #     kernel_size=5,
    #     in_channels=2,
    #     out_channels=8,
    #     out_channel_idx=0,
    # )
