import torch
from rich import print


# General grid shift function
def general_grid_shift(grid, skip_permute=False, reverse_axis=-1):
    """
    Reorder the coordinate axes of a grid tensor for use with torch.nn.functional.grid_sample.

    Note:
        - The permutation performed here is not fully general: it always moves the first axis (axis 0, typically the coordinate axis)
          to the last position, i.e., (2, H, W) -> (H, W, 2) or (3, D, H, W) -> (D, H, W, 3).
        - The reverse_axis argument only applies to a specific dimension, typically the first or last one in this code.
          For example, reverse_axis=0 reverses the first axis, reverse_axis=-1 reverses the last axis.

    Args:
        grid (torch.Tensor): The grid tensor to reorder.
        skip_permute (bool): If True, do not permute axes, only reverse the specified axis. Default: False.
        reverse_axis (int or None): Which axis to reverse. Default: -1 (last axis). If None, do not reverse any axis.

    Returns:
        torch.Tensor: The reordered grid tensor, ready for grid_sample.
    """
    if not skip_permute:
        # Move axis 0 (coordinate axis) to the end
        grid = grid.permute(*range(1, grid.ndim), 0)
    if reverse_axis is not None:
        # Reverse the specified axis
        idx = [slice(None)] * grid.ndim
        idx[reverse_axis] = list(reversed(range(grid.shape[reverse_axis])))
        grid = grid[tuple(idx)]
    return grid


# 2D Example: grid shape (2, H, W)
grid_2d = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, 3),  # x
        torch.linspace(-1, 1, 3),  # y
        indexing="ij",
    ),
    dim=0,
)  # shape (2, 3, 3)
print("[bold yellow]Original 2D grid (2, H, W): shape:[/bold yellow]", grid_2d.shape)
print(grid_2d)

# Option 1: skip_permute=True, reverse_axis=0, then permute to (H, W, 2)
shifted_2d_skip = general_grid_shift(grid_2d, skip_permute=True, reverse_axis=0)
shifted_2d_skip_perm = shifted_2d_skip.permute(1, 2, 0)
print(
    "\n[bold cyan]2D Option 1: skip_permute=True, reverse_axis=0, then permute to (H, W, 2): shape:[/bold cyan]",
    shifted_2d_skip_perm.shape,
)
print(shifted_2d_skip_perm)

# Option 2: skip_permute=False, reverse_axis=-1 (default), direct to (H, W, 2)
shifted_2d = general_grid_shift(grid_2d, skip_permute=False, reverse_axis=-1)
print(
    "\n[bold magenta]2D Option 2: skip_permute=False, reverse_axis=-1 (direct to (H, W, 2)): shape:[/bold magenta]",
    shifted_2d.shape,
)
print(shifted_2d)
print(
    "[green]2D Option 1 == Option 2:[/green]",
    torch.allclose(shifted_2d_skip_perm, shifted_2d),
)

# 3D Example: grid shape (3, D, H, W)
grid_3d = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, 2),  # x
        torch.linspace(-1, 1, 2),  # y
        torch.linspace(-1, 1, 2),  # z
        indexing="ij",
    ),
    dim=0,
)  # shape (3, 2, 2, 2)
print(
    "\n[bold yellow]Original 3D grid (3, D, H, W): shape:[/bold yellow]", grid_3d.shape
)
print(grid_3d)

# Option 1: skip_permute=True, reverse_axis=0, then permute to (D, H, W, 3)
shifted_3d_skip = general_grid_shift(grid_3d, skip_permute=True, reverse_axis=0)
shifted_3d_skip_perm = shifted_3d_skip.permute(1, 2, 3, 0)
print(
    "\n[bold cyan]3D Option 1: skip_permute=True, reverse_axis=0, then permute to (D, H, W, 3): shape:[/bold cyan]",
    shifted_3d_skip_perm.shape,
)
print(shifted_3d_skip_perm)

# Option 2: skip_permute=False, reverse_axis=-1 (default), direct to (D, H, W, 3)
shifted_3d = general_grid_shift(grid_3d, skip_permute=False, reverse_axis=-1)
print(
    "\n[bold magenta]3D Option 2: skip_permute=False, reverse_axis=-1 (direct to (D, H, W, 3)): shape:[/bold magenta]",
    shifted_3d.shape,
)
print(shifted_3d)
print(
    "[green]3D Option 1 == Option 2:[/green]",
    torch.allclose(shifted_3d_skip_perm, shifted_3d),
)
