import torch
import numpy as np
from abc import ABC, abstractmethod

# This file is based on the following notebook:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html


class GroupBase(torch.nn.Module, ABC):
    """
    Base class for implementing group operations.

    This abstract class defines the interface for working with (finite) group operations in
    group equivariant convolutional neural networks.
    """

    def __init__(self, dimension: int, identity: list):
        """
        Initialize a group.

        :param dimension: Dimensionality of the group (number of parameters of the finite group)
        :param identity: Identity element of the group
        """
        super().__init__()
        self.dimension = dimension
        self.register_buffer("identity", torch.Tensor(identity))

    def to(self, device):
        """Override to ensure all group tensors are moved to the same device."""
        super().to(device)
        return self

    @abstractmethod
    def elements(self) -> torch.Tensor:
        """
        Obtain a tensor containing all group elements in this group.
        """
        pass

    @abstractmethod
    def numel(self) -> int:
        """
        Obtain the number of elements in the finite group
        """
        pass

    @abstractmethod
    def product(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Defines group product on two group elements.
        """
        pass

    @abstractmethod
    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        """
        Defines inverse for group element.
        """
        pass

    @abstractmethod
    def matrix_representation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Obtain a matrix representation in R^2 for an element h.
        """
        pass

    @abstractmethod
    def left_action_on_R2(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Group action of an element from the (sub)group H on a vector in R^2.
        """
        pass

    @abstractmethod
    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """
        Calculate the determinant of the representation of a group element h.
        """
        pass

    @abstractmethod
    def normalize_group_parameterization(self, h: torch.Tensor) -> torch.Tensor:
        """
        Map the group elements to an interval [-1, 1]. Used to create
        a standardized input for obtaining weights over the group.
        """
        pass


class CyclicGroup(GroupBase):
    """
    Implementation of a (parameterized) cyclic group of rotations.

    This class represents a cyclic group of rotations, such as C4 (4-fold rotations).
    e.g., the group of 90 degrees rotations of the plane
    """

    def __init__(self, order: int):
        """
        Initialize a cyclic group with specified order.

        :param order: The order of the cyclic group (number of elements)
        """
        super().__init__(
            dimension=1,  # In this case C4 is parameterized by the rotation angle
            identity=[0.0],
        )

        if order < 1:
            raise ValueError("Order must be at least 1")
        self.order = torch.tensor(order)

    def elements(self) -> torch.Tensor:
        """
        Obtain a tensor containing a parameterized representation of all group elements in this group.

        :return: Tensor containing group elements of shape [self.order]
        """
        return torch.linspace(
            start=0,
            end=2 * np.pi * float(self.order - 1) / float(self.order),
            steps=self.order,
            device=self.identity.device,
        )

    def numel(self):
        return int(self.order.item())

    def product(self, h: torch.Tensor, h_prime: torch.Tensor) -> torch.Tensor:
        """
        Defines group product on two elements of the cyclic group.
        For rotations, this is addition modulo 2π.

        :param h: Group element 1
        :param h_prime: Group element 2
        :return: Result of the group product operation
        """

        # r = a - b * a//b
        return torch.remainder(h + h_prime, 2 * np.pi)

    def inverse(self, h: torch.Tensor) -> torch.Tensor:
        """
        Defines group inverse for an element of the cyclic group.
        For rotations, this is negation modulo 2π.

        :param h: Group element
        :return: Inverse of the group element
        """
        return torch.remainder(-h, 2 * np.pi)

    def matrix_representation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Obtain a rotation matrix representation for an element h.

        :param h: A group element (rotation angle)
        :return: 2x2 rotation matrix
        """
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)

        return torch.tensor(
            [
                [cos_t, -sin_t],
                [sin_t, cos_t],
            ],
            device=self.identity.device,
        )

    def left_action_on_R2(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Group action of an element from the cyclic group on a vector in R^2.
        For rotations, this is a matrix-vector product with the rotation matrix.

        :param h: A group element from subgroup H
        :param x: Vectors in R^2
        :return: Transformed vector in R^2
        """
        return torch.tensordot(self.matrix_representation(h), x, dims=1)

    def normalize_group_parameterization(self, h: torch.Tensor) -> torch.Tensor:
        """
        Normalize values of group elements to range between -1 and 1.
        The group elements range from 0 to 2pi * (self.order - 1) / self.order,
        so we normalize accordingly.

        :param h: A group element
        :return: Normalized group element value between -1 and 1
        """
        num = 2  # 1 - (-1)
        den = 2 * np.pi * (self.order - 1) / self.order  # - 0
        return (num / den) * h + (-1.0)

    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """
        Rotation matrices in C4 <= SO(2) have determinant 1.
        """
        return torch.tensor([1.0])


class DihedralGroup(GroupBase):
    def __init__(self, order: int):
        """
        Implements the Dihedral group D_n (with 2n elements).
        Each element is parameterized by a pair [theta, s]:
          - theta: rotation angle (as in CyclicGroup)
          - s: reflection flag (0 for rotation, 1 for reflection)

        The identity is [0, 0] (i.e. no rotation and no reflection).
        """
        super().__init__(
            dimension=2,  # rotation and reflection
            identity=[0.0, 0.0],
        )
        self.order = order

    def elements(self):
        """
        Returns a tensor of shape [2*order, 2] containing all group elements.

        The first order rows are rotations: [theta, 0]
        The next order rows are reflections: [theta, 1]
        where theta is one of the order discrete angles.
        """
        polygon_sides = self.order // 2
        angles = torch.linspace(
            start=0,
            end=2 * np.pi * (polygon_sides - 1) / polygon_sides,
            steps=polygon_sides,
            device=self.identity.device,
        )
        rotations = torch.stack(
            [
                torch.tensor([theta, 0.0], device=self.identity.device)
                for theta in angles
            ]
        )
        reflections = torch.stack(
            [
                torch.tensor([theta, 1.0], device=self.identity.device)
                for theta in angles
            ]
        )
        return torch.cat(
            [rotations, reflections], dim=0
        )  # Tensor shape: [self.order, 2]

    def numel(self):
        return int(self.order)

    def product(self, h, h_prime):
        """
        Group product for D_n.

        For elements h = (theta1, s1) and h_prime = (theta2, s2),
        the product is defined as:
          (theta1, s1) * (theta2, s2) = (theta1 + (-1)^s1 * theta2 (mod 2pi), (s1 + s2) mod 2)
        """
        # todo: parallelize this?
        # group elements are small objects, so it's not worth it to parallelize
        # it just constraints to loop over the elements when making the product between multiple elements at once
        theta1, s1 = h[0], h[1]
        theta2, s2 = h_prime[0], h_prime[1]
        # Compute (-1)^s1 as 1 - 2*s1 (since s1 is 0 or 1)
        sign = 1.0 - 2.0 * s1
        new_theta = torch.remainder(theta1 + sign * theta2, 2 * np.pi)
        new_s = torch.remainder(s1 + s2, 2)
        return torch.tensor([new_theta, new_s], device=self.identity.device)

    def inverse(self, h):
        """
        Inverse for an element h = (theta, s).
        The inverse of a reflection is the reflection itself and the inverse of a rotation
        in D4 needs to be reversed according to the reflection modulo 2 * pi.

        h^-1 = ((-1)^{s+1} theta mod 2*pi, s)

        : return: Inverse of dihedral group element
        """
        # todo: parallelize this?
        theta = h[0]
        s = h[1]
        sign = 2.0 * s - 1.0
        theta_inv = torch.remainder(sign * theta, 2 * np.pi)
        return torch.tensor([theta_inv, s])

    def matrix_representation(self, h):
        """
        For h = (theta, s), return the corresponding 2x2 matrix.

        If s == 0, the representation is R(theta).
        If s == 1, it is R(theta)F, with F the fixed reflection matrix.
        """
        theta, s = h[0], h[1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        if s.item() == 0:
            return torch.tensor(
                [
                    [cos_t, -sin_t],
                    [sin_t, cos_t],
                ],
                device=self.identity.device,
            )

        return torch.tensor(
            [
                [cos_t, sin_t],
                [sin_t, cos_t],
            ],
            device=self.identity.device,
        )

    def left_action_on_R2(self, h, x):
        """
        Applies the group action of h on vector x ∈ R^2.
        """
        M = self.matrix_representation(h)
        return torch.tensordot(M, x, dims=1)

    def normalize_group_parameterization(self, h):
        """
        Normalize an element h = (theta, s) to the interval [-1, 1].

        For theta: using the same scale as in CyclicGroup.
        For s: map 0 → -1 and 1 → 1.
        """
        polygon_sides = self.order // 2
        num_t = 2  # 1 - (-1)
        den_t = 2 * np.pi * (polygon_sides - 1) / polygon_sides  # - 0
        normalized_theta = num_t / den_t * h[0] + (-1.0)
        normalized_s = 2 * h[1] - 1
        return torch.tensor(
            [normalized_theta, normalized_s], device=self.identity.device
        )

    def determinant(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes the determinant of the matrix representation of h.

        :param h: group element of D_n in parametric form
        :return: determinant of h
        """
        return torch.det(self.matrix_representation(h))


def general_grid_shift(
    grid: torch.Tensor, skip_permute: bool = False, reverse_axis: int = -1
) -> torch.Tensor:
    """
    Reorder the coordinate axes of a grid tensor for use with torch.nn.functional.grid_sample.

    This function takes a grid tensor of shape (coord_dim, ...), where the first axis is the coordinate dimension
    (e.g., 2 for (x, y), 3 for (x, y, z)), and moves the coordinate axis to the end, then reverses the order of the
    coordinates. This is required because grid_sample expects the last axis to be the coordinate dimension, and the
    order should be (y, x) for 2D or (z, y, x) for 3D, matching the memory layout of PyTorch tensors.

    IMPORTANT:
        - The permutation performed here is not fully general: it always moves the first axis (axis 0, typically the coordinate axis)
          to the last position, i.e., (2, H, W) -> (H, W, 2) or (3, D, H, W) -> (D, H, W, 3).
        - The reverse_axis argument only applies to a specific dimension, typically the first or last one in this code.
          For example, reverse_axis=0 reverses the first axis, reverse_axis=-1 reverses the last axis (the coordinate axis after permutation).
        - This is required for PyTorch's grid_sample, which expects the last axis to be the coordinate dimension, and the order
          to match the memory layout of the tensor (e.g., (y, x) for 2D, (z, y, x) for 3D).

    The permutation is performed by moving axis 0 (the coordinate axis) to the end, i.e.,
        grid.permute(1, 2, ..., N, 0)
    which is equivalent to:
        grid.permute(*range(1, grid.ndim), 0)
    The coordinate reversal is done by:
        grid[..., list(reversed(range(grid.shape[-1])))]
    which reverses the last axis (e.g., (x, y) -> (y, x), (x, y, z) -> (z, y, x)).

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


def interpolate_signal(
    signal: torch.Tensor, grid: torch.Tensor, skip_permute: bool = False
) -> torch.Tensor:
    """
    Interpolate values from an input signal at specified grid coordinates,
    supporting both 2D and 3D signals.

    This function prepares the grid for torch.nn.functional.grid_sample by permuting and reversing
    the coordinate axes as needed, so that the grid matches the expected memory layout of PyTorch tensors.
    Optionally, you can skip the permutation and only reverse the coordinate axes by setting skip_permute=True.

    Args:
        signal (torch.Tensor): Tensor representing an "image" or (multi-dimensional) feature map.
        grid (torch.Tensor): Tensor of coordinate points from which to sample the signal.
            2D:
                - signal: (C, H, W) or (N, C, H, W)
                - grid: (2, H, W) or (2, N, H, W) [x, y]
            3D:
                - signal: (C, D, H, W) or (N, C, D, H, W)
                - grid: (3, D, H, W) or (3, N, D, H, W) [x, y, z]
        skip_permute (bool): If True, skip the permutation and only reverse the coordinate axes. Default: False.

    In 2D, the function performs bilinear interpolation.
    Bilinear interpolation computes each output value as a weighted average
    of the four nearest
    input pixels based on the fractional grid coordinates.
    Reference: https://en.wikipedia.org/wiki/Bilinear_interpolation

    Returns:
        torch.Tensor: Tensor with interpolated values from the input signal at the grid points.
    """
    # ----------------------------------------
    # Indexing in PyTorch image/volume tensors
    # ----------------------------------------
    # For 2D images:
    #   Tensor shape: (C, H, W)
    #   Indexing: signal[c, y, x]
    # For 3D volumes:
    #   Tensor shape: (C, D, H, W)
    #   Indexing: signal[c, z, y, x]
    # This reflects the memory layout where the first spatial axis is depth (z),
    # then height (y), then width (x). This is why [x, y] or [x, y, z] user input
    # must be reordered to [y, x] or [z, y, x] respectively before sampling.

    coord_dim = grid.shape[0]  # Number of coordinate axes (2 for 2D, 3 for 3D)

    # Add batch dimension if missing.
    if grid.ndim == coord_dim + 1:
        grid = grid.unsqueeze(1)  # e.g., (2, H, W) -> (2, N, H, W)
    if (coord_dim == 2 and signal.ndim == 3) or (coord_dim == 3 and signal.ndim == 4):
        signal = signal.unsqueeze(0)  # e.g., (C, H, W) -> (N, C, H, W)

    # Generalized permutation and coordinate reversal for grid_sample
    # By default, permute axes and reverse coordinates; if skip_permute=True, only reverse coordinates
    grid = general_grid_shift(grid, skip_permute=False, reverse_axis=-1)

    # Perform bilinear or trilinear interpolation using grid_sample.
    # - 'padding_mode="zeros"' ensures that coordinates outside the signal return 0.
    # - 'align_corners=True' ensures the grid aligns with the signal corners.
    # - 'mode="bilinear"' is bilinear for 2D and trilinear for 3D.
    return torch.nn.functional.grid_sample(
        signal,
        grid,
        padding_mode="zeros",
        align_corners=True,
        mode="bilinear",
    )


class BaseKernel(torch.nn.Module, ABC):
    """
    Base class for all kernels.
    """

    def __init__(self, group, kernel_size: int, in_channels: int, out_channels: int):
        """
        Initialize a lifting kernel.

        :param group: Group implementation
        :param kernel_size: Size of the kernel
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super().__init__()
        self.group = group
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create spatial grid over which the kernel is defined
        # (2, ks, ks)
        self.register_buffer("grid_R2", self._create_grid_R2())

    def _create_grid_R2(self) -> torch.Tensor:
        """
        Create a grid over R^2.

        :return: Tensor containing grid over R^2, shape [2, kernel_size, kernel_size]
        with [x, y] coordinates
        """
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )
        return torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, self.kernel_size, device=device),  # k_H
                torch.linspace(-1.0, 1.0, self.kernel_size, device=device),  # k_W
                indexing="ij",
            ),
            dim=0,
        )

    @abstractmethod
    def _create_transformed_grid(self) -> torch.Tensor:
        """
        Create a transformed grid over R^2 or R^2 ⋊ H with the group elements.

        :return: Tensor containing transformed grid over R^2 or R^2 ⋊ H
        """
        pass

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample the group transformed kernels.
        """
        pass


class InterpolativeLiftingKernel(BaseKernel):
    """
    Implementation of a kernel that lifts from the spatial domain to the group domain.
    """

    def __init__(self, group, kernel_size: int, in_channels: int, out_channels: int):
        """
        Initialize a lifting kernel.

        :param group: Group implementation
        :param kernel_size: Size of the kernel
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super().__init__(group, kernel_size, in_channels, out_channels)

        # Transform the grid by the elements in the group
        # It stores transformed **copies** of the grid, over which the kernel is defined
        # (2, |H|, ks, ks)
        self.register_buffer("transformed_grid_R2", self._create_transformed_grid())

        # Create and initialise a set of weights over the spatial domain (x, y)
        self.weight = torch.nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            ),
        )

        # Initialize weights using kaiming uniform intialisation.
        torch.nn.init.kaiming_uniform_(self.weight.data, a=np.sqrt(5))

    def _create_transformed_grid(self) -> torch.Tensor:
        """
        Transform the created grid over R^2 with the group action of each element in H.
        This lifts and transforms the domain of feature maps from R^2 to G = R^2 ⋊ H
        between convolutions.

        :return: Tensor containing transformed grids
        """
        group_elements = self.group.elements()
        inv_group_elements = torch.stack(
            [self.group.inverse(group_element) for group_element in group_elements]
        )

        # For 2D inputs we need a grid of size: (2, H_g, W_g) for each h in H
        transformed_grids = torch.stack(
            [
                # (2, 2) x (2, ks, ks) -> (2, ks, ks) |H| times
                self.group.left_action_on_R2(inv_g, self.grid_R2)
                for inv_g in inv_group_elements
            ]
        ).transpose(0, 1)
        # (2, |H|, ks, ks)
        return transformed_grids

    def sample(self) -> torch.Tensor:
        """
        Sample the group transformed kernels.

        :return: Transformed kernel values
        """
        # The 2D kernel weights are sampled with the transformed grids
        # The weights must have size: (N, C, W, H) or **(C, W, H)**
        # The grids must have size: (2, N, H, W) or **(2, H, W)**

        # (out_C, in_C, ks, ks) -> (out_C * in_C, ks, ks)
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        # Sample the transformed 2D kernels along the H dimension
        transformed_weight = torch.stack(
            [
                general_grid_shift(
                    interpolate_signal(
                        # (out_C * in_C, ks, ks)
                        weight,
                        # (2, ks, ks) for each h in H
                        self.transformed_grid_R2[:, grid_idx, :, :],
                    ),  # (C, H_out, W_out)
                    skip_permute=True,
                    reverse_axis=0,
                )
                for grid_idx in range(self.group.numel())
            ],
            dim=0,
        )
        # (|H|, out_C * in_C, ks, ks)
        transformed_weight = transformed_weight.view(
            self.group.numel(),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        # (out_C, |H|, in_C, ks, ks) for 2D convolution
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight


class InterpolativeGroupKernel(BaseKernel):
    """
    Implementation of a kernel for group convolution.
    """

    def __init__(self, group, kernel_size: int, in_channels: int, out_channels: int):
        """
        Initialize a group kernel.

        :param group: Group implementation
        :param kernel_size: Size of the kernel
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super().__init__(group, kernel_size, in_channels, out_channels)

        # Create the spatial and subgroup grids over which the kernels are defined
        self.register_buffer("grid_H", self.group.elements())
        # Transforms the spatial and subgroup grids into copies for each group element
        self.register_buffer("transformed_grid_R2xH", self._create_transformed_grid())

        # Create and initialise a set of weights over the group domain (x, y, h)
        self.weight = torch.nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels,
                self.group.numel(),
                kernel_size,
                kernel_size,
            ),
        ).to(device=self.group.identity.device)

        # Initialize weights using kaiming uniform intialisation.
        torch.nn.init.kaiming_uniform_(self.weight.data, a=np.sqrt(5))

    def _create_transformed_grid(self) -> torch.Tensor:
        """
        Transforms the individual grids over R^2 and H with the group elements and
        creates a unique transformed grid over the entire group G = R^2 ⋊ H.
        This transforms the domain of the kernels defined over the group.
        For SE(2), the grid rotates over the spatial dimensions, and shifts over the H dimension.

        :return: Tensor containing transformed grids
        """
        group_elements = self.group.elements()
        inv_group_elements = torch.stack(
            [self.group.inverse(group_element) for group_element in group_elements]
        )

        # Transform the grid defined over R^2 with the group elements
        transformed_grid_R2 = torch.stack(
            [
                # (2, ks, ks) for each h in H
                self.group.left_action_on_R2(g_inverse, self.grid_R2)
                for g_inverse in inv_group_elements
            ]
        ).transpose(0, 1)
        # (2, |H|, ks, ks)

        # Transform the grid defined over H with the group elements
        transformed_grid_H = torch.stack(
            [
                # (|H|) for each h in H
                torch.stack([self.group.product(inv_g, h) for h in self.grid_H])
                for inv_g in inv_group_elements
            ]
        )
        # (|H|, |H|) per group element representation
        # here, Cn is represented with just 1 value
        # but for Dn, it is represented with 2 values, 2 dimensions
        # and thus the grid is (|H|, |H|, 2)

        try:
            transformed_grid_H = self.group.normalize_group_parameterization(
                transformed_grid_H
            )
        except Exception:
            try:
                transformed_grid_H = torch.stack(
                    [
                        torch.stack(
                            [
                                self.group.normalize_group_parameterization(h)
                                for h in row_h
                            ]
                        )
                        for row_h in transformed_grid_H
                    ]
                )
            except Exception:
                raise NotImplementedError(
                    "Group representation does not support correct normalization for transformed_grid_H."
                )

        # The complete kernel grid is the product of the grids over R^2 and H (i.e., grid over R^2 x H)
        # For 3D inputs (x, y, h) we need a grid of size: (3, D_g, H_g, W_g) for each h in H
        # ! Here the DihedralGroup is not supported because it uses a 2D representation of the group elements
        return torch.cat(
            (
                # (2, |H|, ks, ks) -> (2, |H|, 1, ks, ks) -> (2, |H|, |H|, ks, ks)
                transformed_grid_R2.view(
                    2,
                    self.group.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size,
                ).repeat(1, 1, self.group.numel(), 1, 1),
                # (|H|, |H|) -> (1, |H|, |H|, 1, 1) -> (1, |H|, |H|, ks, ks)
                transformed_grid_H.view(
                    1,
                    self.group.numel(),
                    self.group.numel(),
                    1,
                    1,
                ).repeat(1, 1, 1, self.kernel_size, self.kernel_size),
            ),
            dim=0,
        )  # (3, |H|, |H|, ks, ks) with 2 spatial dimensions and 1 H group dimension

    def sample(self) -> torch.Tensor:
        """
        Sample the group transformed kernels.

        :return: Transformed kernel values
        """
        # The 3D kernel weights are sampled with the transformed grids
        # The weights must have size: (N, C, D, W, H) or **(C, D, H, W)**
        # The grids must have size: (3, N, D, H, W) or **(3, D, H, W)**

        # (out_C, in_C, |H|, ks, ks) -> (out_C * in_C, |H| ks, ks)
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.group.numel(),
            self.kernel_size,
            self.kernel_size,
        )

        # Sample the transformed 3D kernels along the H dimension
        # |H| tensors of shape (out_C * in_C, |H|, ks, ks)
        # ! NO GO FOR THE DIHEDRAL GROUP, FOR THE MOMENT
        transformed_weight = torch.stack(
            [
                general_grid_shift(
                    interpolate_signal(
                        # (out_C * in_C, |H|, ks, ks)
                        weight,
                        # (3, |H|, ks, ks) for each group element in H
                        self.transformed_grid_R2xH[:, grid_idx, :, :, :],
                    ),  # (C, D, H_out, W_out)
                    skip_permute=False,
                    reverse_axis=0,
                )
                for grid_idx in range(self.group.numel())
            ],
            dim=0,
        )  # (|H|, out_C * in_C, |H|, ks, ks)

        # (|H|, out_C, in_C, |H|, ks, ks)
        transformed_weight = transformed_weight.view(
            self.group.numel(),
            self.out_channels,
            self.in_channels,
            self.group.numel(),
            self.kernel_size,
            self.kernel_size,
        )
        # (out_C, |H|, in_C, |H|, ks, ks)
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight


class LiftingConvolution(torch.nn.Module):
    """
    Implementation of a lifting convolution layer.

    This layer lifts a standard CNN feature map to a group-equivariant feature map.
    """

    def __init__(
        self,
        group,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        """
        Initialize a lifting convolution layer.

        :param group: Group implementation
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param padding: Padding to apply to the input
        :param stride: Stride to apply to the input
        """
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # Create the lifting kernel
        self.kernel = InterpolativeLiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform lifting convolution.

        :param x: Input tensor of shape [B, in_C, in_H, in_W]
        :return: Output tensor of shape [B, out_C, |H|, out_H, out_W]
        """
        # Obtain convolution kernels transformed under the group
        # (out_C, |H|, in_C, ks, ks)
        conv_kernels = self.kernel.sample()

        # Apply lifting convolution
        # We fold the group dimension into the output channel dimension in the weights
        # to use torch.nn.functional.conv2d and because that's the dimension the
        # input is lifted to.
        # We can do this because the group dimension is the same for all the input channels
        x = torch.nn.functional.conv2d(
            # (B, in_C, in_H, in_W)
            input=x,
            # (out_C * |H|, in_C, kernel_size, kernel_size) ~ (out_C, in_C, ks, ks)
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.group.numel(),
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size,
            ),
            padding=self.padding,
            stride=self.stride,
        )  # (B, out_C * |H|, out_H, out_W)
        # out_H = (in_H + 2 * padding - kernel_size) / stride + 1
        # out_W = (in_W + 2 * padding - kernel_size) / stride + 1

        # (B, out_C * |H|, out_H, out_W) -> (B, out_C, |H|, out_H, out_W)
        # Now the input is lifted to the group domain
        return x.view(
            -1,
            self.kernel.out_channels,
            self.group.numel(),
            x.shape[-2],
            x.shape[-1],
        )


class GroupConvolution(torch.nn.Module):
    """
    Implementation of a group convolution layer.

    This layer performs convolution that preserves group equivariance.
    """

    def __init__(
        self,
        group,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        """
        Initialize a group convolution layer.

        :param group: Group implementation
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param padding: Padding to apply to the input
        :param stride: Stride to apply to the input
        """
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # Create the kernel
        self.kernel = InterpolativeGroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform group convolution.

        :param x: Input tensor of shape [B, in_C, |H|, in_H, in_W]
        :return: Output tensor of shape [B, out_C, |H|, out_H, out_W]
        """
        # In order to use torch.nn.functional.conv2d, we need to fold the group
        # dimension into the input channel dimension since the input is in the group domain
        # (B, in_C, |H|, in_H, in_W) -> (B, in_C * |H|, in_H, in_W)
        x = x.reshape(-1, x.shape[-4] * x.shape[-3], x.shape[-2], x.shape[-1])

        # We obtain convolution kernels transformed under the group
        # (out_C, |H|, in_C, |H|, ks, ks)
        conv_kernels = self.kernel.sample()

        # Apply group convolution
        # We fold the group dimension into the input channel dimension and into the
        # output channel dimension of the weights to use torch.nn.functional.conv2d
        x = torch.nn.functional.conv2d(
            # (B, in_C * |H|, in_H, in_W) ~ (B, in_C, H, W)
            input=x,
            # (out_C * |H|, in_C * |H|, k_H, k_W) ~ (out_C, in_C, ks, ks)
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.group.numel(),
                self.kernel.in_channels * self.group.numel(),
                self.kernel.kernel_size,
                self.kernel.kernel_size,
            ),
            padding=self.padding,
            stride=self.stride,
        )  # (B, out_C * |H|, out_H, out_W)
        # out_H = (in_H + 2 * padding - kernel_size) / stride + 1
        # out_W = (in_W + 2 * padding - kernel_size) / stride + 1

        # (B, out_C * |H|, out_H, out_W) -> (B, out_C, |H|, out_H, out_W)
        return x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.numel(),
            x.shape[-2],
            x.shape[-1],
        )


class GroupEquivariantCNNClassifier(torch.nn.Module):
    """
    Implementation of a group equivariant CNN.

    This network is equivariant with respect to the specified group transformations.
    """

    def __init__(
        self,
        group,
        in_channels: int,
        num_classes: int,
        kernel_size: int,
        num_hidden: int,
        hidden_channels: int,
        padding: int = 0,
        stride: int = 1,
    ):
        """
        Initialize a group equivariant CNN.

        :param group: Group implementation
        :param in_channels: Number of input channels
        :param num_classes: Number of output classes
        :param kernel_size: Size of the kernel
        :param num_hidden: Number of hidden layers
        :param hidden_channels: Number of channels in hidden layers
        :param padding: Padding to apply to the input
        :param stride: Stride to apply to the input
        """
        super().__init__()

        # Store num_classes for compatibility with lightning model
        self.num_classes = num_classes

        # Create the lifting convolution
        self.lifting_conv = LiftingConvolution(
            group=group,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        # Create a set of group convolutions
        self.gconvs = torch.nn.ModuleList()
        self.gconvs.extend(
            GroupConvolution(
                group=group,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            for _ in range(num_hidden)
        )

        # Create the projection layer
        # (N, hidden_channels, |H|, in_H, in_W) -> (N, hidden_channels, 1, 1, 1)
        self.projection_layer = torch.nn.AdaptiveAvgPool3d(1)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the group equivariant CNN.

        :param x: Input tensor of shape [N, C, H, W]
        :return: Output tensor of shape [N, num_classes]
        """
        # Lift and disentangle features in the input
        # (N, in_C, in_H, in_W) -> (N, hidden_C, |H|, out_H_lc, out_W_lc)
        x = self.lifting_conv(x)

        # Normalize each element of the batch over (hidden_C, |H|, out_H_lc, out_W_lc)
        x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)

        # Apply group convolutions
        for gconv in self.gconvs:
            x = gconv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)
        # (N, hidden_C, |H|, out_H_lc, out_W_lc) ->
        # -> (N, hidden_C, |H|, out_H_gc_i, out_W_gc_i) for i=1,...,num_hidden
        # -> (N, hidden_C, |H|, out_H, out_W)
        # out_H = (in_H + 2 * padding - kernel_size) / stride + 1
        # out_W = (in_W + 2 * padding - kernel_size) / stride + 1

        # To ensure invariance, apply max pooling over group and spatial dims
        # (N, hidden_C, |H|, out_H, out_W) -> (N, hidden_C, 1, 1, 1) -> (N, hidden_C)
        x = self.projection_layer(x).squeeze()

        # (N, hidden_C) -> (N, num_classes)
        return self.classifier(x)

    def to(self, device):
        """Override to ensure all components are moved to the same device."""
        super().to(device)
        # Ensure the group is also moved to the device
        if hasattr(self, "lifting_conv") and hasattr(self.lifting_conv, "group"):
            self.lifting_conv.group.to(device)
        return self
