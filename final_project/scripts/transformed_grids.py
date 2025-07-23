import torch
from rich import print
from src.gcnn import CyclicGroup, DihedralGroup


def transform_grid_H(group):
    grid_H = group.elements()
    inv_group_elements = torch.stack(
        [group.inverse(group_element) for group_element in grid_H]
    )

    # Compute the product for each inv_group_element with all elements in grid_H
    # Result: (num_group_elements, num_group_elements, 2)
    transformed_grid_H = torch.stack(
        [
            torch.stack([group.product(inv_g, h) for h in grid_H])
            for inv_g in inv_group_elements
        ]
    )
    # transformed_grid_H1 = group.normalize_group_parameterization(transformed_grid_H)
    # print(transformed_grid_H1)
    transformed_grid_H = torch.stack(
        [
            torch.stack([group.normalize_group_parameterization(h) for h in row_h])
            for row_h in transformed_grid_H
        ]
    )
    # print(transformed_grid_H)

    print(transformed_grid_H.shape)


def transform_grid_R2(group, kernel_size):
    grid_R2 = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, kernel_size),  # k_H
            torch.linspace(-1.0, 1.0, kernel_size),  # k_W
            indexing="ij",
        ),
        dim=0,
    )
    # print(grid_R2)
    # print(grid_R2.shape)
    group_elements = group.elements()
    inv_group_elements = torch.stack(
        [group.inverse(group_element) for group_element in group_elements]
    )
    transformed_grid_R2 = torch.stack(
        [group.left_action_on_R2(inv_g, grid_R2) for inv_g in inv_group_elements]
    ).transpose(0, 1)
    # round the values to 2 decimal places
    transformed_grid_R2 = torch.round(transformed_grid_R2, decimals=2)
    # print(transformed_grid_R2)
    print(transformed_grid_R2.shape)


def main():
    c4 = CyclicGroup(order=4)
    d4 = DihedralGroup(order=4)
    print("Cyclic Group:")
    print("Grid R2 transformed:")
    transform_grid_R2(c4, 3)
    print("Grid H transformed:")
    transform_grid_H(c4)
    print("Dihedral Group:")
    print("Grid R2 transformed:")
    transform_grid_R2(d4, 3)
    print("Grid H transformed:")
    transform_grid_H(d4)


if __name__ == "__main__":
    main()
