from src.gcnn import CyclicGroup, DihedralGroup, GroupEquivariantCNNClassifier


def create_group(group_type: str, group_order: int):
    """
    Create a group instance based on the specified type and order.

    Args:
        group_type: Type of group ('cyclic' or 'dihedral')
        group_order: Order of the group

    Returns:
        Group instance

    Raises:
        ValueError: If group_type is 'dihedral' (not supported yet)
        ValueError: If group_type is invalid
    """
    if group_type.lower() == "cyclic":
        return CyclicGroup(order=group_order)
    elif group_type.lower() == "dihedral":
        raise ValueError("❌ Dihedral groups are not supported yet")
    else:
        raise ValueError(f"❌ Unknown group type: {group_type}")


def build_gcnn_model(dataset_config, model_config):
    """
    Build GCNN model with specified configuration.

    Args:
        dataset_config: Dataset configuration dictionary
        model_config: Model configuration dictionary

    Returns:
        GroupEquivariantCNNClassifier instance
    """
    group = create_group(
        group_type=model_config["group_type"], group_order=model_config["group_order"]
    )

    return GroupEquivariantCNNClassifier(
        group=group,
        in_channels=dataset_config["in_channels"],
        num_classes=dataset_config["num_classes"],
        kernel_size=model_config["kernel_size"],
        num_hidden=model_config["num_hidden"],
        hidden_channels=model_config["hidden_channels"],
        padding=model_config["padding"],
        stride=model_config["stride"],
    )
