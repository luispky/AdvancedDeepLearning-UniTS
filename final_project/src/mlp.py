import torch
import torch.nn as nn
from typing import List


class MLPClassifier(nn.Module):
    """
    Flexible Multi-Layer Perceptron for classification.

    This class implements a configurable MLP with flexible architecture,
    dropout, batch normalization, and different activation functions.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: List[int],
        dropout_rate: float = 0.0,
        use_batch_norm: bool = True,
        activation: str = "relu",
        use_bias: bool = False,
    ):
        """
        Initialize the MLP classifier.

        Args:
            input_size: Size of input features (e.g., 28*28=784 for MNIST)
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes, e.g., [512, 256, 128]
            dropout_rate: Dropout probability (0.0 = no dropout)
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'tanh', 'leaky_relu')
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        # Build the network layers
        self.flatten = nn.Flatten()
        self.layers = self._build_layers()
        self.activation_fn = self._get_activation(activation)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }

        if activation.lower() not in activations:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Supported: {list(activations.keys())}"
            )

        return activations[activation.lower()]

    def _build_layers(self) -> nn.ModuleList:
        """Build the MLP layers."""
        layers = nn.ModuleList()

        # Create list of all layer sizes
        all_sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]

        # Build layers
        for i in range(len(all_sizes) - 1):
            in_size = all_sizes[i]
            out_size = all_sizes[i + 1]

            # Linear layer
            layers.append(nn.Linear(in_size, out_size, bias=self.use_bias))

            # Don't add batch norm or dropout to the final layer
            if i < len(all_sizes) - 2:
                # Batch normalization
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_size))

                # Dropout
                if self.dropout_rate > 0.0:
                    layers.append(nn.Dropout(self.dropout_rate))

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor of shape [N, C, H, W] for images or [N, features] for vectors

        Returns:
            Output tensor of shape [N, num_classes]
        """
        # Flatten input if it's an image
        if x.dim() > 2:
            x = self.flatten(x)

        # Pass through all layers except the last one
        layer_idx = 0
        for _ in range(len(self.hidden_sizes)):
            # Linear layer
            x = self.layers[layer_idx](x)
            layer_idx += 1

            # Batch normalization
            if self.use_batch_norm:
                x = self.layers[layer_idx](x)
                layer_idx += 1

            # Activation
            x = self.activation_fn(x)

            # Dropout
            if self.dropout_rate > 0.0:
                x = self.layers[layer_idx](x)
                layer_idx += 1

        # Final linear layer (no activation for classification)
        x = self.layers[layer_idx](x)

        return x
