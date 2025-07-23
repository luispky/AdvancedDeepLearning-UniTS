import torch


class CNNClassifier(torch.nn.Module):
    """
    Standard convolutional neural network.

    This class implements a standard CNN for comparison with group equivariant CNNs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int,
        num_hidden: int,
        hidden_channels: int,
        padding: int = 0,
        stride: int = 1,
    ):
        """
        Initialize a standard CNN.

        :param in_channels: Number of input channels
        :param num_classes: Number of output classes
        :param kernel_size: Size of the kernel
        :param num_hidden: Number of hidden layers
        :param hidden_channels: Number of channels in hidden layers
        :param padding: Padding to apply to the input
        :param stride: Stride to apply to the input
        """
        super().__init__()

        self.num_classes = num_classes  # Add this for compatibility with Classifier

        self.first_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        self.convs = torch.nn.ModuleList()
        self.convs.extend(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            for _ in range(num_hidden)
        )

        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        :param x: Input tensor of shape [N, C, H, W]
        :return: Output tensor of shape [N, num_classes]
        """
        # (N, in_C, in_H, in_W) -> (N, hidden_C, out_H_fc, out_W_fc)
        x = self.first_conv(x)
        x = torch.nn.functional.layer_norm(x, x.shape[-3:])
        x = torch.nn.functional.relu(x)

        for conv in self.convs:
            x = conv(x)
            x = torch.nn.functional.layer_norm(x, x.shape[-3:])
            x = torch.nn.functional.relu(x)
        # (N, hidden_C, out_H_fc, out_W_fc) ->
        # -> (N, hidden_C, out_H_ic, out_W_ic) for i=1,...,num_hidden
        # -> (N, hidden_C, out_H, out_W)
        # out_H = (in_H + 2 * padding - kernel_size) / stride + 1
        # out_W = (in_W + 2 * padding - kernel_size) / stride + 1

        # Apply average pooling over remaining spatial dimensions
        # (N, hidden_C, out_H, out_W) -> (N, hidden_C)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).squeeze()

        # (N, hidden_C) -> (N, num_classes)
        return self.classifier(x)
