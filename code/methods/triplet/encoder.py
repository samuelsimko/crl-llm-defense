import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    An encoder class consisting of num_encoders independent residual MLP encoders.

    Each independent encoder consists of n_layer residual 2-layer MLP blocks
    followed by a final projection.

    Takes an input tensor of shape (batch_size, num_encoders, input_dim)
    and outputs a tensor of shape (batch_size, num_encoders, output_dim),
    where output_dim <= input_dim. Each 'num_encoders' slice is processed
    by its dedicated, independent encoder.

    Args:
        num_encoders (int): The number of independent encoders to create.
        input_dim (int): The dimensionality of the input representation (d) for each encoder.
        output_dim (int): The dimensionality of the output representation (k) for each encoder, must be <= input_dim.
        num_layers (int): The number of residual MLP layers (n_layer) within each independent encoder.
        mlp_ratio (int): The expansion factor for the hidden layer in each 2-layer MLP.
                         E.g., if 4, the MLP maps d -> 4*d -> d.
    """

    def __init__(
        self,
        num_encoders: int,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        if output_dim > input_dim:
            raise ValueError(
                f"Output dimension ({output_dim}) must be less than or equal to input dimension ({input_dim})"
            )

        self.num_encoders = num_encoders
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio

        # Create a ModuleList to hold num_encoders independent encoder blocks
        self.encoders = nn.ModuleList()
        for _ in range(num_encoders):
            # Each element in the list is a complete encoder for one 'num_encoders' level
            encoder_layers = nn.ModuleList()
            for _ in range(num_layers):
                # Each layer is a 2-layer MLP with a residual connection
                mlp_block = nn.Sequential(
                    nn.Linear(input_dim, input_dim * mlp_ratio),
                    nn.ReLU(),  # Activation function
                    nn.Linear(input_dim * mlp_ratio, input_dim),
                )
                encoder_layers.append(mlp_block)

            # Final projection for this specific encoder
            final_projection = nn.Linear(input_dim, output_dim)

            # Store the layers and final projection for this encoder level
            # We wrap them in a Sequential or a custom Module if needed,
            # but for simplicity, we can just store the list of layers and the projection
            # and apply them sequentially in the forward pass.
            # A cleaner way is to make a sub-module for a single encoder block:
            class SingleEncoderBlock(nn.Module):
                def __init__(self, input_dim, output_dim, num_layers, mlp_ratio):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    for _ in range(num_layers):
                        mlp_block = nn.Sequential(
                            nn.Linear(input_dim, input_dim * mlp_ratio),
                            nn.ReLU(),
                            nn.Linear(input_dim * mlp_ratio, input_dim),
                        )
                        self.layers.append(mlp_block)
                    self.final_projection = nn.Linear(input_dim, output_dim)

                def forward(self, x):
                    for layer in self.layers:
                        x = x + layer(x)
                    return self.final_projection(x)

            self.encoders.append(
                SingleEncoderBlock(input_dim, output_dim, num_layers, mlp_ratio)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the independent encoders.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_encoders, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_encoders, output_dim).
        """
        # The input shape is now expected to be (batch_size, num_encoders, input_dim)
        if (
            x.ndim != 3
            or x.shape[-1] != self.input_dim
            or x.shape[1] != self.num_encoders
        ):
            raise ValueError(
                f"Input tensor shape must be (batch_size, num_encoders, input_dim), but got {x.shape}. Expected num_encoders={self.num_encoders}"
            )

        outputs = []
        # Iterate through each encoder level and apply the corresponding independent encoder
        for i in range(self.num_encoders):
            # Select the slice for the i-th encoder level across all batch items
            input_slice = x[:, i, :]  # Shape: (batch_size, input_dim)

            # Pass the slice through the i-th independent encoder block
            output_slice = self.encoders[i](
                input_slice
            )  # Shape: (batch_size, output_dim)

            outputs.append(output_slice)

        # Stack the outputs from each encoder back along the num_encoders dimension
        output = torch.stack(
            outputs, dim=1
        )  # Shape: (batch_size, num_encoders, output_dim)

        return output
