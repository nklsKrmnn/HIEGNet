from torch import nn
import torch


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        """
        Weighted mean squared error loss.

        Args:
            class_weights (torch.Tensor): Class weights.
        """
        super().__init__()
        self.register_buffer('class_weights', class_weights)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted mean squared error loss.

        Computes nn.MSELoss with a weighting factor for each class.

        Args:
            output (torch.Tensor): Output tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Weighted mean squared error loss.
        """
        # Compute the mean squared error
        mse = nn.MSELoss(reduction='none')(output, target)

        # Get the weights based on the target values
        weights = self.class_weights[target.long()]

        # Apply class weights
        weighted_mse = mse * weights

        return weighted_mse.mean()


