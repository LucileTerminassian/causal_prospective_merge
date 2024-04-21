import math
from tqdm import trange
import torch
from torch import Tensor
import torch.nn as nn


class InfoNCE(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        x_encoder: nn.Module,
        y_encoder: nn.Module,
        batch_size: int,
        scale: bool = True,
    ) -> None:
        """
        InfoNCE esitmator of mutual information between two random variables X and Y.

        Args:
            model: nn.Module with forward method that generates a batch of joint
                samples of X and Y
            x_encoder: nn.Module to encode X
            y_encoder: nn.Module to encode Y. Should encode to the same dimension
                as x_encoder
            batch_size: int, batch size to use for training
            scale: bool, whether to scale the dot product by 1/sqrt(d) where d is the
                dimension of the encoded representations;
        """
        super(InfoNCE, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.scale = scale

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """ """
        # Get the encoded representations of the data
        encoded_x = self.x_encoder(x)  # (batch_size, encoding_dim)
        encoded_y = self.y_encoder(y)  # (batch_size, encoding_dim)
        assert encoded_x.shape == encoded_y.shape, "Encoders should output same shape"

        factor = 1 / math.sqrt(encoded_y.shape[-1]) if self.scale else 1.0
        # dot prod the encodings -> (batch_size, batch_size)
        dot_product = torch.matmul(encoded_x, encoded_y.transpose(-1, -2)) * factor

        # diagonal is the positive samples (joint), logsumexp is the negative samples
        # joint - product of marginals is the MI estimate
        mi_estimate = torch.mean(
            torch.diag(dot_product) - torch.logsumexp(dot_product, dim=1)
        )
        # need to return loss, i.e. -mi_estimate. I suggest not subtracting log
        # (batch_size) in order to assess if you're hitting the log(batch_size)
        # upper bound more easily (i.e. if loss gets close to 0)
        return -mi_estimate  # - math.log(self.batch_size)

    def train_networks(self, num_steps: int, lr: float):
        """Train the encoders to tighten the InfoNCE bound"""
        optimizer = torch.optim.Adam(
            list(self.x_encoder.parameters()) + list(self.y_encoder.parameters()), lr=lr
        )
        num_steps_range = trange(1, num_steps + 1, desc="Loss: 0.000 ")
        for i in num_steps_range:
            optimizer.zero_grad()
            x, y = self.model(self.batch_size)
            loss = self(x, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                num_steps_range.set_description("Loss: {:.3f} ".format(loss))
                # Use this to check how gradients behave
                # print("grads:", list(self.x_encoder.parameters())[0].grad.norm().item())

    def estimate_mi(self, x: Tensor, y: Tensor) -> float:
        """Estimate the MI between x and y using the trained encoders"""
        assert x.shape[0] == y.shape[0], "Batch size should be the same"
        with torch.no_grad():
            loss = self(x, y).item()
        return -loss + math.log(x.shape[0])


if __name__ == "__main__":
    torch.manual_seed(42)

    # Mdel: return sample sfrom correlated 1D Gaussians:
    class GaussianModel(nn.Module):
        def __init__(self, correlation_level: float):
            super(GaussianModel, self).__init__()
            self.correlation_level = correlation_level

        def forward(self, batch_size: int) -> tuple[Tensor, Tensor]:
            """
            Generate joint samples: quantities of interest, observations
            """
            x = torch.randn(batch_size, 1)  # Sample from N(0, 1)
            # y should be correlated with x
            beta = self.correlation_level / math.sqrt(1 - self.correlation_level**2)
            y = beta * x + torch.randn(batch_size, 1)
            return x, y

        def true_mi(self) -> float:
            return -0.5 * math.log(1 - self.correlation_level**2)

    target_corr = 0.8
    model = GaussianModel(correlation_level=target_corr)
    x, y = model(1000)
    # compute correlation between x and y
    print(torch.corrcoef(torch.stack([x.squeeze(), y.squeeze()])))

    # define encoders 2 layer MLPs
    x_dim = 1
    y_dim = 1
    hidden_dim = 128
    encoding_dim = 16
    batch_size = 256
    x_encoder = nn.Sequential(
        nn.Linear(x_dim, x_dim * 4),
        nn.ReLU(),
        nn.Linear(x_dim * 4, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, encoding_dim),
    )
    y_encoder = nn.Sequential(
        nn.Linear(y_dim, y_dim * 4),
        nn.ReLU(),
        nn.Linear(y_dim * 4, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, encoding_dim),
    )
    mi_estimator = InfoNCE(
        model=model, x_encoder=x_encoder, y_encoder=y_encoder, batch_size=batch_size
    )

    print(f"Estimated MI before train: {mi_estimator.estimate_mi(x, y)}")
    # Train the encoders
    mi_estimator.train_networks(num_steps=5000, lr=3e-3)
    # Estimate the MI
    mi_estimate_end = mi_estimator.estimate_mi(x, y)
    print(f"Estimated MI: {mi_estimate_end}")
    print(f"True MI: {model.true_mi()}")
