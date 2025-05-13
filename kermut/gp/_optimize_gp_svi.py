from typing import Tuple, Optional, Dict
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal
from omegaconf import DictConfig
import hydra
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange

from kermut.kernels import CompositeKernel


class KermutSVGP(ApproximateGP):
    """Sparse Variational Gaussian Process regression model for supervised variant effects predictions.

    A specialized Sparse Variational GP implementation that combines sequence and structural
    information for predicting the effects of protein mutations. It extends gpytorch's
    ApproximateGP class and supports both composite and single kernel architectures, as well
    as zero-shot prediction capabilities through its mean function.

    Args:
        inducing_points: Tuple of inducing points for each input type (x_toks, x_embed)
        kernel_cfg (DictConfig): Configuration dictionary for kernel specifications,
            containing settings for sequence_kernel and structure_kernel if composite
            is True, or a single kernel configuration if composite is False.
        use_zero_shot_mean (bool, optional): Whether to use a linear mean function
            for zero-shot predictions. If True, uses LinearMean; if False, uses
            ConstantMean. Defaults to True.
        composite (bool, optional): Whether to use a composite kernel combining
            sequence and structure information. If False, uses a single kernel
            specified in kernel_cfg. Defaults to True.
        **kwargs: Additional keyword arguments passed to the kernel initialization.
    """

    def __init__(
        self,
        inducing_points: Tuple[torch.Tensor, ...],
        kernel_cfg: DictConfig,
        use_zero_shot_mean: bool = True,
        composite: bool = True,
        **kwargs,
    ):
        # Initialize variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(inducing_points[0].size(-2))

        # Create a custom VariationalStrategy that handles tuple inputs
        class TupleVariationalStrategy(VariationalStrategy):
            def __init__(self, model, inducing_points, variational_distribution, **kwargs):
                # Convert tuple of inducing points to a single tensor for the base strategy
                inducing_points_tensor = torch.cat([x.flatten() for x in inducing_points])
                super().__init__(model, inducing_points_tensor, variational_distribution, **kwargs)
                self.inducing_points_tuple = inducing_points  # Store original tuple

            def forward(
                self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs
            ):
                # Handle tuple input
                if isinstance(x, tuple):
                    x_toks, x_embed = x[:2]  # Take first two elements
                    x_tensor = torch.cat([x_toks.flatten(), x_embed.flatten()])
                else:
                    x_tensor = x
                return super().forward(
                    x_tensor, inducing_points, inducing_values, variational_inducing_covar, **kwargs
                )

        variational_strategy = TupleVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # Initialize kernel
        if composite:
            self.covar_module = CompositeKernel(
                sequence_kernel=kernel_cfg.sequence_kernel,
                structure_kernel=kernel_cfg.structure_kernel,
                **kwargs,
            )
        else:
            self.covar_module = hydra.utils.instantiate(kernel_cfg.kernel, **kwargs)

        # Initialize mean function
        self.use_zero_shot_mean = use_zero_shot_mean
        if self.use_zero_shot_mean:
            self.mean_module = LinearMean(input_size=1, bias=True)
        else:
            self.mean_module = ConstantMean()

    def forward(self, x_toks, x_embed, x_zero=None) -> MultivariateNormal:
        if x_zero is None:
            x_zero = x_toks
        mean_x = self.mean_module(x_zero)
        covar_x = self.covar_module((x_toks, x_embed))
        return MultivariateNormal(mean_x, covar_x)


def optimize_gp(
    train_inputs: Tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    kernel_cfg: DictConfig,
    gp_inputs: Dict,
    n_inducing: int = 100,
    lr: float = 3.0e-4,
    n_steps: int = 150,
    batch_size: int = 100,
    use_zero_shot_mean: bool = True,
    composite: bool = True,
    progress_bar: bool = True,
) -> Tuple[KermutSVGP, GaussianLikelihood]:
    """Optimizes a KermutGP using Stochastic Variational Inference.

    Args:
        train_inputs: Tuple of input tensors for training (x_toks, x_embed, x_zero)
        train_targets: Target values tensor for training
        kernel_cfg: Configuration for the kernel(s)
        gp_inputs: Dictionary containing additional inputs for kernel initialization
            (e.g., wt_sequence, conditional_probs, coords)
        n_inducing: Number of inducing points for the sparse GP
        lr: Learning rate for the AdamW optimizer
        n_steps: Number of optimization steps
        batch_size: Size of mini-batches for training
        use_zero_shot_mean: Whether to use zero-shot mean function
        composite: Whether to use composite kernel
        progress_bar: Whether to show progress bar

    Returns:
        Tuple containing:
            - KermutSVGP: The optimized Sparse Variational GP model
            - GaussianLikelihood: The optimized likelihood function
    """
    # Filter out None inputs
    x_train = tuple([x for x in train_inputs if x is not None])

    # Initialize inducing points using random selection from first input
    inducing_points = tuple(x[torch.randperm(len(x))[:n_inducing]] for x in x_train)

    # Initialize model and likelihood
    model = KermutSVGP(
        inducing_points=inducing_points,
        kernel_cfg=kernel_cfg,
        use_zero_shot_mean=use_zero_shot_mean,
        composite=composite,
        **gp_inputs,  # Pass additional kernel inputs
    )
    likelihood = GaussianLikelihood()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        x_train = tuple(x.cuda() for x in x_train)
        train_targets = train_targets.cuda()

    # Set up data loader for mini-batch training
    train_dataset = TensorDataset(*x_train, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=lr
    )

    # Initialize variational ELBO
    mll = VariationalELBO(likelihood, model, num_data=len(train_targets))

    # Training loop
    model.train()
    likelihood.train()

    iterator = trange(n_steps, disable=not progress_bar)
    for _ in iterator:
        for batch in train_loader:
            optimizer.zero_grad()
            # Unpack batch into inputs and target
            inputs = batch[:-1]  # All but last element
            target = batch[-1]  # Last element
            # Pass only x_toks and x_embed to the model
            output = model(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()

            iterator.set_description(f"Loss: {loss.item():.3f}")

    return model, likelihood
