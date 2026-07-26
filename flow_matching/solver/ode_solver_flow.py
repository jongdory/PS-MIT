# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union
from enum import Enum
import numpy as np
import torch
from torch import Tensor
from torchdiffeq import odeint

from flow_matching.path.path import ProbPath
from flow_matching.solver.solver import Solver
from flow_matching.utils import gradient, ModelWrapper

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass

@register_conditioning_method('dps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 0.3) # based on step size 0.001 -> if step size is 0.1, scale should be 30

    def grad_and_value(self, x_0_hat, measurement, **kwargs):
        with torch.set_grad_enabled(True):
            x_0_hat_clone = x_0_hat.clone().requires_grad_()
            difference = measurement - self.operator.forward(x_0_hat_clone, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_0_hat_clone)[0]
        return norm_grad

    def conditioning(self, x_t, x_0_hat, measurement, **kwargs):
        norm_grad = self.grad_and_value(x_0_hat=x_0_hat, measurement=measurement)

        x_0_hat_cond = x_0_hat - norm_grad * self.scale
        return x_0_hat_cond


@register_conditioning_method('ddnm')
class DDNM(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        
    def conditioning(self, x_t, x_0_hat, measurement, **kwargs):
        # x̂'_t = x̂_t - A†(A(x̂_t) - y) = (I - A†A)x̂_t + A†y
        y_hat = self.operator.A(x_0_hat)
        residual = y_hat - measurement
        correction = self.operator.A_dagger(residual)
        
        x_0_hat_cond = x_0_hat - correction
            
        return x_0_hat_cond

@register_conditioning_method('dds')
class DDS(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.cg_iter = kwargs.get('cg_iter', 5)
        self.gamma = kwargs.get('gamma', 0.95)

    def CG(self, A, b, x, n_inner=5, eps=1e-5):
        r = b - A(x)
        p = r.clone()
        rsold = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A(p)
            a = rsold / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x = x + a * p
            r = r - a * Ap

            rsnew = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.abs(torch.sqrt(rsnew)) < eps:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def conditioning(self, x_t, x_0_hat, measurement, **kwargs):
        if self.noiser.sigma == 0.0:
            # Acg = A^T * A
            # bcg = A^T * y
            Acg_fn = lambda x: self.operator.AT(self.operator.A(x))
            bcg = self.operator.AT(measurement)
            
            # Find x_0_hat that satisfies data consistency using CG
            x_0_hat_cond = self.CG(Acg_fn, bcg, x_0_hat, n_inner=self.cg_iter)

        else:
            # Acg = I + gamma * A^T * A
            # bcg = x_0_hat + gamma * A^T * y
            Acg_fn = lambda x: x + self.gamma * self.operator.AT(self.operator.A(x))
            bcg = x_0_hat + self.gamma * self.operator.AT(measurement)
            
            # Find x_0_hat that satisfies data consistency using CG
            x_0_hat_cond = self.CG(Acg_fn, bcg, x_0_hat, n_inner=self.cg_iter)
            
        return x_0_hat_cond


@register_conditioning_method('flowdps')
class FlowDPS(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.gd_steps = kwargs.get('gd_steps', 10)
        self.step_size = kwargs.get('step_size', 20.0)

    def conditioning(self, x_t, v, measurement, t, dt, **kwargs):
        with torch.set_grad_enabled(True):
            x_t = x_t.clone().requires_grad_(True)
            v = v.clone().requires_grad_(True)

            x0t = x_t - t * v

            x0t_cond = x0t.clone()
            for _ in range(self.gd_steps):
                x0t_cond.requires_grad_(True)
                
                loss = torch.linalg.norm((measurement - self.operator.A(x0t_cond)))
                grad = torch.autograd.grad(loss, x0t_cond, allow_unused=True)[0]
                x0t_cond = x0t_cond.detach() - self.step_size * grad

            v_cond = (x_t - x0t_cond) / (t if t > 0 else 1e-8)
        return v_cond


@register_conditioning_method('flowchef')
class FlowChef(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.step_size = kwargs.get('step_size', 50.0)

    def conditioning(self, x_t, v, measurement, t, dt, **kwargs):
        with torch.set_grad_enabled(True):
            x_t = x_t.clone().requires_grad_(True)
            v = v.clone().requires_grad_(True)

            x0t = x_t - t * v
            
            x0t_grad = x0t.clone().requires_grad_(True)
            
            loss = torch.linalg.norm((measurement - self.operator.A(x0t_grad)))
            grad = torch.autograd.grad(loss, x0t_grad, allow_unused=True)[0]

            v_cond = v - (self.step_size / dt) * grad
            return v_cond

@register_conditioning_method('psld')
class PSLD(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser, **kwargs)
        self.step_size = kwargs.get('step_size', 50.0)
        self.x0_residue_weight = kwargs.get('x0_residue_weight', 0.1)

    def conditioning(self, x_t, v, measurement, t, dt, **kwargs):
        with torch.set_grad_enabled(True):
            x_t_grad = x_t.clone().requires_grad_(True)
            v = v.clone().requires_grad_(True)
            
            x0t = x_t_grad - t * v
            x_pred = x0t.clone().requires_grad_(True)
            y_pred = self.operator.A(x_pred)
            
            y_residue = torch.linalg.norm((y_pred - measurement))
            
            recon_x = self.operator.A_dagger(measurement)
            x0_residue = torch.linalg.norm((x0t - recon_x))
            
            residue = y_residue + self.x0_residue_weight * x0_residue
            grad = torch.autograd.grad(residue, x_t_grad, allow_unused=True)[0]

            v_cond = v - (self.step_size / dt) * grad
            return v_cond


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

class ODESolverDPS_v(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable], operator: ConditioningMethod, sigma: float = 0.3, path: ProbPath = None, method: str = "dps"):
        super().__init__()
        self.velocity_model = velocity_model
        self.operator = operator
        self.method = method    
        self.noiser = GaussianNoise(sigma=sigma)

        if method != "none":    
            self.posterior_sampling = get_conditioning_method(method, operator, self.noiser)
        self.path = path

    def sample(
        self,
        x_init: Tensor,
        measurement: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            # Predict velocity field
            v = self.velocity_model(x=x, t=t, **model_extras)

            x_0_hat = x + (1 - t) * v if t < 1.0 else x

            # Sampling method (DPS, DDS, DDNM)
            if self.method in ["dps", "dds", "ddnm"]:
                x1_t_hat = self.posterior_sampling.conditioning(
                        x_t=x,
                        measurement=measurement,
                        x_prev=x,
                        x_0_hat=x_0_hat
                    )

                # Update velocity field
                v_cond = (x1_t_hat - x) / (1 - t if t < 1.0 else 1e-8)
                x = x.detach()
            elif self.method in ["flowdps", "flowchef", "psld"]:
                v_cond = self.posterior_sampling.conditioning(
                        x_t=x,
                        v=v,
                        measurement=measurement,
                        t=t,
                        dt=step_size
                    )
                v_cond = v_cond.detach().clone()
            else:
                # No sampling method
                v_cond = v

            # edit velocity field to match measurement
            v_measurement = (measurement - x) / (1 - t if t < 1.0 else 1e-8)
            v_cond = v_cond.clone()
            for channel_idx in range(v_cond.shape[1]):
                if channel_idx not in self.operator.missing_channels:
                    v_cond[:, channel_idx] = v_measurement[:, channel_idx]

            # TODO: 3D alignment

            return v_cond

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]
