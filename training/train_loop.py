import argparse
import gc
import logging
import math
import itertools
from typing import Iterable

from einops import rearrange
import torch
from flow_matching.path import CondOTProbPath
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 100


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    path = CondOTProbPath()

    # Generate all scenarios once (11 scenarios with at least 2 active channels)
    # scenarios_list = [list(item) for item in itertools.product([0, 1], repeat=4)]
    scenarios_list = [[1, 1, 1, 1]]
    scenarios_list = [scenario for scenario in scenarios_list if sum(scenario) > 1]
    
    # Global step counter for proper gradient accumulation across scenarios
    global_step = 0
    
    for data_iter_step, samples in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        samples = rearrange(samples, 'b c h w d -> (b d) c h w')

        samples = samples * 2.0 - 1.0
        noise = torch.randn_like(samples).to(device)
        # noise = torch.randn_like(samples[:,0:1]).to(device)
        # noise = repeat(noise, ' b 1 h w -> b c h w', c = samples.shape[1])
        t = torch.torch.rand(samples.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t_original = path_sample.x_t  # Save original x_t
        u_t_original = path_sample.dx_t  # Save original u_t

        for scenario_idx, scenario_list in enumerate(scenarios_list):
            # Reset gradients at the start of accumulation cycle
            if global_step % accum_iter == 0:
                optimizer.zero_grad()
                batch_loss.reset()
                if data_iter_step > 0 and args.test_run:
                    break

            # Create scenario tensor for this batch
            scenario = torch.tensor(scenario_list).unsqueeze(0).repeat(samples.size(0), 1)
            scenario = scenario.to(device, non_blocking=True)
            conditioning = {"scenario": scenario}
            
            # Create scenario mask [B, C, 1, 1]
            scenario_mask = scenario.view(scenario.size(0), -1, 1, 1)
            
            # Count active channels for proper loss scaling
            num_active_channels = sum(scenario_list)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Apply masking to copies (not original tensors)
                x_t = x_t_original * scenario_mask
                u_t = u_t_original * scenario_mask
                
                v_t = model(x_t, t, extra=conditioning)
                v_t = v_t * scenario_mask
                
                loss = torch.pow(v_t - u_t, 2).sum(dim=[1, 2, 3]).mean()
                loss = loss / (num_active_channels * samples.shape[2] * samples.shape[3])

            loss_value = loss.item()
            batch_loss.update(loss)
            epoch_loss.update(loss)

            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")

            # Scale loss by accumulation steps (accounts for both accum_iter and scenarios)
            loss /= accum_iter

            # Apply gradient update when accumulation cycle is complete
            apply_update = (global_step + 1) % accum_iter == 0
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=apply_update,
            )
            
            # Update EMA weights after optimizer step
            if apply_update and isinstance(model, EMA):
                model.update_ema()
            elif (
                apply_update
                and isinstance(model, DistributedDataParallel)
                and isinstance(model.module, EMA)
            ):
                model.module.update_ema()
            
            global_step += 1

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
