import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from einops import rearrange
from einops import repeat

import torch
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 100


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float = 0.0, **kwargs
    ):
        
        assert (
            cfg_scale == 0.0
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t
        
        # Extract extra arguments for the model
        extra = kwargs.get('extra', {})

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra=extra)
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra=extra)

        self.nfe_counter += 1
        
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    solver = ODESolver(velocity_model=cfg_scaled_model)
    ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    snapshots_saved = False
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, samples in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        samples = rearrange(samples, 'b c h w d -> (b d) c h w')
        samples_fid = repeat(samples[:,:1], 'b 1 h w -> b c h w', c=3)
        fid_metric.update(samples_fid, real=True)

        scenarios = [[1, 1, 1, 1]]
        scenarios = [torch.tensor(scenario).unsqueeze(0).repeat(samples.size(0), 1) for scenario in scenarios if sum(scenario) > 1]

        for scenario in scenarios:
            scen = scenario[0].numpy().tolist()
            scenario = scenario.to(device, non_blocking=True)

            if num_synthetic < fid_samples:
                cfg_scaled_model.reset_nfe_counter()
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)
                
                time_grid = torch.tensor([0.0, 1.0], device=device)

                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=ode_opts["step_size"]
                    if "step_size" in ode_opts
                    else None,
                    cfg_scale=args.cfg_scale,
                    extra={"scenario": scenario},
                )

                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
                synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
                logger.info(
                    f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
                )
                if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                    synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
                synthetic_samples_fid = repeat(synthetic_samples[:,:1], 'b 1 h w -> b c h w', c=3)
                fid_metric.update(synthetic_samples_fid, real=False)
                num_synthetic += synthetic_samples.shape[0]
                if not snapshots_saved and args.output_dir:
                    samples = (synthetic_samples + 1) / 2
                    samples = samples.permute(0, 3, 2, 1)
                    samples = samples.contiguous()
                    samples = samples.cpu().numpy()

                    image_dir = Path(args.output_dir) / "snapshots" 
                    os.makedirs(image_dir, exist_ok=True)

                    ch = 2
                    if args.dataset == "brats_ch2" or args.dataset == "ixi_ch2": ch = 2
                    elif args.dataset == "brats_ch3" or args.dataset == "ixi_ch3": ch = 3
                    elif args.dataset == "brats": ch = 4

                    plt.figure(figsize=(12, 12))
                    for i in range(ch):
                        for j in range(ch):
                            plt.subplot(ch, ch, i * ch + j + 1)
                            plt.imshow(samples[i, :, :, j], cmap="gray")
                            plt.axis("off")
                            plt.tight_layout()
                    plt.savefig(image_dir / f"{epoch}_{scen}_{data_iter_step}.png", dpi=300, bbox_inches="tight")
                    plt.close()
                    snapshots_saved = True

                if args.save_fid_samples and args.output_dir:
                    samples = (synthetic_samples + 1) / 2
                    samples = samples.permute(0, 3, 2, 1)
                    samples = samples.contiguous()
                    samples = samples.cpu().numpy()

                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)

                    ch = 2
                    if args.dataset == "brats_ch2" or args.dataset == "ixi_ch2": ch = 2
                    elif args.dataset == "brats_ch3" or args.dataset == "ixi_ch3": ch = 3
                    elif args.dataset == "brats": ch = 4

                    plt.figure(figsize=(12, 12))
                    for i in range(ch):
                        for j in range(ch):
                            plt.subplot(ch, ch, i * ch + j + 1)
                            plt.imshow(samples[i, :, :, j], cmap="gray")
                            plt.axis("off")
                            plt.tight_layout()
                    plt.savefig(image_dir / f"{epoch}_{scen}_{data_iter_step}.png", dpi=300, bbox_inches="tight")
                    plt.close()

            if not args.compute_fid:
                return {}

            if data_iter_step % PRINT_FREQUENCY == 0:
                # Sync fid metric to ensure that the processes dont deviate much.
                gc.collect()
                running_fid = fid_metric.compute()
                logger.info(
                    f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
                )

            if args.test_run:
                break

    return {"fid": float(fid_metric.compute().detach().cpu())}
