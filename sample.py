from abc import ABC, abstractmethod
import os
import yaml
import nibabel as nib
import numpy as np
import torch
from einops import rearrange, repeat
from pathlib import Path

from models.model_configs import instantiate_model
from training.eval_loop import CFGScaledModel
from training.load_and_save import load_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from flow_matching.solver.ode_solver_ps import ODESolverPS
from flow_matching.path import CondOTProbPath
from test_arg_parser import get_args_parser
from mri_dataset import load_data


def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


class MissingModalitiesOperator(NonLinearOperator):
    def __init__(self, device, missing_channels=None):
        self.device = device
        self.missing_channels = missing_channels if missing_channels is not None else []
         
    def forward(self, data, **kwargs):
        if not self.missing_channels:
            return data
        
        result = data.clone()
        for channel_idx in self.missing_channels:
            if channel_idx < data.shape[1]: 
                result[:, channel_idx, :, :] = -1
        
        return result

class MissingModalities():
    def __init__(self, channels, device, missing_channels=None, source_channels=None):
        self.device = device
        self.missing_channels = missing_channels if missing_channels is not None else []
        self.source_channels = source_channels if source_channels is not None else []
        
        mask = torch.zeros(1, channels, 1, 1, device=self.device)
        for channel_idx in self.source_channels:
            mask[:, channel_idx, :, :] = 1
        self.mask = mask

    def forward(self, data, **kwargs):
        if not self.missing_channels:
            return data

        return data * self.mask

    def A(self, x):
        return x * self.mask

    def AT(self, y):
        return y * self.mask

    def A_dagger(self, y):
        return y * self.mask


def save_nifti(data, path):
    data = data.detach().cpu().numpy()
    data = data.transpose(1, 2, 0)
    data = data.astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.dataset in ["brats", "brats_ch2", "brats_ch3"]:
        dataset = load_data(
            data_dir=args.data_path,
            phase="test",
            dataset=args.dataset,
        )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    device = torch.device(args.device)
    model = instantiate_model(
        architechture=args.dataset,
        use_ema=args.use_ema,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=args.optimizer_betas
    )

    if args.decay_lr:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )
    loss_scaler = NativeScaler()
    load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    model = model.to(device)
    model.eval()

    path = CondOTProbPath()

    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    if args.dataset == "brats_ch2": ch = 2
    elif args.dataset == "brats_ch3": ch = 3
    elif args.dataset == "brats": ch = 4

    # Scenario: 1 for active channels (source + missing), 0 for inactive channels
    scenario = torch.zeros([1, ch], dtype=torch.int32, device=device)
    for channel_idx in args.source_channels:
        scenario[0, channel_idx] = 1
    for channel_idx in args.missing_channels:
        scenario[0, channel_idx] = 1

    operator = MissingModalities(channels=ch, device=device, missing_channels=args.missing_channels, source_channels=args.source_channels)
    solver = ODESolverPS(velocity_model=cfg_scaled_model, operator=operator, sigma=args.sigma, path=path, method=args.method)
    ode_opts = args.ode_options

    output_dir = Path(args.sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_size = args.step_size

    for i, (samples, subject) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        synthetic_samples = torch.zeros(samples.shape, dtype=torch.float32, device=device)

        samples = rearrange(samples, 'b c h w d -> (b d) c h w')
        samples = samples * 2.0 - 1.0

        samples_masked = torch.zeros_like(samples)
        for channel_idx in args.source_channels:
            samples_masked[:, channel_idx, :, :] = samples[:, channel_idx, :, :]
        for channel_idx in args.missing_channels:
            samples_masked[:, channel_idx, :, :] = samples[:, channel_idx, :, :]

        samples = samples_masked

        measurement = operator.forward(samples)
        subject = subject[0]

        print(f"processing {subject} ...")

        cfg_scaled_model.reset_nfe_counter()

        time_grid = torch.tensor([0.0, 1.0], device=device)

        gt_dir = output_dir / "gt" / subject 
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        save_nifti(samples[:,0], gt_dir / f"t1.nii.gz")
        save_nifti(samples[:,1], gt_dir / f"t2.nii.gz")
        save_nifti(samples[:,2], gt_dir / f"t1ce.nii.gz")
        save_nifti(samples[:,3], gt_dir / f"flair.nii.gz")

        pred_dir = output_dir / f"{args.method}_[STEP_SIZE={step_size}]" / subject / f"{args.source_channels}_to_{args.missing_channels}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        nex_list = [1, 2, 4, 6, 8, 10] #[1, 2, 4, 6, 8, 10, 12]
        recon_list = []
        for i in range(nex_list[-1]):
            noise = torch.randn((400, *samples.shape[1:]), dtype=torch.float32, device=device)
            noise = noise.sum(dim=0, keepdim=False) / 20 # n noise is 400
            noise = repeat(noise, ' c h w -> d c h w', d=samples.shape[0])
            observed_mean = measurement.clone()
            observed_mean = observed_mean.sum(dim=1, keepdim=True) / len(args.source_channels)
            observed_mean = repeat(observed_mean, 'd 1 h w -> d c h w', c=samples.shape[1])
            noise = noise + 0.025 * (observed_mean - noise)
            with torch.no_grad():
                x_0 = noise.clone()
                for ch in args.source_channels:
                    x_0[:, ch, :, :] = measurement[:, ch, :, :] * 0.5 + noise[:, ch, :, :] * 0.5
                x_0 = x_0 * scenario.view(scenario.size(0), -1, 1, 1)
                synthetic_samples = solver.sample(
                    time_grid=time_grid,
                    x_init=x_0,
                    measurement=measurement.clone(),
                    method=args.ode_method,
                    return_intermediates=False,
                    atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                    rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                    step_size=step_size,
                    cfg_scale=args.cfg_scale,
                    extra={"scenario": scenario},
                )
                recon_list.append(synthetic_samples)

            save_nifti(synthetic_samples[:,0], pred_dir / f"t1_{i}.nii.gz")
            save_nifti(synthetic_samples[:,1], pred_dir / f"t2_{i}.nii.gz")
            save_nifti(synthetic_samples[:,2], pred_dir / f"t1ce_{i}.nii.gz")
            save_nifti(synthetic_samples[:,3], pred_dir / f"flair_{i}.nii.gz")

        for nex in nex_list:
            recon_nex = sum(recon_list[:nex]) / nex
            save_nifti(recon_nex[:,0], pred_dir / f"t1_EX{nex}.nii.gz")
            save_nifti(recon_nex[:,1], pred_dir / f"t2_EX{nex}.nii.gz")
            save_nifti(recon_nex[:,2], pred_dir / f"t1ce_EX{nex}.nii.gz")
            save_nifti(recon_nex[:,3], pred_dir / f"flair_EX{nex}.nii.gz")

        # break

if __name__ == '__main__':
    main()