"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from torchvision.transforms import RandomCrop
from . import dist_util
from .flows import OnlineSlimFlow
from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator
from torchvision.utils import save_image, make_grid
import os
from mpi4py import MPI

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class RectifiedDenoiser:
    def __init__(
        self,
        device,
        num_steps=1000,
        TN=16,
        adapt_cu="origin",
        predstep=1):
        self.sam = False
        self.N = num_steps
        self.device = device
        assert adapt_cu in ["origin","rule","uniform"],"adapt_cu must be one of 'origin','rule','uniform'."
        self.adapt_cu = adapt_cu
        self.discrete = False
        self.TN = TN
        self.device = device
        self.predstep = predstep
        self.eps = 1e-5
        self.num_timesteps = TN
        self.iteration = 74000
        self.test_dir = "./test"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.cudflow = lambda model,ema_model : OnlineSlimFlow(
            device=device,
            model = model,
            ema_model=ema_model,
            predstep=predstep,
            num_steps=num_steps,
            TN = self.TN,
            adapt_cu=adapt_cu,
        )
        self.criticion_1 = self.criticion_2 = nn.MSELoss()

    
    def catchupdist_loss(
        self,
        flow_model,
        forward_model,
        ema_model,
        x_start,
        independent=False,
        noise=None,    
        model_kwargs=None,    
    ):
        flow_model_fn = lambda x,t,return_features=False:flow_model(x,t,return_features=return_features,**model_kwargs)
        cudflow = self.cudflow(flow_model_fn, ema_model)
        if model_kwargs is None:
            model_kwargs = {}

        def get_kl(mu, logvar):
            # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
            kl = 0.5 * th.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1,2,3])
            _loss_prior = th.mean(kl) / (mu.shape[1]*mu.shape[2]*mu.shape[3])
            return _loss_prior
        
        if noise is None:
            if independent:
                z = th.randn_like(x_start)
                loss_prior = 0
            else:
                assert forward_model is not None, "forward_model must be provided when independint is False."
                z, mu, logvar = forward_model(x_start, th.ones((x_start.shape[0]), device=self.device))
                loss_prior = get_kl(mu, logvar)
        else:
            z = noise
        z = z.type_as(x_start)
        predstep_loss_list = []
        if self.predstep==1:
            pred_z_t,ema_z_t,gt_z_t = cudflow.get_train_tuple(z0=x_start, z1=z,pred_step=self.predstep)
            # Learn reverse model
            loss_fm =self.criticion_1(pred_z_t , ema_z_t) +  self.criticion_2(pred_z_t , gt_z_t)
        elif self.predstep==2 or self.predstep==3:
            loss_fm = th.Tensor([0.]).to(self.device)
            pred_z_t_list,ema_z_t_list,gt_z_t = cudflow.get_train_tuple(z0=x_start, z1=z,pred_step=self.predstep)
            for pred_z_t,ema_z_t in zip(pred_z_t_list,ema_z_t_list):
                if self.predstep == 2:
                    predstep_loss_list.append(self.criticion_2(pred_z_t,ema_z_t))
                if self.predstep == 3:
                    predstep_loss_list.append(self.criticion_2(pred_z_t,ema_z_t))
            for pred_z_t in (pred_z_t_list):
                predstep_loss_list.append(self.criticion_2(pred_z_t,gt_z_t))
            for _loss in predstep_loss_list:
                loss_fm+=_loss
                _loss = round(_loss.detach().clone().item(),2)
        else:
            raise NotImplementedError
        loss_fm = loss_fm.mean()
        loss = loss_fm + 5 * loss_prior
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0 and self.iteration%2000==0:
            save_image(x_start[:16]*0.5+0.5,os.path.join(self.test_dir,f"x_start_{self.iteration}.png"), nrow=4)
            save_image((z[:16] - pred_z_t[:16])*0.5+0.5,os.path.join(self.test_dir,f"one_step_{self.iteration}.png"), nrow=4)
        self.iteration +=1
        terms = {}
        terms["loss"] = loss
        return terms

    def denoise(self,model, x_t, sigma, **model_kwargs):
        z_x = model(x_t,sigma,**model_kwargs)
        return z_x

class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        predstep = 1,
        TN = 16,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.predstep = predstep
        self.TN = TN
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
    
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        if target_model:

            @th.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, **model_kwargs)[1]

        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:

            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t)

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss
        return terms

    def progdist_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        @th.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(
                next_t - t, dims
            )
            return denoiser

        indices = th.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t3 = t3**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        denoised_x = denoise_fn(x_t, t)

        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()

        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(denoised_x - target_x)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (denoised_x - target_x) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                denoised_x = F.interpolate(denoised_x, size=224, mode="bilinear")
                target_x = F.interpolate(target_x, size=224, mode="bilinear")
            loss = (
                self.lpips_loss(
                    (denoised_x + 1) / 2.0,
                    (target_x + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    def catchupdist_loss(self, model, forward_model, x_start, num_scales, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            def get_kl(mu, logvar):
                # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
                kl = 0.5 * th.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1,2,3])
                _loss_prior = th.mean(kl) / (mu.shape[1]*mu.shape[2]*mu.shape[3])
                return _loss_prior
            assert forward_model is not None, "forward_model must be provided when independint is False."
            noise, mu, logvar = forward_model(x_start, th.ones((x_start.shape[0]), device=self.device))
            loss_prior = get_kl(mu, logvar)

        terms = {}
        dims = x_start.ndim
        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho


        x_t = x_start + noise * append_dims(t, dims)
        model_output, denoised = self.denoise(model, x_t, t, predstep=self.predstep, **model_kwargs)
        pred_v = denoised
        gt_v = x_start
        snrs = self.get_snr(t)
        catchingup_size = int(num_scales/self.TN)
        catchingup_size = th.randint(
            1, catchingup_size+1, (x_start.shape[0],), device=x_start.device
        )
        catchingup_size = th.where(indices+catchingup_size>num_scales-1, num_scales-1-indices, catchingup_size)
        t2 = self.sigma_max ** (1 / self.rho) + (indices + catchingup_size) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho


        @th.no_grad()
        def rk34_solver(samples, t, next_t, x0):
            x = samples
            h = t - next_t
            denoiser_1 = self.denoise(x, t,**model_kwargs)
            d = (x - denoiser_1) / append_dims(t, dims)
            samples = x - d * append_dims(h, dims)
            denoiser_2 = self.denoise(samples, next_t,**model_kwargs)
            next_d = (samples - denoiser_2) / append_dims(next_t, dims)
            next_next_t = t-2*h
            next_next_t = t-3*h
            thresbold = self.sigma_min

            next_next_t = th.where(next_next_t<thresbold, thresbold, next_next_t)
            next_next_next_t = th.where(next_next_next_t<thresbold, thresbold, next_next_t)
            samples = x-(7*h*d/4+1*h*next_d/4)
            denoise_3 = self.denoise(samples,next_next_t,**model_kwargs)
            next_next_d = (samples - denoise_3) / append_dims(next_next_t, dims)
            samples1 = x - h*((5/12)*d+(2/3)*next_d-(1/12)*next_next_d)
            samples2 = x - 2*h*((5/12)*d+(2/3)*next_d-(1/12)*next_next_d)
            samples3 = x - 3*h*((5/12)*d+(2/3)*next_d-(1/12)*next_next_d)
            return self.denoise(samples1, next_t,**model_kwargs),self.denoise(samples2, next_next_t,**model_kwargs),self.denoise(samples3, next_next_next_t,**model_kwargs)
        

        @th.no_grad()
        def rk23_solver(samples, t, next_t, x0):
            x = samples
            h = t - next_t
            next_next_t = t-2*h
            thresbold = self.sigma_min
            
            next_next_t = th.where(next_next_t<thresbold, thresbold, next_next_t)
            denoiser = self.denoise(x, t,**model_kwargs)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x - d * append_dims(t-next_t, dims)
            denoiser = self.denoise(samples, next_t,**model_kwargs)
            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples1 = x + ((d + next_d) / 2)* append_dims((next_t - t), dims)
            samples2 = x + (d + next_d)*append_dims((next_t - t), dims)
            return self.denoise(samples1,next_t,**model_kwargs),self.denoise(samples2,next_next_t,**model_kwargs)

        @th.no_grad()
        def rk12_solver(samples, t, next_t, x0):
            x = samples
            denoiser = self.denoise(x, t,**model_kwargs)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x - d * append_dims(t-next_t, dims)
            return self.denoise(samples, next_t,**model_kwargs)
        
        if self.predstep == 1:
            ema_v1 = rk12_solver(pred_v[0] if isinstance(pred_v,list) else pred_v, t, t2, x_start)
        elif self.predstep == 2:
            ema_v1,ema_v2 = rk23_solver(pred_v[0] if isinstance(pred_v,list) else pred_v, t, t2, x_start)
        elif self.predstep == 3:
            ema_v1,ema_v2,ema_v3 = rk34_solver(pred_v[0] if isinstance(pred_v,list) else pred_v, t, t2, x_start)
        else:
            raise ValueError(f"Unknown predstep {self.predstep}")
        
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        if self.predstep==1:
            terms["xs_ori"] = mean_flat((pred_v - gt_v) ** 2)
            terms["ori"] = mean_flat(weights * (pred_v - gt_v) ** 2)
            terms["prior"] = loss_prior
            terms["kd"] = mean_flat((pred_v - ema_v1) ** 2)
        elif self.predstep==2:
            terms["xs_ori"] = mean_flat((pred_v[0] - gt_v) ** 2)+ mean_flat((pred_v[1] - gt_v) ** 2)
            terms["ori"] =mean_flat(weights*(pred_v[0] - gt_v) ** 2)+ mean_flat(weights*(pred_v[1] - gt_v) ** 2)
            terms["prior"] = loss_prior
            terms["kd"] = mean_flat((pred_v[0] - ema_v1) ** 2) + mean_flat((pred_v[1] - ema_v2) ** 2)
        elif self.predstep==2:
            terms["xs_ori"] = mean_flat((pred_v[0] - gt_v) ** 2)+ mean_flat((pred_v[1] - gt_v) ** 2) + mean_flat((pred_v[2] - gt_v) ** 2)
            terms["ori"] = mean_flat(weights*(pred_v[0] - gt_v) ** 2)+ mean_flat(weights*(pred_v[1] - gt_v) ** 2) +  mean_flat(weights*(pred_v[2] - gt_v) ** 2)
            terms["prior"] = loss_prior
            terms["kd"] = mean_flat((pred_v[0] - ema_v1) ** 2) + mean_flat((pred_v[1] - ema_v2) ** 2) + mean_flat((pred_v[2] - ema_v3) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["ori"] + terms["vb"] + terms["prior"] + terms["kd"]
        else:
            terms["loss"] = terms["ori"] + terms["prior"] + terms["kd"]

        return terms
            

    def denoise(self, model, x_t, sigmas, predstep=1, **model_kwargs):
        import torch.distributed as dist

        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        if predstep>=2:
            denoised = []
            model_output_list = model(c_in * x_t, rescaled_t, return_features=True, **model_kwargs)
            for i in range(len(model_output_list)):
                denoised.append(c_out * model_output_list[i] + c_skip * x_t)
            return model_output_list[0], denoised
        else:
            model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
            denoised = c_out * model_output + c_skip * x_t
            return model_output, denoised


def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images

@th.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -th.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images



@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images

def rectified_sample(
    model,
    shape,
    steps,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    sampler="heun",
    generator_id=1,
    generator=None,
):
    """
    (self, z1=None, N=None, use_tqdm=True, solver = 'euler',momentum=0.0,generator_id=1):
    """
    if generator is None:
        generator = get_generator("dummy")
    x_T = generator.randn(*shape, device=device)
    noise = x_T.detach().clone()
    sample_fn = {
        "heun": sample_heun_rect,
        "euler": sample_euler_rect,
        "onestep":sample_onestep_rect,
        "rk45":sample_rk45_rect
    }[sampler]

    sampler_args = dict(N=steps,clip_denoised=clip_denoised)
    print(sampler_args)
    def denoiser(x_t, t):
        denoised = model(x_t, t, return_features=True, **model_kwargs)
        if isinstance(denoised,tuple) or isinstance(denoised,list):
            denoised = denoised[generator_id-1]
        return denoised

    x_0,x0hat_list,_ = sample_fn(
        denoiser,
        x_T,
        steps,
        generator,
        **sampler_args,
    )
    return x_0,noise,x0hat_list


@th.no_grad()
def sample_euler_rect(
    denoiser,
    x,
    ts,
    generator,
    N,
    clip_denoised=True,
):
    indices = reversed(range(1,ts+1))
    from tqdm.auto import tqdm
    indices = tqdm(indices)
    dt = -1./N
    traj = [] # to store the trajectory
    x0hat_list = []
    x = x.detach().clone()
    z = x.detach().clone()
    batchsize = x.shape[0]
    traj.append(x.detach().clone())

    for i in indices:
        t = th.ones((batchsize,1), device=x.device) * i / N
        d = denoiser(x,t.squeeze())
        x0hat = x - d * t.view(-1,1,1,1)
        zhat = (d+x0hat)
        if clip_denoised:
            x0hat = x0hat.clamp(-1, 1)
        d = zhat - x0hat
        x = x + d * dt
        x0hat_list.append(x0hat)
        traj.append(x.detach().clone())
    return x.clamp(-1,1),x0hat_list,traj

@th.no_grad()
def sample_rk45_rect(
    denoiser,
    x,
    ts,
    generator,
    N,
    rtol=1e-5, 
    atol=1e-5,
    clip_denoised=True,
):
    from scipy import integrate
    dshape = x.shape
    device = x.device
    eps = 1e-5
    def ode_func(t, x):
      x = th.from_numpy(x.reshape(dshape)).to(device).type(th.float32)
      vec_t = th.ones(dshape[0], device=x.device) * t
      vt = denoiser(x, vec_t.squeeze())
      vt = vt.detach().cpu().numpy().reshape(-1)
      return vt
    solution = integrate.solve_ivp(ode_func, (ts/N, eps), x.detach().cpu().numpy().reshape(-1), method="RK45", rtol = rtol, atol = atol)
    nfe = solution.nfev
    print("NFE:",nfe)
    x = th.from_numpy(solution.y[:,-1].reshape(dshape))
    return x.clamp(-1,1),None,None

@th.no_grad()
def sample_heun_rect(
    denoiser,
    x,
    ts,
    generator,
    N,
    clip_denoised=True
):
    if N % 2 == 0:
      raise ValueError("N must be odd when using Heun's method.")
    N = (N + 1) // 2
    ts = (ts+1) // 2
    indices = reversed(range(1,ts+1))
    from tqdm.auto import tqdm
    indices = tqdm(indices)
    dt = -1./N
    traj = [] # to store the trajectory
    x0hat_list = []
    x = x.detach().clone()
    z = x.detach().clone()
    batchsize = x.shape[0]
    traj.append(x.detach().clone())

    for i in indices:
        t = th.ones((batchsize,1), device=x.device) * i / N
        d = denoiser(x,t.squeeze())
        if i!=1:
            x_next = x.detach().clone() + d * dt
            d_next = denoiser(x_next,(t+dt).squeeze())
            d = (d+d_next)/2
        x0hat = x - d * t.view(-1,1,1,1)
        # zhat = (d+x0hat)
        # if clip_denoised:
        #     x0hat = x0hat.clamp(-1, 1)
        # d = zhat - x0hat
        x = x + d * dt
        x0hat_list.append(x0hat)
        traj.append(x.detach().clone())
    return x.clamp(-1,1), x0hat_list, traj

@th.no_grad()
def sample_onestep_rect(
    denoiser,
    x,
    generator,
    N,
):

    x = x.detach().clone()
    batchsize = x.shape[0]
    t = th.ones((batchsize,1), device=x.device)
    d = denoiser(x,t.squeeze())
    x = x - d
    return x.clamp(1,1),None,None