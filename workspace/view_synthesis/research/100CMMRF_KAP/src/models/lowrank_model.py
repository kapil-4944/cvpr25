from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.density_fields import KPlaneDensityField
from models.kplane_field import KPlaneField, FlowField
from ops.activations import init_density_activation, init_flow_activation
from raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from utils.timer import CudaTimer


class LowrankModel(nn.Module):
    def __init__(self,
                 grid_config: Union[str, Dict[str, Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 flow_activation: Optional[str] = 'tanh',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 canonical_time: int = 0,
                 time_dependent_color: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: Dict[str, Dict] = eval(grid_config)
        else:
            self.config: Dict[str, Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.flow_act = init_flow_activation(flow_activation)
        self.canonical_time = canonical_time
        self.time_dependent_color = time_dependent_color
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.field_3d = KPlaneField(
            aabb,
            grid_config=self.config['model_3d'],
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
            time_dependent_color=time_dependent_color,
        )

        self.field_bf = FlowField(
            aabb,
            grid_config=self.config['model_bf'],
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            flow_activation=self.flow_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
            canonical_time=canonical_time
        )

        self.field_ff = FlowField(
            aabb,
            grid_config=self.config['model_ff'],
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            flow_activation=self.flow_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
            canonical_time=canonical_time
        )

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps, sf_mask=None, timestamps2=None, df_mask: Optional[torch.Tensor] = None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions are 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)

        pi = ray_samples.get_positions()  # (nr, ns, 3)

        # Flow is not a function of directions
        bwd_flow_dict = self.field_bf(pi, timestamps=timestamps)
        pi_prime = bwd_flow_dict['displaced_points']  # (nr, ns, 3)

        # Provide timestamps to the 3d model if color has to be predicted as a function of time
        field_3d_dict = self.field_3d(pi_prime, ray_bundle.directions, timestamps)
        rgb, density = field_3d_dict["rgb"], field_3d_dict["density"]

        if self.training:
            fwd_flow_dict2 = None
            if sf_mask is not None and df_mask is not None:
                sf_neighbour_point_mask = torch.logical_not(torch.logical_or(sf_mask, df_mask))
                if sf_neighbour_point_mask.sum() == 0:
                    timestamps2 = None
                else:
                    timestamps2 = timestamps2[sf_neighbour_point_mask]
            else:
                sf_neighbour_point_mask = None
                timestamps2 = None
            if self.config['flow_consistency_stop_gradient_forward_flow']:
                fwd_flow_dict1 = self.field_ff(pi_prime.detach(), timestamps=timestamps)
                if timestamps2 is not None:
                    fwd_flow_dict2 = self.field_ff(pi_prime.detach()[sf_neighbour_point_mask], timestamps=timestamps2)
            else:
                fwd_flow_dict1 = self.field_ff(pi_prime, timestamps=timestamps)
                if timestamps2 is not None:
                    fwd_flow_dict2 = self.field_ff(pi_prime[sf_neighbour_point_mask], timestamps=timestamps2)
            pi_dprime = fwd_flow_dict1['displaced_points']  # (nr, ns, 3)
            if fwd_flow_dict2 is not None :
                pi_tprime = fwd_flow_dict2['displaced_points']
            else:
                pi_tprime = None

        else:                                                                       # TODO Kapil: line 242-244 added for grid visulization
            flow_after_10_frames = False
            if flow_after_10_frames:
                a = (10 / (300-1)) * 2 - 1
            else:
                a = 0
            fwd_flow_dict1 = self.field_ff(pi_prime, timestamps=timestamps+a)
            pi_dprime = fwd_flow_dict1['displaced_points']  # (nr, ns, 3)
            pi_tprime = None

        weights = ray_samples.get_weights(density)  # (nr, ns, 1)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

            # Include all the points to compute sparse flow loss.
            outputs['weights'] = weights
            outputs['pi'] = pi
            outputs['pi_prime'] = pi_prime
            outputs['pi_dprime'] = pi_dprime
            outputs['pi_tprime'] = pi_tprime
        else:                                                           # TODO Kapil: line 269+6-273+6 added for grid visulization
            outputs["weights"] = weights
            outputs["pi"] = pi
            outputs["pi_prime"] = pi_prime
            outputs["pi_dprime"] = pi_dprime
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_3d_params = self.field_3d.get_params()
        model_bf_params = self.field_bf.get_params()

        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_3d_params["field"] + model_bf_params["field"] + [p for pnp in pn_params for p in pnp["field"]] # + model_ff_params["field"]
        nn_params = model_3d_params["nn"] + model_bf_params["nn"]+ [p for pnp in pn_params for p in pnp["nn"]] # + model_ff_params["nn"]
        other_params = model_3d_params["other"] + model_bf_params["other"] + [p for pnp in pn_params for p in pnp["other"]] # + model_ff_params["other"]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]

    def get_ff_params(self, lr: float):
        model_ff_params = self.field_ff.get_params()
        field_params = model_ff_params["field"]
        nn_params = model_ff_params["nn"]
        other_params = model_ff_params["other"]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]

    def get_params_with_ff(self, lr: float):
        model_3d_params = self.field_3d.get_params()
        model_bf_params = self.field_bf.get_params()
        model_ff_params = self.field_ff.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_3d_params["field"] + model_bf_params["field"] + model_ff_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_3d_params["nn"] + model_bf_params["nn"] + model_ff_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_3d_params["other"] + model_bf_params["other"] + model_ff_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]