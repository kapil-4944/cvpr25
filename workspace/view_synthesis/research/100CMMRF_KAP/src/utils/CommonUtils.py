# Shree KRISHNAya Namaha
# Common utility functions
# Author: Nagabhushan S N, Kapil Choudhary
# Last Modified: 16/11/2023

import time
import datetime
import traceback
import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas
from dataclasses import is_dataclass
import torch

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

from datasets.intrinsics import Intrinsics

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def inverse_ndc(pts_ndc: torch.Tensor, intrinsics: Intrinsics, near, fp16: bool):
    if is_dataclass(intrinsics):
        w2ndc2 = torch.tensor([
            [intrinsics.focal_x / intrinsics.center_x, 0, 0, 0],
            [0, intrinsics.focal_y / intrinsics.center_y, 0, 0],
            [0, 0, -1, -2*near],
            [0, 0, -1, 0],
        ]).type(torch.float32).to(pts_ndc.device)[None, None]  # (1, 1, 4, 4)
    else:
        w2ndc2 = torch.tensor([
            [intrinsics[0, 0, 0] / intrinsics[0, 0, 2], 0, 0, 0],
            [0, intrinsics[0, 1, 1] / intrinsics[0, 1, 2], 0, 0],
            [0, 0, -1, -2*near],
            [0, 0, -1, 0],
        ]).type(torch.float32).to(pts_ndc.device)[None, None]
    norm_const = 5e-5 if fp16 else 1e-6

    pts_ndc_homo = torch.cat([pts_ndc, torch.ones_like(pts_ndc[:, :, :1])], dim=2)[:, :, :, None]  # (nr, ns, 4, 1)
    pts_world_homo = torch.matmul(torch.linalg.inv(w2ndc2), pts_ndc_homo)
    norm_const1 = norm_const * sign_no_zeros(pts_world_homo[:, :, 3:, 0])
    pts_world = pts_world_homo[:, :, :3, 0] / (pts_world_homo[:, :, 3:, 0] + norm_const1)  # (nr, ns, 3)
    # elif len(pts_ndc.shape)==2:
    #     w2ndc2 = torch.tensor([
    #         [intrinsics.focal_x / intrinsics.center_x, 0, 0, 0],
    #         [0, intrinsics.focal_y / intrinsics.center_y, 0, 0],
    #         [0, 0, -1, -2*near],
    #         [0, 0, -1, 0],
    #     ]).type(torch.float32).to(pts_ndc.device)[ None]  # (1, 4, 4)
    #     norm_const = 5e-5 if fp16 else 1e-6

    if not torch.isfinite(pts_world).all():
            print('gotcha')
    return pts_world


def perspective_projection(pts: torch.Tensor, extrinsics_w2c: torch.Tensor, intrinsics: Intrinsics, fp16: bool):
    '''

    :param pts:
    :param extrinsics_w2c: shape should be (n,4,4)
    :param intrinsics: (n,3,3)
    :param fp16:
    :return: projected points (n,1,2)
    '''
    p2 = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]).type(torch.float32).to(pts.device)[None, None]
    if is_dataclass(intrinsics):
        if fp16:
            intrinsics_mat = intrinsics.as_matrix_resolution_normalized()
        else:
                intrinsics_mat = intrinsics.to_matrix()
        intrinsics_mat = torch.from_numpy(intrinsics_mat).type(torch.float32).to(pts.device)[None, None]  # (1, 1, 3, 3)
    else:
        intrinsics_mat = intrinsics[0]
    intrinsics_mat = intrinsics_mat.to(pts.device)[None, None]  # (1, 1, 3, 3)
    norm_const = 5e-5 if fp16 else 1e-6

    pts_world_homo = torch.cat([pts, torch.ones_like(pts[:, :, :1])], dim=2)[:, :, :, None]  # (nr, ns, 4, 1)
    # if extrinsics_w2c[1].sum() == extrinsics_w2c[100].sum():  # TODO Kapil : remove this hardcoded part
    #     extrinsics_w2c = extrinsics_w2c[0]
    #     extrinsics_w2c=extrinsics_w2c[None,None,:,:]
    pts_cam = torch.matmul(extrinsics_w2c[:,None,:,:], pts_world_homo)
    pts_cam_cv = torch.matmul(p2, pts_cam)
    pixels_homo = torch.matmul(intrinsics_mat, pts_cam_cv[:, :, :3, :])
    norm_const1 = norm_const * sign_no_zeros(pixels_homo[:, :, 2:, 0])
    pixel_locs = pixels_homo[:, :, :2, 0] / (pixels_homo[:, :, 2:, 0] + norm_const1)
    if not torch.isfinite(pixel_locs).all():
        print('gotcha')
    return pixel_locs


def inverse_ndcflow_pred(pts_ndc: torch.Tensor, intrinsics: Intrinsics, near, fp16: bool):
    if is_dataclass(intrinsics):
        w2ndc2 = torch.tensor([
            [intrinsics.focal_x / intrinsics.center_x, 0, 0, 0],
            [0, intrinsics.focal_y / intrinsics.center_y, 0, 0],
            [0, 0, -1, -2*near],
            [0, 0, -1, 0],
        ]).type(torch.float32).to(pts_ndc.device)[None, None]  # (1, 1, 4, 4)
    else:
        w2ndc2 = torch.tensor([
            [intrinsics[0, 0, 0] / intrinsics[0, 0, 2], 0, 0, 0],
            [0, intrinsics[0, 1, 1] / intrinsics[0, 1, 2], 0, 0],
            [0, 0, -1, -2*near],
            [0, 0, -1, 0],
        ]).type(torch.float32).to(pts_ndc.device)[None, None]
    norm_const = 5e-5 if fp16 else 1e-6

    pts_ndc_homo = torch.cat([pts_ndc, torch.ones_like(pts_ndc[:, :, :1])], dim=2)[:, :, :, None]  # (nr, ns, 4, 1)
    pts_world_homo = torch.matmul(torch.linalg.inv(w2ndc2), pts_ndc_homo)
    norm_const1 = norm_const * sign_no_zeros(pts_world_homo[:, :, 3:, 0])
    pts_world = pts_world_homo[:, :, :3, 0] / (pts_world_homo[:, :, 3:, 0] + norm_const1)  # (nr, ns, 3)
    # elif len(pts_ndc.shape)==2:
    #     w2ndc2 = torch.tensor([
    #         [intrinsics.focal_x / intrinsics.center_x, 0, 0, 0],
    #         [0, intrinsics.focal_y / intrinsics.center_y, 0, 0],
    #         [0, 0, -1, -2*near],
    #         [0, 0, -1, 0],
    #     ]).type(torch.float32).to(pts_ndc.device)[ None]  # (1, 4, 4)
    #     norm_const = 5e-5 if fp16 else 1e-6

    if not torch.isfinite(pts_world).all():
            print('gotcha')
    return pts_world


def perspective_projection_flow_pred(pts: torch.Tensor, extrinsics_w2c: torch.Tensor, intrinsics: Intrinsics, fp16: bool):
    '''

    :param pts:
    :param extrinsics_w2c: shape should be (n,4,4)
    :param intrinsics: (n,3,3)
    :param fp16:
    :return: projected points (n,1,2)
    '''
    p2 = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]).type(torch.float32).to(pts.device)[None, None]
    if is_dataclass(intrinsics):
        if fp16:
            intrinsics_mat = intrinsics.as_matrix_resolution_normalized()
        else:
                intrinsics_mat = intrinsics.to_matrix()
        intrinsics_mat = torch.from_numpy(intrinsics_mat).type(torch.float32).to(pts.device)[None, None]  # (1, 1, 3, 3)
    else:
        intrinsics_mat = intrinsics[0]
        intrinsics_mat = intrinsics_mat.to(pts.device)[None, None]  # (1, 1, 3, 3)
    norm_const = 5e-5 if fp16 else 1e-6

    pts_world_homo = torch.cat([pts, torch.ones_like(pts[:, :, :1])], dim=2)[:, :, :, None]  # (nr, ns, 4, 1)
    if extrinsics_w2c[1].sum() == extrinsics_w2c[100].sum():  # TODO Kapil : remove this hardcoded part
        extrinsics_w2c = extrinsics_w2c[0]
        extrinsics_w2c=extrinsics_w2c[None,None,:,:]
        pts_cam = torch.matmul(extrinsics_w2c, pts_world_homo)
    else:
        pts_cam = torch.matmul(extrinsics_w2c[:,None,:,:], pts_world_homo)
    pts_cam_cv = torch.matmul(p2, pts_cam)
    pixels_homo = torch.matmul(intrinsics_mat, pts_cam_cv[:, :, :3, :])
    norm_const1 = norm_const * sign_no_zeros(pixels_homo[:, :, 2:, 0])
    pixel_locs = pixels_homo[:, :, :2, 0] / (pixels_homo[:, :, 2:, 0] + norm_const1)
    if not torch.isfinite(pixel_locs).all():
        print('gotcha')
    return pixel_locs
def sign_no_zeros(tensor: torch.Tensor):
    tensor_sign = (tensor >= 0).float() * 2 - 1
    return tensor_sign
