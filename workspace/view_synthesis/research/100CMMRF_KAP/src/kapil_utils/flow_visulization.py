import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path


this_filepath = Path(__file__)
this_filename = this_filepath.stem


def projection_2d(world_points, extrinsic:torch.Tensor,intrinsics:torch.Tensor):
    """
    Args:
    world_points: torch.Tensor of shape (N,3) where N is the number of points
    extrinsic: torch.Tensor of shape (4,4) representing the extrinsic matrix
    intrinsics: torch.Tensor of shape (3,3) representing the intrinsics matrix
    """
    world_points = torch.cat([world_points, torch.ones(world_points.shape[0],1)], dim=1)
    world_points = world_points.t()
    projected_points = intrinsics @ extrinsic @ world_points
    projected_points = projected_points[:2]/projected_points[2]
    return projected_points.t()
