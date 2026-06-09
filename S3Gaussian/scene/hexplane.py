import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    #方向向量归一化到 [0,1]，用于球谐（SH）编码
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    '''
    空间坐标归一化：把世界坐标映射到网络训练用的 [-1,1] 标准空间
    输入：原始点 + 场景包围盒（AABB）
    输出：归一化后的坐标
    '''
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    '''
    网格采样：根据坐标从网格中采样特征
    输入：网格特征 + 坐标
    输出：采样后的特征
    '''  
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_ms_features(
    pts: torch.Tensor,            # 输入：归一化后的 4D 坐标 [N, 4]
    ms_grids: Collection[Iterable[nn.Module]],  # 多尺度网格（每个尺度6个平面）
    grid_dimensions: int,         # 固定=2，平面维度
    concat_features: bool,        # 多尺度：拼接/相加，这里固定True
    num_levels: Optional[int],    # 使用几个尺度
) -> torch.Tensor:

    # 1. 生成 4D 坐标的所有 2D 平面组合 → 共 6 个：(0,1)(0,2)(0,3)(1,2)(1,3)(2,3)
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    
    # 2. 确定使用多少个尺度
    if num_levels is None:
        num_levels = len(ms_grids)

    # 3. 存储多尺度结果：拼接用列表，相加用数字0
    multi_scale_interp = [] if concat_features else 0.

    # 4. 遍历每一个尺度（粗尺度 → 细尺度）
    #列表里每一个元素 = 一套完整 HexPlane（6 个 2D 平面）
    # 例如 multires=[1,2] → 2 套完整 6 平面：
    # scale 0：倍率 1，6 个平面全部低分辨率
    # scale 1：倍率 2，6 个平面全部高分辨率
    
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.0  # 初始化乘积：1 乘任何数都不变
        
        # 5. 遍历 6 个平面，逐个插值、相乘
        for ci, coo_comb in enumerate(coo_combs):
            # coo_comb = 平面坐标，例如 (0,1)=xy平面
            
            # 从4D坐标中取出当前平面需要的2个坐标 → 插值
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])  # 平面插值
                .view(-1, feature_dim)  # 展平成 [N, 特征维度]
            )
            
            # ✅ HexPlane 核心：6个平面特征 逐元素相乘
            interp_space = interp_space * interp_out_plane

        # 6. 把当前尺度的特征存起来
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    # 7. 多尺度特征拼接（最后维度拼接）
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    
    return multi_scale_interp
class HexPlaneField(nn.Module):
    def __init__(
        self,    
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        # 定义了多个分辨率层次，用于创建多尺度的平面网格
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)
    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]
    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()
        pts = normalize_aabb(pts, self.aabb.to(pts.device))#归一化坐标到[-1,1]空间
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4] 4D编码

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)


        return features

    def forward(self,
                pts: torch.Tensor,#中心点坐标
                timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
