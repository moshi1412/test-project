import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda")

# ==============================
# 1. FDGK 频域高斯内核
# ==============================
class FDGK(nn.Module):
    def __init__(self, num_gaussians):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.full((num_gaussians,), 0.5, device=DEVICE))
        self.beta = nn.Parameter(torch.full((num_gaussians,), 0.5, device=DEVICE))

    def get_params(self):
        lam = torch.clamp(self.lambda_, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)
        return lam, beta

# ==============================
# 2. 傅里叶高频形变网络
# 作用：给 Hexplane 输出做高频残差修正
# ==============================
class FourierHighFreqDeform(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden = hidden_dim

        # 对 xyz + t 做轻量编码
        self.embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 傅里叶频率
        self.register_buffer("freqs", torch.exp(torch.linspace(math.log(1), math.log(32), 32)))

        # 输出：delta_pos (3) + delta_rot (4) + delta_scale (3)
        self.out = nn.Linear(hidden_dim + 64, 10)

    def forward(self, xyz, t):
        B = xyz.shape[0]
        xyt = torch.cat([xyz, t.view(B,1)], dim=-1)

        # 基础特征
        feat = self.embed(xyt)

        # 傅里叶时序特征
        t_ = t.view(B,1)
        sin = torch.sin(2 * math.pi * self.freqs.view(1,-1) * t_)
        cos = torch.cos(2 * math.pi * self.freqs.view(1,-1) * t_)
        fre = torch.cat([sin, cos], dim=-1)

        # 融合
        full = torch.cat([feat, fre], dim=-1)
        deform = self.out(full)

        dx = deform[..., 0:3]
        dr = deform[..., 3:7]
        ds = deform[..., 7:10]
        return dx, dr, ds

# ==============================
# 3. 频域损失 FFT Loss
# ==============================
def frequency_loss(render, gt):
    # [B,3,H,W]
    fr = torch.fft.rfft2(render, norm='ortho')
    fg = torch.fft.rfft2(gt, norm='ortho')
    ar = torch.abs(fr)
    ag = torch.abs(fg)
    return F.l1_loss(ar, ag)