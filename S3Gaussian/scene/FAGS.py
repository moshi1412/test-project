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
    def __init__(self, feat_dim, n_freqs=16, hidden_dim=128):
        super().__init__()
        self.n_freqs = n_freqs
        # 1. 点级自适应振幅生成器：输入 HexPlane 特征，输出振幅 [N, n_freqs]
        self.amplitude_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_freqs),
            nn.Softplus()   # 保证正数
        )
        # 2. 预定义频率基 (几何级数，覆盖 1Hz 到 8Hz)
        gamma = torch.exp(torch.linspace(0, 3, n_freqs) * math.log(2))
        self.register_buffer("gamma", gamma)   # [n_freqs]
        # 3. 融合特征并预测 {η, Rx, Tx, Δr, Δs}
        # 输入维度 = HexPlane特征 + 2*n_freqs (sin+cos)
        self.D_theta = nn.Sequential(
            nn.Linear(feat_dim + 2 * n_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 + 4 + 3 + 4+3)  # η, Rx, Tx, Δr(4), Δs(3)
        )

    def forward(self, grid_feat, t):
        """
        grid_feat: [N, feat_dim]  来自 HexPlane 的特征
        t: [N, 1]  时间（已归一化到 [0,1] 或原始范围）
        """
        # 振幅
        amps = self.amplitude_net(grid_feat)          # [N, M] M=16
        # 频率基 sin(2πγt), cos(2πγt)
        sin_feat = torch.sin(2 * math.pi * self.gamma.unsqueeze(0) * t)   # [N, M]
        cos_feat = torch.cos(2 * math.pi * self.gamma.unsqueeze(0) * t)   # [N, M]
        # 振幅加权
        weighted_sin = amps * sin_feat
        weighted_cos = amps * cos_feat
        f_fre = torch.cat([weighted_sin, weighted_cos], dim=-1)   # [N, 2M]
        # 拼接 HexPlane 特征与频域特征
        feat_full = torch.cat([grid_feat, f_fre], dim=-1)         # [N, feat_dim+2M] 128+32=160
        out = self.D_theta(feat_full)
        eta = torch.sigmoid(out[:, 0:1])
        Rx_raw = out[:, 1:5]
        Rx = F.normalize(Rx_raw, dim=-1)
        Tx = out[:, 5:8]
        dr = out[:, 8:12]        # 4维
        ds = out[:, 12:15]       # 3维
        return eta, Rx, Tx, dr, ds #1  4 3 4 3

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