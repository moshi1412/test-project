import torch
import numpy as np
import os
from PIL import Image
from argparse import ArgumentParser

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams


# ----------------------------
# orbit camera generator
# ----------------------------
def create_orbit_camera(radius=4.0, elev=30, azim=0, fov=60, image_size=(800, 800)):
    """
    简单绕Y轴轨道相机（斜俯视）
    """

    theta = np.deg2rad(azim)
    phi = np.deg2rad(elev)

    cam_pos = np.array([
        radius * np.cos(phi) * np.cos(theta),
        radius * np.cos(phi) * np.sin(theta),
        radius * np.sin(phi)
    ])

    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0, 0, 1])

    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    view = np.eye(4)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = forward
    view[:3, 3] = cam_pos

    return view


# ----------------------------
# load checkpoint
# ----------------------------
def load_gaussians(ckpt_path, sh_degree=3, opt=None, hyper=None):
    gaussians = GaussianModel(sh_degree, hyper)

    model_params, _ = torch.load(ckpt_path)

    gaussians.training_setup(opt)
    gaussians.restore(model_params, opt)

    # gaussians.eval()
    return gaussians


# ----------------------------
# render single image
# ----------------------------
@torch.no_grad()
def render_view(gaussians, pipe, viewpoint, bg):
    gaussians._deformation = gaussians._deformation.to('cuda')
    out = render(
        viewpoint,
        gaussians,
        pipe,
        bg,
        stage="fine"
    )
    return out["render"]


# ----------------------------
# fake camera object (关键)
# ----------------------------
class DummyCamera:
    def __init__(self, R, T, width=800, height=800, time=0.0, fov=60):
        self.world_view_transform = torch.eye(4).cuda()
        self.full_proj_transform = torch.eye(4).cuda()
        self.camera_center = torch.tensor(T).float().cuda()

        self.image_width = width
        self.image_height = height
        self.FoVx = np.deg2rad(fov)
        self.FoVy = np.deg2rad(fov)

        self.original_image = None
        self.time = time


# ----------------------------
# main orbit render
# ----------------------------
def render_orbit(ckpt, output_dir, steps=60):

    os.makedirs(output_dir, exist_ok=True)

    # ⚠️ 你必须提供 opt / hyper（来自训练代码）
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    opt = op.extract(args)
    hyper = hp.extract(args)
    pipe = pp.extract(args)

    gaussians = load_gaussians(ckpt, opt=opt, hyper=hyper)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    radius = 4.0

    for i in range(steps):
        azim = i * 360.0 / steps
        elev = 25  # 斜俯视角

        view = create_orbit_camera(radius, elev, azim)

        cam = DummyCamera(
            R=view[:3, :3],
            T=view[:3, 3],
            width=800,
            height=800,
            time=0
        )

        image = render_view(gaussians, pipe, cam, bg)

        img = (torch.clamp(image, 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        Image.fromarray(img).save(f"{output_dir}/{i:04d}.png")

    print("done →", output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="./orbit_render")
    parser.add_argument("--steps", type=int, default=1)

    args = parser.parse_args()

    render_orbit(args.ckpt, args.out, args.steps)
    '''
    python visualize_ply.py         --ckpt /mnt/ljy/S3Gaussian/output/waymo_dynamic32/chkpnt_fine_50000.pth         --steps 50000         --out orbit_render
    '''