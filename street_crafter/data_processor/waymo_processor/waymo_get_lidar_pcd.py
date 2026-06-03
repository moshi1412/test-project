import os
import numpy as np
import cv2
import math
import argparse
import sys
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from waymo_open_dataset import dataset_pb2, label_pb2
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils.lidar_utils import convert_range_image_to_cartesian
from waymo_open_dataset.v2.perception import (
    box as _v2_box,
    camera_image as _v2_camera_image,
    context as _v2_context,
    lidar as _v2_lidar,
    pose as _v2_pose,
)

from waymo_helpers import load_calibration, load_track, get_object, ParquetReader, TFRecordReader, load_ego_poses
from utils.img_utils import visualize_depth_numpy
from utils.box_utils import bbox_to_corner3d, inbbox_points
from utils.pcd_utils import storePly, fetchPly
from utils.base_utils import project_numpy

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

from typing import Dict, List, Optional, Tuple, TypedDict
import tensorflow as tf
DUMMY_DISTANCE_VALUE = 2e3  # meters, used for missing points
np.set_printoptions(precision=4, suppress=True)
import transforms3d
from copy import deepcopy

def convert_range_image_to_point_cloud_v1(
    range_image,
    calibration,
    pixel_pose=None,
    frame_pose=None,
    keep_polar_features=False,
):
    """Convert range image to point cloud from v1 tfrecord format."""
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    
    # Convert to Cartesian coordinates
    range_image_cartesian = convert_range_image_to_cartesian(
        range_image=range_image_tensor,
        extrinsic=tf.convert_to_tensor(calibration.extrinsic.transform, dtype=tf.float32),
        intrinsic=tf.convert_to_tensor(calibration.intrinsic, dtype=tf.float32),
        keep_polar_features=keep_polar_features,
    )
    
    # Identify missing points
    no_return = range_image_tensor[..., 0] < 0
    range_image_mask = ~no_return
    
    points_tensor = tf.boolean_mask(range_image_cartesian, range_image_mask)
    missing_points_tensor = tf.boolean_mask(range_image_cartesian, ~range_image_mask)
    
    return points_tensor, missing_points_tensor, range_image_mask

def extract_pointwise_camera_projection_v1(
    camera_projection,
    range_image_shape,
):
    """Extract camera projection from v1 tfrecord format."""
    proj_tensor = tf.convert_to_tensor(camera_projection.data)
    proj_tensor = tf.reshape(proj_tensor, camera_projection.shape.dims)
    
    # Create mask from range image (we need to know which points are valid)
    # For v1 format, we assume all projections are valid
    return proj_tensor, None, None
def save_lidar_v1(root_dir, seq_path, seq_save_dir):
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')

    dataset = tf.data.TFRecordDataset(seq_path, compression_type='')
    
    os.makedirs(seq_save_dir, exist_ok=True)
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_dir = os.path.join(seq_save_dir, 'lidar')
    lidar_dir_background = os.path.join(lidar_dir, 'background')
    lidar_dir_actor = os.path.join(lidar_dir, 'actor')
    lidar_dir_depth = os.path.join(lidar_dir, 'depth')
    os.makedirs(lidar_dir_background, exist_ok=True)
    os.makedirs(lidar_dir_actor, exist_ok=True)
    os.makedirs(lidar_dir_depth, exist_ok=True)
    
    pointcloud_actor = {tid: dict(xyz=[], rgb=[], mask=[]) for tid, traj in trajectory.items() if not traj['stationary'] and traj['label'] != 'sign'}
    for tid in pointcloud_actor:
        os.makedirs(os.path.join(lidar_dir_actor, tid), exist_ok=True)

    frame_id = 0
    for data in tqdm(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        # ==============================================
        # ✅ 官方唯一正确点云提取（兼容所有版本）
        # ==============================================
        range_images, camera_projections, seg_labels, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        points_local, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )

        # --------------- 【关键修改：不转到世界坐标系】---------------
        # 直接使用 车体坐标系 点云
        xyzs = np.concatenate(points_local, axis=0)
        projections = np.concatenate(cp_points, axis=0)
        # 车体 -> 世界变换（兼
        # ==============================================

        rgbs = np.zeros((xyzs.shape[0], 3), np.uint8)
        camera_id = projections[:, 0]
        masks = camera_id > 0

        # 上色
        for i in range(5):
            img_fn = os.path.join(image_dir, f'{frame_id:06d}_{i}.png')
            if not os.path.exists(img_fn): img_fn = img_fn.replace('.png', '.jpg')
            if not os.path.exists(img_fn): continue
            im = cv2.imread(img_fn)
            if im is None: continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            m = (camera_id == i+1)
            if not m.any(): continue
            u = np.clip(projections[m,1].round(), 0, w-1).astype(int)
            v = np.clip(projections[m,2].round(), 0, h-1).astype(int)
            rgbs[m] = im[v, u]

        # 分割 actor/背景
        actor_mask = np.zeros(xyzs.shape[0], bool)
        frame_key = f'{frame_id:06d}'
        if frame_key in track_info:
            for tid, obj in track_info[frame_key].items():
                if tid not in pointcloud_actor: continue
                if frame_id not in trajectory[tid]['frames']: continue

                lb = obj['lidar_box']
                l, w, h = lb['length'], lb['width'], lb['height']
                pidx = trajectory[tid]['frames'].index(frame_id)
                pose_v = trajectory[tid]['poses_vehicle'][pidx]

                homo = np.concatenate([xyzs, np.ones((len(xyzs),1))], axis=1)
                xyz_actor = (homo @ np.linalg.inv(pose_v).T)[:, :3]
                bbox = np.array([[-l,-w,-h],[l,w,h]]) * 0.5
                corners = bbox_to_corner3d(bbox)
                in_box = inbbox_points(xyz_actor, corners)

                actor_mask |= in_box
                xyz_in = xyz_actor[in_box]
                rgb_in = rgbs[in_box]
                msk_in = masks[in_box]

                pointcloud_actor[tid]['xyz'].append(xyz_in)
                pointcloud_actor[tid]['rgb'].append(rgb_in)
                pointcloud_actor[tid]['mask'].append(msk_in)
                try:
                    storePly(os.path.join(lidar_dir_actor,tid,f'{frame_id:06d}.ply'), xyz_in, rgb_in, msk_in[:,None])
                except: pass

        # 保存背景
        bg_xyz = xyzs[~actor_mask]
        bg_rgb = rgbs[~actor_mask]
        bg_msk = masks[~actor_mask, None]
        storePly(os.path.join(lidar_dir_background, f'{frame_id:06d}.ply'), bg_xyz, bg_rgb, bg_msk)

        frame_id += 1

    # 保存 actor full.ply
    for tid, p in pointcloud_actor.items():
        if not p['xyz']: continue
        x = np.concatenate(p['xyz'])
        r = np.concatenate(p['rgb'])
        m = np.concatenate(p['mask'])[:,None]
        try:
            storePly(os.path.join(lidar_dir_actor,tid,'full.ply'), x, r, m)
        except: pass
def save_lidar_v2(root_dir, seq_path, seq_save_dir):
    """Process LiDAR data from v2 parquet format (original implementation)."""
    track_info, track_camera_visible, trajectory = load_track(seq_save_dir)
    extrinsics, intrinsics = load_calibration(seq_save_dir)
    print(f'Processing sequence {seq_path}...')
    print(f'Saving to {seq_save_dir}')

    base_seq_name = os.path.basename(seq_path).split('.')[0]
    seq_context = base_seq_name[8:-19]
    seq_reader = ParquetReader(context_name=seq_context, dataset_dir=root_dir)

    lidar_calib = seq_reader("lidar_calibration").compute()
    lidar_proj_df = seq_reader("lidar_camera_projection").compute()
    lidar_df = seq_reader("lidar").compute()
    lidar_pose_df = seq_reader("lidar_pose").compute()
    vehicle_pose_df = seq_reader("vehicle_pose").compute()

    os.makedirs(seq_save_dir, exist_ok=True)
    
    image_dir = os.path.join(seq_save_dir, 'images')
    lidar_dir = os.path.join(seq_save_dir, 'lidar')
    os.makedirs(lidar_dir, exist_ok=True)
    lidar_dir_background = os.path.join(lidar_dir, 'background')
    os.makedirs(lidar_dir_background, exist_ok=True)
    lidar_dir_actor = os.path.join(lidar_dir, 'actor')
    os.makedirs(lidar_dir_actor, exist_ok=True)
    lidar_dir_depth = os.path.join(lidar_dir, 'depth')
    os.makedirs(lidar_dir_depth, exist_ok=True)
    

    pointcloud_actor = dict()
    for track_id, traj in trajectory.items():
        dynamic = not traj['stationary']
        if dynamic and traj['label'] != 'sign':
            os.makedirs(os.path.join(lidar_dir_actor, track_id), exist_ok=True)
            pointcloud_actor[track_id] = dict()
            pointcloud_actor[track_id]['xyz'] = []
            pointcloud_actor[track_id]['rgb'] = []
            pointcloud_actor[track_id]['mask'] = []
    
    print("Processing LiDAR data...")

    for frame_id, (_, v) in tqdm(enumerate(vehicle_pose_df.iterrows())):
        xyzs = []
        camera_projections = []
        missing_xyzs = []
        VehiclePoseCom = v2.VehiclePoseComponent.from_dict(v)
        lidar_df_frame = lidar_df[lidar_df["key.frame_timestamp_micros"] == VehiclePoseCom.key.frame_timestamp_micros]
        
        for _, r in lidar_df_frame.iterrows():
            LidarComp = v2.LiDARComponent.from_dict(r)
            
            lidar_pose_df_ = lidar_pose_df[
                (lidar_pose_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_pose_df["key.laser_name"] ==  LidarComp.key.laser_name)
            ]
            if len(lidar_pose_df_) == 0:
                pixel_pose = None
                frame_pose = None
            else:    
                LidarPoseComp = v2.LiDARPoseComponent.from_dict(lidar_pose_df_.iloc[0])
                pixel_pose = LidarPoseComp.range_image_return1
                frame_pose = VehiclePoseCom

            lidar_proj_df_ = lidar_proj_df[
                (lidar_proj_df["key.frame_timestamp_micros"] == LidarComp.key.frame_timestamp_micros)
                & (lidar_proj_df["key.laser_name"] == LidarComp.key.laser_name)
            ]
            LidarProjComp = v2.LiDARCameraProjectionComponent.from_dict(lidar_proj_df_.iloc[0])
            
            lidar_calib_ = lidar_calib[lidar_calib["key.laser_name"] == LidarComp.key.laser_name]
            LidarCalibComp = v2.LiDARCalibrationComponent.from_dict(lidar_calib_.iloc[0])
            
            pts_lidar, missing_pts, _ = convert_range_image_to_point_cloud(
                LidarComp.range_image_return1,
                LidarCalibComp,
                pixel_pose=pixel_pose,
                frame_pose=frame_pose,
                keep_polar_features=False,
            )
            missing_pts = missing_pts.numpy()

            pts_projection, missing_projection, _ = extract_pointwise_camera_projection(
                LidarComp.range_image_return1,
                LidarProjComp.range_image_return1,
            )
            
            xyzs.append(pts_lidar.numpy())
            camera_projections.append(pts_projection.numpy())
            missing_xyzs.append(missing_pts)

        xyzs = np.concatenate(xyzs, axis=0)
        camera_projections = np.concatenate(camera_projections, axis=0)
        missing_xyzs = np.concatenate(missing_xyzs, axis=0)

        rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.uint8)        
        camera_id = camera_projections[:, 0]
        masks = camera_id > 0
        
        # Generate lidar depth and get pointcloud rgb
        for i in range(5):
            image_filename = os.path.join(image_dir, f'{frame_id:06d}_{i}.png')
            image = cv2.imread(image_filename)[..., [2, 1, 0]].astype(np.uint8)      
            h, w = image.shape[:2]
            
            depth_filename = os.path.join(lidar_dir_depth, f'{frame_id:06d}_{i}.npz')
            depth = (np.ones((h, w)) * np.finfo(np.float32).max).reshape(-1)
            depth_vis_filename = os.path.join(lidar_dir_depth, f'{frame_id:06d}_{i}.png')
            
            # Sprase lidar depth
            num_pts = xyzs.shape[0]
            pts_idx = np.arange(num_pts)
            pts_idx = np.tile(pts_idx[..., None], (1, 2)).reshape(-1) # (num_pts * 2)
            pts_camera_id = camera_projections.reshape(-1, 3)[:, 0] 
            mask_depth_idx = (pts_camera_id == i+1)
            mask_depth = pts_idx[mask_depth_idx]
            
            xyzs_mask = xyzs[mask_depth]
            xyzs_mask_homo = np.concatenate([xyzs_mask, np.ones_like(xyzs_mask[..., :1])], axis=-1)
            
            c2w = extrinsics[i]
            w2c = np.linalg.inv(c2w)
            xyzs_mask_cam = xyzs_mask_homo @ w2c.T
            xyzs_mask_depth = xyzs_mask_cam[..., 2]
            xyzs_mask_depth = np.clip(xyzs_mask_depth, a_min=1e-1, a_max=1e2)
            
            u_depth, v_depth = camera_projections[mask_depth, 1], camera_projections[mask_depth, 2]
            u_depth = np.clip(u_depth, 0, w-1).astype(np.int32)
            v_depth = np.clip(v_depth, 0, h-1).astype(np.int32)
            indices = v_depth * w + u_depth
            
            np.minimum.at(depth, indices, xyzs_mask_depth)
            depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
            valid_depth_pixel = (depth != 0)
            valid_depth_value = depth[valid_depth_pixel].astype(np.float32)
            valid_depth_pixel = valid_depth_pixel.reshape(h, w).astype(np.bool_)
            
            np.savez_compressed(depth_filename, mask=valid_depth_pixel, value=valid_depth_value)
            
            try:
                if i == 0:
                    depth = depth.reshape(h, w).astype(np.float32)
                    depth_vis, _ = visualize_depth_numpy(depth)
                    depth_on_img = image[..., [2, 1, 0]]
                    depth_on_img[depth > 0] = depth_vis[depth > 0]
                    cv2.imwrite(depth_vis_filename, depth_on_img)      
            except:
                print(f'error in visualize depth of {image_filename}, depth range: {depth.min()} - {depth.max()}')
            
            # Colorize 
            mask_rgb = (camera_id == i+1)
            if mask_rgb.sum() != 0:
                u_rgb, v_rgb = camera_projections[mask_rgb, 1], camera_projections[mask_rgb, 2]
                u_rgb = np.clip(u_rgb, 0, w-1).astype(np.int32)
                v_rgb = np.clip(v_rgb, 0, h-1).astype(np.int32)
                rgb = image[v_rgb, u_rgb]
                rgbs[mask_rgb] = rgb
        
        actor_mask = np.zeros(xyzs.shape[0], dtype=np.bool_)
        track_info_frame = track_info[f'{frame_id:06d}']
        for track_id, track_info_actor in track_info_frame.items():
            if track_id not in pointcloud_actor.keys():
                continue
            
            lidar_box = track_info_actor['lidar_box']
            height = lidar_box['height']
            width = lidar_box['width']
            length = lidar_box['length']
            pose_idx = trajectory[track_id]['frames'].index(frame_id)
            pose_vehicle = trajectory[track_id]['poses_vehicle'][pose_idx]

            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]
            
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
            corners3d = bbox_to_corner3d(bbox)
            inbbox_mask = inbbox_points(xyzs_actor, corners3d)
            
            actor_mask = np.logical_or(actor_mask, inbbox_mask)
            
            xyzs_inbbox = xyzs_actor[inbbox_mask]
            rgbs_inbbox = rgbs[inbbox_mask]
            masks_inbbox = masks[inbbox_mask]
            
            pointcloud_actor[track_id]['xyz'].append(xyzs_inbbox)
            pointcloud_actor[track_id]['rgb'].append(rgbs_inbbox)
            pointcloud_actor[track_id]['mask'].append(masks_inbbox)
            
            masks_inbbox = masks_inbbox[..., None]
            ply_actor_path = os.path.join(lidar_dir_actor, track_id, f'{frame_id:06d}.ply')
            try:
                storePly(ply_actor_path, xyzs_inbbox, rgbs_inbbox, masks_inbbox)
            except:
                pass

        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        masks_background = masks_background[..., None]
        ply_background_path = os.path.join(lidar_dir_background, f'{frame_id:06d}.ply')
        
        storePly(ply_background_path, xyzs_background, rgbs_background, masks_background)
    
    for track_id, pointcloud in pointcloud_actor.items():
        xyzs = np.concatenate(pointcloud['xyz'], axis=0)
        rgbs = np.concatenate(pointcloud['rgb'], axis=0)
        masks = np.concatenate(pointcloud['mask'], axis=0)
        masks = masks[..., None]
        ply_actor_path_full = os.path.join(lidar_dir_actor, track_id, 'full.ply')
        
        try:
            storePly(ply_actor_path_full, xyzs, rgbs, masks)
        except:
            pass

def convert_range_image_to_point_cloud(
    range_image: _v2_lidar.RangeImage,
    calibration: _v2_context.LiDARCalibrationComponent,
    pixel_pose: Optional[_v2_lidar.PoseRangeImage] = None,
    frame_pose: Optional[_v2_pose.VehiclePoseComponent] = None,
    keep_polar_features=False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Converts one range image from polar coordinates to point cloud."""
    val_clone = deepcopy(range_image.tensor.numpy())
    no_return = val_clone[..., 0] == -1
    val_clone[..., 0][no_return] = DUMMY_DISTANCE_VALUE
    object.__setattr__(range_image, "values", val_clone.flatten())

    if pixel_pose is not None:
        assert frame_pose is not None
        pixel_pose_clone = deepcopy(pixel_pose.tensor.numpy())
        pixel_pose_mask = pixel_pose_clone[..., 0] == 0
        tr_orig = frame_pose.world_from_vehicle.transform.reshape(4, 4)
        rot = tr_orig[:3, :3]
        x, y, z = tr_orig[:3, 3]
        yaw, pitch, roll = transforms3d.euler.mat2euler(rot, "szyx")
        pixel_pose_clone[..., 0][pixel_pose_mask] = roll
        pixel_pose_clone[..., 1][pixel_pose_mask] = pitch
        pixel_pose_clone[..., 2][pixel_pose_mask] = yaw
        pixel_pose_clone[..., 3][pixel_pose_mask] = x
        pixel_pose_clone[..., 4][pixel_pose_mask] = y
        pixel_pose_clone[..., 5][pixel_pose_mask] = z
        object.__setattr__(pixel_pose, "values", pixel_pose_clone.flatten())

    range_image_cartesian = convert_range_image_to_cartesian(
        range_image=range_image,
        calibration=calibration,
        pixel_pose=pixel_pose,
        frame_pose=frame_pose,
        keep_polar_features=keep_polar_features,
    )

    range_image_tensor = range_image.tensor
    range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]
    points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))
    missing_points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(~range_image_mask))

    return points_tensor, missing_points_tensor, range_image_mask

def extract_pointwise_camera_projection(
    range_image: _v2_lidar.RangeImage,
    camera_projection: _v2_lidar.CameraProjectionRangeImage,
) -> tf.Tensor:
    range_image_tensor = range_image.tensor
    range_image_mask = DUMMY_DISTANCE_VALUE / 2 > range_image_tensor[..., 0]
    camera_project_tensor = camera_projection.tensor
    pointwise_camera_projection_tensor = tf.gather_nd(camera_project_tensor, tf.compat.v1.where(range_image_mask))
    missing_points_camera_projection_tensor = tf.gather_nd(camera_project_tensor, tf.compat.v1.where(~range_image_mask))

    return pointwise_camera_projection_tensor, missing_points_camera_projection_tensor, range_image_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='./test_data/')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--use_v1', action='store_true', help='Use Waymo v1 tfrecord format instead of v2 parquet')
    args = parser.parse_args()
    
    root_dir = args.root_dir
    save_dir = args.save_dir

    all_sequence_names = sorted([x for x in os.listdir(root_dir) if x.endswith('.tfrecord')])
    all_sequence_paths = [os.path.join(root_dir, x) for x in all_sequence_names]
    for i, sequence_path in enumerate(all_sequence_paths):
        print(f'{i}: {sequence_path}')
        sequence_save_dir = os.path.join(save_dir, str(i).zfill(3))
        if os.path.exists(os.path.join(sequence_save_dir, 'lidar/depth')) and args.skip_existing:
            print(f'lidar pcd exists for {sequence_path}, skipping...')
            continue
                
        if args.use_v1:
            save_lidar_v1(
                root_dir=root_dir,
                seq_path=sequence_path,
                seq_save_dir=sequence_save_dir,
            )
        else:
            save_lidar_v2(
                root_dir=root_dir,
                seq_path=sequence_path,
                seq_save_dir=sequence_save_dir,
            )


if __name__ == '__main__':
    main()
