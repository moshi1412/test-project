## Waymo Open Dataset
### Download dataset
First download the 1.4.1 version of Waymo dataset (training and validation set), please refer to [this document](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for more download details. 

Since it is extremely slow for the lidar sensor processing when each scene is stored in the format of `.tfrecord` file, we additionally download part of the 2.0.0 version data. For instance, you need to run the following command under the training set:

``` shell
<<<<<<< HEAD
gsutil -m cp -r "gs://waymo_open_dataset_v_2_0_0/training/lidar" \
=======
gsutil -m cp -r \                                                  
  "gs://waymo_open_dataset_v_2_0_0/training/lidar" \
>>>>>>> bee74afc1408be807bc9aa7e82c2935b972b6baf
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_box" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_calibration" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_projection" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_synced_box" \
  "gs://waymo_open_dataset_v_2_0_0/training/lidar_pose" \
  "gs://waymo_open_dataset_v_2_0_0/training/vehicle_pose" \
  .
```




Install Waymo Development Toolkit
```
pip install waymo-open-dataset-tf-2-11-0==1.6.0
```

### Process dataset (except LiDAR)


``` shell
python waymo_processor/waymo_converter.py \
<<<<<<< HEAD
    --root_dir  /mnt/ljy/EmerNeRF/data/waymo/raw \
    --save_dir  /mnt/ljy/street_crafter/data/waymo \
    --process_list pose calib image track dynamic lidar
=======
    --root_dir  {ROOT_DIR} \
    --save_dir  {SAVE_DIR} \
    --process_list pose calib image track dynamic
>>>>>>> bee74afc1408be807bc9aa7e82c2935b972b6baf
```

The processed dataset would be like:

```shell
ProjectPath/data/
  └── waymo
    └── training_set_processed/
        ├── 000/
        │  ├──images/             # Images: {frame:06d}_{cam_id}.jpg
        │  ├──ego_pose/           # Ego vehicle poses: {frame:06d}.txt
        │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
        │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
        │  ├──dynamic_masks/      # Coarse dynamic masks: {frame:06d}_{cam_id}.png
        │  ├──track/              # Object tracklets
        ├── 001/
        ├── ...
```

### Process dataset (LiDAR)

Convert LiDAR range image into background and actor colored pointcloud and get sparse LiDAR depth for each camera view. 

``` shell
python waymo_processor/waymo_get_lidar_pcd.py \
<<<<<<< HEAD
    --root_dir  /mnt/ljy/EmerNeRF/data/waymo/raw \
    --save_dir  /mnt/ljy/street_crafter/data/waymo
=======
    --root_dir  {ROOT_DIR} \
    --save_dir  {SAVE_DIR} 
>>>>>>> bee74afc1408be807bc9aa7e82c2935b972b6baf
```
    
The processed LiDAR would be like:

```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/lidar/
        │  ├──actor/             # Actor pointcloud
        │  ├──background/        # Background pointcloud: {frame:06d}.ply
        │  ├──depth/             # Sparse LiDAR depth: {frame:06d}_{cam_id}.npz
        ├── 001/lidar/
        ├── ...
```

#### New feature (Predicted pointcloud)
We also provide the script `waymo_get_moge_pcd.py`
for generating predicted pointcloud with geometry foundation model such as [MoGe](https://wangrc.site/MoGePage/). However, we have 
not fully tested the effectiveness of this type of guidance.

### Render LiDAR

Render aggregated LiDAR pointcloud to image plane.

``` shell
<<<<<<< HEAD
python waymo_processor/waymo_render_lidar_pcd_v2.py \
    --data_dir /mnt/ljy/street_crafter/data/waymo/training_set_processed \
    --save_dir /mnt/ljy/street_crafter/data/waymo/training_set_processed \
    --delta_frames 10 \
    --cams 0 \
    --shifts 0
=======
python waymo_processor/waymo_render_lidar_pcd.py \
    --data_dir {DARA_DIR} \
    --save_dir {SAVE_DIR} \
    --delta_frames 10 \
    --cams 0, \
    --shifts 0, 
>>>>>>> bee74afc1408be807bc9aa7e82c2935b972b6baf
```

Parameter explanation:
- `--data_dir`: Path to the folder containing all the processed data where each sequence is in the format of scene id `xxx`.

- `--save_dir`: Path to store the rendering LiDAR condition in each sequence direcotry, default as `color_render` where the rendering condition is saved in `xxx/color_render`.

- `--delta_frames`: Number of nearby frames LiDAR point cloud to aggerate, default as 10 for 1s.

- `--cams`: Camera ids to perform LiDAR condition rendering, default as 0 for only front camera. You can set the parameter to `0 1 2 3 4` to generate surrounding LiDAR condition for all cameras.

- `--shifts`: The shifting meters for each sequence, default as 0 to only generate the LiDAR condition of input trajectory.
You can change this parameter to generate multi-trajectory LiDAR condition. For instance, `--shifts 2 3` will generate conditions for lane-change settings where the camera is laterally shifted for 2 or 3 meters to the left. Noted that the condition of novel trajectory will be automatically generated during the distillation process, you can also directly inference the video model in `render.py` script.

The rendered LiDAR pointcloud would be like:
```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/lidar/color_render
        │  ├──{frame:06d}_{cam_id}.png            # Rendered LiDAR color
        │  ├──{frame:06d}_{cam_id}_depth.npy      # Rendered LiDAR depth
        │  ├──{frame:06d}_{cam_id}_mask.png       # Rendered LiDAR mask
        ├── 001/lidar/color_render
        ├── ...
```



### Prepare meta data

Prepare training and validation meta data for video diffusion model. 
Save meta data for training and validation of streetcrafter. You can also make your own json file to test the view synthesis result.

```shell
python waymo_processor/waymo_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'train' 

python waymo_processor/waymo_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'val' 
```

Parameter explanation:
- `--scene_ids`: specify which sequence to be included in the json file.

- `--postfix`: specify the postfix of json file name.


After running the command, meta data will be saved in root directory with each entry in the following format:
```json
{
    "frames": [
        "training_set_processed/000/images/000000_0.png",
        "....",
        "training_set_processed/000/images/000024_0.png"
    ],
    "guidances": [
        "training_set_processed/000/lidar/color_render/000000_0.png",
        "....",
        "training_set_processed/000/lidar/color_render/000024_0.png"
    ],
    "guidances_mask": [
        "training_set_processed/000/lidar/color_render/000000_0_mask.png",
        "....",
        "training_set_processed/000/lidar/color_render/000024_0_mask.png"
    ]
}
```


The final processed dataset will be like:
```shell
ProjectPath/data/
└── waymo/
    └── training_set_processed/
        ├── 000/
        ├── ...
        ├── 797/
    └── validation_set_processed/
        ├── 000/
        ├── ...
        ├── 201/
    ├── meta_info_train.json # training json file
    ├── meta_info_val.json   # validation json file
```


### Generate sky mask 
This step is only required during the distillation process.

Install GroundingDINO following [this repo](https://github.com/IDEA-Research/GroundingDINO) and download SAM checkpoint from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

```
python waymo_processor/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CKPT
```

<<<<<<< HEAD
### 处理单个场景数据

如果你只想处理部分场景数据（而不是全部），可以参考 EmerNeRF/docs/NOTR.md 的说明下载特定序号的场景。处理单个场景的 LiDAR 数据时，可以使用以下命令：

``` shell
# 使用 Waymo v1 tfrecord 格式处理单个场景
python waymo_processor/waymo_get_lidar_pcd.py \
    --root_dir /mnt/ljy/EmerNeRF/data/waymo/raw \
    --save_dir /mnt/ljy/street_crafter/data/waymo \
    --use_v1
```

**注意事项：**

1. **数据格式选择**：
   - 默认使用 Waymo v2 parquet 格式（需要提前下载 v2 数据）
   - 使用 `--use_v1` 参数切换到 Waymo v1 tfrecord 格式
   - v1 格式需要提前运行 `waymo_converter.py` 生成 images、ego_pose、extrinsics、intrinsics、track 等文件

2. **目录结构**：
   - 确保 `--root_dir` 下存放原始 tfrecord 文件
   - 确保 `--save_dir` 下已有 `waymo_converter.py` 生成的处理后数据（如 `000/images/`, `000/track/` 等）

3. **输出文件**：
   - `lidar/background/{frame:06d}.ply` - 背景点云
   - `lidar/actor/{track_id}/{frame:06d}.ply` - 动态物体点云
   - `lidar/depth/{frame:06d}_{cam_id}.npz` - 稀疏深度图

4. **依赖要求**：
   - 需要安装 `waymo-open-dataset-tf-2-11-0==1.6.0`
   - 需要 TensorFlow 2.11+ 环境

=======
>>>>>>> bee74afc1408be807bc9aa7e82c2935b972b6baf
## PandaSet
The processing code of PandaSet is built upon [drivestudio](https://ziyc.github.io/omnire/).

### Download dataset
Please follow Section 1, 2 of [this document](https://github.com/ziyc/drivestudio/blob/main/docs/Pandaset.md) to download the raw data and install PandaSet Development Toolkit.

### Process dataset
```shell
python pandaset_processor/pandaset_convertor.py \
    --data_root {ROOT_DIR} \
    --target_dir {SAVE_DIR} \  
    --process_keys images lidar_forward calib pose dynamic_masks objects
```

### Render LiDAR

Render aggregated LiDAR pointcloud to image plane.

``` shell
python pandaset_processor/pandaset_render_lidar_pcd.py \
    --data_dir {DARA_DIR} \
    --save_dir {SAVE_DIR} \
    --delta_frames 10 \
    --cams 0, \
    --shifts 0, 
```

### Prepare meta data
```shell
python pandaset_processor/pandaset_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'train' 

python pandaset_processor/pandaset_prepare_meta.py \
    --root_dir {ROOT_DIR} \
    --split 'val' 
```

### Generate sky mask 

```
python pandaset_processor/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CKPT
```

