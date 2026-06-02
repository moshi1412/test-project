让我查看这两个脚本文件的内容来分析它们的区别：


        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/test_magicdrive.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/inference_magicdrive.py
          
让我继续查看这两个脚本的主要部分来了解它们的用途：



        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/test_magicdrive.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/inference_magicdrive.py
          
让我继续查看这两个脚本的核心逻辑部分：



        
            
toolName: search_by_regex
            
status: success
          
query: def main\(\)|for.*dataloader|save_sample
          

        
            
toolName: search_by_regex
            
status: success
          
query: def main\(\)|for.*dataloader|save_sample
          
让我查看两个脚本的主要循环和处理逻辑：



        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/test_magicdrive.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/ljy/MagicDrive-V2/scripts/inference_magicdrive.py
          
根据对两个脚本的分析，我来总结它们的主要区别：

## 核心区别

| 特性 | `test_magicdrive.py` | `inference_magicdrive.py` |
|------|----------------------|---------------------------|
| **主要用途** | 模型测试/验证 | 交互式推理/编辑 |
| **数据处理** | 从数据集加载，支持多种验证索引 | 支持编辑注释（删除框、过滤类别等） |
| **保存模式** | single-view / all-in-one / image_filename | 主要是 all-in-one 模式 |
| **场景控制** | 无 | 支持 `force_daytime`、`force_rainy`、`force_night` |
| **Mask策略** | 无 | 支持 `apply_mask_strategy` |
| **Negative Prompts** | 无 | 支持 |
| **VAE分块** | 无 | 支持 `vae_tiling` 参数 |
| **分布式设置** | 更复杂（sp_size配置） | 相对简单 |
| **编辑功能** | 无 | 支持 `del_box_ratio`、`drop_nearest_car`、`allow_class` |

## 详细对比

### 1. **test_magicdrive.py** - 测试验证脚本

**适用场景**：训练完成后对模型进行批量测试和验证

**主要特点**：
- 从验证数据集加载数据进行推理
- 同时保存生成视频和原始GT（Ground Truth）视频用于对比
- 支持 `validation_index` 参数选择测试数据（even/odd/all/具体索引）
- 支持三种保存模式：
  - `single-view`: 每个视角单独保存
  - `all-in-one`: 6个视角拼接成一个视频
  - `image_filename`: 按原始文件名保存为图片
- 更完善的分布式训练支持（sequence parallelism）

### 2. **inference_magicdrive.py** - 交互式推理脚本

**适用场景**：交互式编辑和推理，支持场景修改

**主要特点**：
- 支持**编辑注释**：
  - `del_box_ratio`: 删除一定比例的边界框
  - `drop_nearest_car`: 删除最近的车辆
  - `allow_class`: 只保留指定类别的物体
- 支持**场景强制转换**：
  - `force_daytime`: 强制转换为白天场景
  - `force_rainy`: 强制转换为雨天场景
  - `force_night`: 强制转换为夜晚场景
- 支持**Negative Prompts**（负面提示词）
- 支持**VAE分块解码**（`vae_tiling`）以节省GPU内存
- 使用**Mask策略**进行条件生成
- 提供 `TILING_PARAM` 配置支持不同分辨率

## 使用建议

- 如果需要**批量测试模型性能**，对比生成结果与GT，使用 `test_magicdrive.py`
- 如果需要**交互式编辑场景**、修改物体、转换天气条件，使用 `inference_magicdrive.py`