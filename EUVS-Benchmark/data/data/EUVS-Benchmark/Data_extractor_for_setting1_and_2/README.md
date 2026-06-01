# nuPlan Polygon-based Image Extractor

This tool extracts camera images from **nuPlan SQLite driving logs** that fall inside a specified **road‑polygon region**, and exports them into a structured dataset with unified naming.

It is designed for:

- Urban multi‑view reconstruction  
- Novel‑view synthesis  
- Cross‑traversal alignment  
- Multi‑camera autonomous‑driving datasets  

---

## 📁 Output Directory Structure

```
output_root /
 └─ nuplan_{city}_location_{loc}/
     └─ {city}_location_{loc}/
         └─ images/
             ├─ 0030_0006_0001_4.jpg
             ├─ 0030_0006_0002_4.jpg
             ├─ ...
```
---

## 🧩 File Naming Convention

```
{location}_{traversal}_{frame_index}_{camera_id}.jpg
```

Example:

```
0030_0006_0123_4.jpg
```

| Field | Meaning |
|------|--------|
| `0030` | location id |
| `0006` | traversal id |
| `0123` | frame index (sorted by timestamp) |
| `4` | camera id |

---

## 🎥 Camera ID Mapping

| Channel | Camera ID |
|--------|--------|
| L0 | 1 |
| L1 | 2 |
| L2 | 3 |
| F0 | 4 |
| B0 | 5 |
| R0 | 6 |
| R1 | 7 |
| R2 | 8 |

Only cameras listed in `cam_ids` are exported.

---

## ⚙️ Traversal Configuration (`config.json`)

```json
[
  {
    "city": "vegas",
    "location": 30,
    "traversal": 6,
    "db_name": "2021.06.09.12.39.51_veh-26_04543_05321.db",
    "cam_ids": [4, 1, 6],
    "interval": 2
  }
]
```

| Key | Description |
|-----|-------------|
| `city` | nuPlan city split |
| `location` | location id |
| `traversal` | traversal index |
| `db_name` | nuPlan SQLite database file |
| `cam_ids` | camera ids to export |
| `interval` | export every N frames |

Frames are strictly **sorted by timestamp** before sampling.

---

## 🗺 Road Polygon Definition (`road_points.json`)

```json
{
  "vegas": {
    "30": [
      {"x": 123.4, "y": 52.1},
      {"x": 124.9, "y": 53.2}
    ]
  }
}
```

Processing steps:

1. Load polygon for `(city, location)`  
2. Query ego poses inside bounding box  
3. Filter poses using **point‑in‑polygon**  
4. Compute `[min_t , max_t]` timestamps  
5. Retrieve images in this time window  

Only frames **inside polygon** are exported.

---

## 🚀 Run Script

```bash
python extract_images.py   --data_root /path/to/nuplan/db/data/cache   --camera_root /path/to/nuplan/camera   --output_root /path/to/output_dataset   --road_points road_points.json   --config config.json   --max_frames 1600
```

---

## 🧾 Arguments

| Argument | Description |
|--------|--------|
| `--data_root` | path to nuPlan `db/data/cache` |
| `--camera_root` | directory storing nuPlan camera images |
| `--output_root` | output dataset directory |
| `--road_points` | polygon annotation json |
| `--config` | traversal configuration json |
| `--max_frames` | keep last N frames (0 = disable) |

Default:

```
max_frames = 1600
```

---


## 🧱 Dependencies

This script uses lightweight Python libraries:

```
sqlite3
json
shutil
logging
tqdm
```
---


## 📣 Citation

If this data is useful in your research or dataset workflow, please consider citing or acknowledging this project.
