---
license: apache-2.0
tags:
- code
pretty_name: Extrapolated Urban View Synthesis
size_categories:
- 100K<n<1M
---


# Dataset Summary
Description: 
  This dataset comprises 104 urban scenes, featuring both **extrapolated** and **interpolated** camera poses.

# Dataset Structure
Dataset_structure: 
  For each scene, four main components are:
  - `images`: Images of each scene.
  - `sparse`: COLMAP format camera poses and sparse point clouds produced by SFM.
  - `training_set.txt`: Image names in the training set.
  - `test_set.txt`: Image names in the test set.

# Supported Tasks
Supported_tasks: 
  The dataset is suited for tasks such as:
  - Novel View Synthesis
  - Extrapolated View Synthesis
  - 3D Reconstruction

# TODO
- Example usage
