/*
 * Copyright (C) 2023, Inria
 * Modified for VG^2GT backward pass with median depth interval info.
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    // 渲染反向传播（新增区间参数）
    // backward.h 中的 render 声明（增加最后一个参数）
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* depth_vars,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixel_depths,
		const float* median_left_depth,
		const float* median_right_depth,
		const float* median_left_T,
		const float* median_right_T,
		const uint32_t* median_left_gid,
		const uint32_t* median_right_gid,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths,
		float* dL_ddepthvar);   // 新增：方差梯度    // 注意：这里 dL_ddepths 是相对于深度（高斯中心深度）的梯度

    // 预处理反向传播（新增 depth_vars 相关）
    void preprocess(
        int P, int D, int M,
        const float3* means,
        const int* radii,
        const float* shs,
        const bool* clamped,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float* view,
        const float* proj,
        const float focal_x, float focal_y,
        const float tan_fovx, float tan_fovy,
        const glm::vec3* campos,
        const float3* dL_dmean2D,
        const float* dL_dconics,
        glm::vec3* dL_dmeans,
        float* dL_dcolor,
        float* dL_ddepth,
        float* dL_dcov3D,
        float* dL_dsh,
		const float* dL_ddepthvar, 
        glm::vec3* dL_dscale,
        glm::vec4* dL_drot);
}

#endif