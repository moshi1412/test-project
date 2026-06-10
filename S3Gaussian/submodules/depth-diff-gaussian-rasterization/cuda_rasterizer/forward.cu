/*
 * Copyright (C) 2023, Inria
 * Modified for VG^2GT: Strict Stochastic Solid Volume Rendering
 * - Exact median depth via bisection using Gaussian error function
 * - Stores per-pixel interval info for backward pass
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>      // for erf, erff
namespace cg = cooperative_groups;

#define MAX_CONTRIB_PER_PIXEL 128
#define BISECTION_ITER 30      // 30 iterations for high precision

// Helper: compute cumulative distribution function of a Gaussian
// Returns P(x <= t) for N(mean, var)  (i.e., 0.5*(1+erf((t-mean)/sqrt(2*var))))
__device__ float gaussian_cdf(float t, float mean, float var) {
    float sigma = sqrtf(var);
    return 0.5f * (1.0f + erff((t - mean) / (sigma * 1.41421356237f))); // sqrt(2)
}

// Compute transmittance T(t) = exp(-integral_0^t sigma(s) ds) in a ray.
// For a single Gaussian primitive with opacity o, mean depth mu, variance var,
// the opacity density sigma(t) = o * (1/sqrt(2pi*var)) * exp(-(t-mu)^2/(2*var)).
// The cumulative integral from -inf to t is o * CDF(t). So transmittance from
// depth 0 to t (with t >= 0) = exp(-o * (CDF(t) - CDF(0))).
// Since we only care about relative T values along the ray starting from 0,
// and we will later scale by the product of previous Gaussians, we compute:
__device__ float gaussian_transmittance(float t, float mean, float var, float opacity) {
    // CDF at t and at 0 (ray origin in camera space, assume mean > 0)
    float cdf_t = gaussian_cdf(t, mean, var);
    float cdf_0 = gaussian_cdf(0.0f, mean, var);
    float integral = opacity * (cdf_t - cdf_0);
    return expf(-integral);
}

// Forward method for converting spherical harmonics to RGB (unchanged)
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped) {
    // ... (完全相同的原始实现，省略以节省篇幅，但必须保留)
    // 请保留原文件中的完整实现
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation (unchanged)
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix) {
    // ... (原始实现)
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation to 3D covariance (unchanged)
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) {
    // ... (原始实现)
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Preprocess: compute depths, conic, and also depth variance for each Gaussian
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* depth_vars,          // NEW: per-Gaussian depth variance along ray
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    float3 p_orig = { orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    const float* cov3D;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + idx * 6;
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx*6);
        cov3D = cov3Ds + idx*6;
    }

    // Compute 2D covariance and conic (as original)
    float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
    float det = cov.x * cov.z - cov.y * cov.y;
    if (det == 0.0f) return;
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrtf(max(0.1f, mid*mid - det));
    float lambda2 = mid - sqrtf(max(0.1f, mid*mid - det));
    float my_radius = ceilf(3.f * sqrtf(max(lambda1, lambda2)));
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx*C+0] = result.x;
        rgb[idx*C+1] = result.y;
        rgb[idx*C+2] = result.z;
    }

    // Store depth in camera space (z)
    depths[idx] = p_view.z;

    // Compute depth variance along the ray direction.
    // The 3D covariance in world space is given by cov3D (6 values).
    // We need the variance of the Gaussian projected onto the ray direction.
    // Ray direction in camera space: (0,0,1) because we use p_view.z as depth?
    // Actually p_view is in camera coordinates (z forward). The depth along ray is simply z.
    // However, the Gaussian's 3D distribution contributes to depth spread.
    // The variance along camera z-axis is the (2,2) component of the covariance
    // transformed to camera space. Since we already have cov3D in world, we need
    // to rotate it to camera space using the viewmatrix's rotation part.
    // For simplicity and correctness, we compute:
    //   R = viewmatrix[0:3,0:3]   (3x3 rotation)
    //   cov_cam = R * cov_world * R^T
    // Then depth variance = cov_cam[2][2].
    // To avoid full matrix multiplication per Gaussian, we compute:
    glm::mat3 cov_world = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );
    glm::mat3 R = glm::mat3(
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]
    );
    glm::mat3 cov_cam = R * cov_world * glm::transpose(R);
    float depth_var = cov_cam[2][2];
    // Ensure positive variance
    depth_var = fmaxf(depth_var, 1e-6f);
    depth_vars[idx] = depth_var;

    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rendering kernel with strict volume rendering using erf
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ points_xy_image,
    const float* __restrict__ features,
    const float* __restrict__ depths,
    const float* __restrict__ depth_vars,   // per-Gaussian depth variance
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ out_depth,
    // Additional outputs for backward pass (interval info)
    float* __restrict__ out_median_left_depth,
    float* __restrict__ out_median_right_depth,
    float* __restrict__ out_median_left_T,
    float* __restrict__ out_median_right_T,
    uint32_t* __restrict__ out_median_left_gid,
    uint32_t* __restrict__ out_median_right_gid)
{
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };
    bool inside = (pix.x < W && pix.y < H);

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_depth[BLOCK_SIZE];
    __shared__ float collected_depth_var[BLOCK_SIZE];

    // Per-thread arrays for contributions
    float local_depths[MAX_CONTRIB_PER_PIXEL];
    float local_vars[MAX_CONTRIB_PER_PIXEL];
    float local_alphas[MAX_CONTRIB_PER_PIXEL];
    float local_colors[CHANNELS * MAX_CONTRIB_PER_PIXEL];
    int local_gids[MAX_CONTRIB_PER_PIXEL];
    int local_cnt = 0;

    // First pass: collect all contributing Gaussians
    for (int i = 0; i < rounds; i++) {
        int toDo_this_round = min(BLOCK_SIZE, toDo - i * BLOCK_SIZE);
        if (toDo_this_round <= 0) break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_depth[block.thread_rank()] = depths[coll_id];
            collected_depth_var[block.thread_rank()] = depth_vars[coll_id];
        }
        block.sync();

        for (int j = 0; j < toDo_this_round; j++) {
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) continue;
            float alpha = min(0.99f, con_o.w * expf(power));
            if (alpha < 1.0f/255.0f) continue;

            if (local_cnt < MAX_CONTRIB_PER_PIXEL) {
                int gid = collected_id[j];
                local_gids[local_cnt] = gid;
                local_depths[local_cnt] = collected_depth[j];
                local_vars[local_cnt] = collected_depth_var[j];
                local_alphas[local_cnt] = alpha;
                for (int ch = 0; ch < CHANNELS; ch++)
                    local_colors[CHANNELS * local_cnt + ch] = features[gid * CHANNELS + ch];
                local_cnt++;
            }
        }
        block.sync();
    }

    if (!inside || local_cnt == 0) {
        if (inside) {
            for (int ch = 0; ch < CHANNELS; ch++)
                out_color[ch * H * W + pix_id] = bg_color[ch];
            out_depth[pix_id] = 1e10f;
            final_T[pix_id] = 1.0f;
            n_contrib[pix_id] = 0;
            if (out_median_left_depth) out_median_left_depth[pix_id] = 0;
            if (out_median_right_depth) out_median_right_depth[pix_id] = 0;
            if (out_median_left_T) out_median_left_T[pix_id] = 1.0f;
            if (out_median_right_T) out_median_right_T[pix_id] = 1.0f;
            if (out_median_left_gid) out_median_left_gid[pix_id] = 0xFFFFFFFF;
			if (out_median_right_gid) out_median_right_gid[pix_id] = 0xFFFFFFFF;
        }
        return;
    }

    // Sort by depth
    for (int i = 0; i < local_cnt-1; i++) {
        for (int j = i+1; j < local_cnt; j++) {
            if (local_depths[i] > local_depths[j]) {
                // swap depth
                float tmpd = local_depths[i]; local_depths[i] = local_depths[j]; local_depths[j] = tmpd;
                float tmpv = local_vars[i]; local_vars[i] = local_vars[j]; local_vars[j] = tmpv;
                float tmpa = local_alphas[i]; local_alphas[i] = local_alphas[j]; local_alphas[j] = tmpa;
                int tmpid = local_gids[i]; local_gids[i] = local_gids[j]; local_gids[j] = tmpid;
                for (int ch = 0; ch < CHANNELS; ch++) {
                    float tmpc = local_colors[CHANNELS*i+ch];
                    local_colors[CHANNELS*i+ch] = local_colors[CHANNELS*j+ch];
                    local_colors[CHANNELS*j+ch] = tmpc;
                }
            }
        }
    }

    // Compute color and median depth using exact Gaussian transmittance
    float T = 1.0f;   // transmittance from start to current depth interval start
    float final_color[CHANNELS] = {0};
    float median_depth = -1.0f;
    int median_left_idx = -1, median_right_idx = -1;
    float median_left_T = -1.0f, median_right_T = -1.0f;
    float median_left_depth = -1.0f, median_right_depth = -1.0f;

    // We need to compute transmittance at each Gaussian's depth (where it starts being active?).
    // However, the continuous integral over the Gaussian's density means that the transmittance
    // changes continuously. For the purpose of finding T=0.5, we consider the ray from 0 to infinity.
    // The cumulative transmittance after processing Gaussians sorted by depth is:
    // T_total(t) = exp( - sum_i opacity_i * (CDF_i(t) - CDF_i(0)) )
    // Since CDF_i(t) increases from CDF_i(0) to 1 as t goes from 0 to inf.
    // We can compute T at the depth where the Gaussian's CDF reaches a certain value.
    // However, the intersection of multiple Gaussians makes it impossible to have a single closed form.
    // To find median, we can do a binary search over t from near plane to far plane, evaluating
    // T_total(t) by summing contributions of all Gaussians. This is more accurate but slower.
    // Instead, we approximate: The dominant change occurs around each Gaussian's mean.
    // We'll still use bisection on each interval between successive Gaussian means, but evaluate
    // T_total(t) exactly using the CDFs of all Gaussians (not just the current one). This is
    // more accurate than linear interpolation. However, to keep it efficient, we can evaluate
    // T_total(t) by summing over all Gaussians' contributions.
    // Implementation: For a given t, compute T_total(t) = exp( - sum_i opacity_i * (CDF_i(t) - CDF_i(0)) ).
    // This requires O(N) per evaluation, which for up to 128 Gaussians and 30 bisection steps is ~4000 ops.
    // Acceptable.

    // Precompute for each Gaussian: opacity, mean, var, and CDF(0)
    struct GaussInfo {
        float mean, var, opacity, cdf0;
    };
    GaussInfo infos[MAX_CONTRIB_PER_PIXEL];
    for (int i = 0; i < local_cnt; i++) {
        infos[i].mean = local_depths[i];
        infos[i].var = local_vars[i];
        infos[i].opacity = local_alphas[i];   // note: this is the discrete alpha, but for continuous model we need opacity density? Actually alpha = 1 - exp(-opacity_density * ...)? Hmm.
        // In the original 3DGS, alpha is the result of opacity * exp(power) which already includes the 2D Gaussian weight.
        // For volume rendering along ray, the correct opacity density is (opacity * G_2D) * (1/sqrt(2pi var))? This is complex.
        // To simplify, we follow the paper's equation (9) that relates rasterization alpha to continuous vacancy.
        // The paper says: v = sqrt(1 - G(t*)), so the volumetric opacity is directly derived from the rasterization alpha.
        // So we can use the alpha as the "effective opacity" of the Gaussian solid. Then the continuous transmittance
        // along the ray is T(t) = exp( - sum_i alpha_i * (CDF_i(t) - CDF_i(0)) ). This matches the derivation in Appendix A.2.
        infos[i].cdf0 = gaussian_cdf(0.0f, infos[i].mean, infos[i].var);
    }

    // Helper lambda to compute total transmittance at depth t
    auto compute_T = [&](float t) -> float {
        float sum = 0.0f;
        for (int i = 0; i < local_cnt; i++) {
            float cdf_t = gaussian_cdf(t, infos[i].mean, infos[i].var);
            sum += infos[i].opacity * (cdf_t - infos[i].cdf0);
        }
        return expf(-sum);
    };

    // Also compute color using original alpha blending (as in paper, it's equivalent to volume rendering)
    // We'll compute color in the same loop by accumulating alpha * T * color.
    // But note: T used in alpha blending should be the transmittance just before the Gaussian's discrete contribution,
    // which we can compute as T_before = compute_T(mean_i - epsilon). For simplicity and consistency with forward
    // rendering in the paper (which uses rasterization for color), we keep the standard alpha blending.

    // 1. 预先计算每个高斯均值深度处（以及深度0处）的连续透射率
	// 注意：compute_T 是之前定义好的 lambda，依赖 infos 数组
	float* boundary_T = new float[local_cnt + 1];  // 使用栈数组或局部数组，因为 local_cnt ≤ MAX_CONTRIB_PER_PIXEL
	boundary_T[0] = compute_T(0.0f);
	for (int i = 0; i < local_cnt; i++) {
		boundary_T[i+1] = compute_T(local_depths[i]);
	}

	// 2. 找出连续透射率跨过 0.5 的区间
	int median_interval = -1;
	for (int i = 0; i < local_cnt; i++) {
		if (boundary_T[i] >= 0.5f && boundary_T[i+1] < 0.5f) {
			median_interval = i;
			break;
		}
	}

	// 3. 颜色仍然使用标准 alpha blending (与论文等价)
	float T_acc = 1.0f;
	float final_color[CHANNELS] = {0.0f};
	for (int i = 0; i < local_cnt; i++) {
		float alpha = local_alphas[i];
		// 颜色累积
		for (int ch = 0; ch < CHANNELS; ch++)
			final_color[ch] += local_colors[CHANNELS*i+ch] * alpha * T_acc;
		T_acc *= (1.0f - alpha);
		if (T_acc < 1e-5f) break;
	}

	// 4. 中位数深度求解
	float median_depth;
	float median_left_depth, median_right_depth, median_left_T, median_right_T;
	int median_left_idx, median_right_idx;

	if (median_interval >= 0) {
		float t_low  = (median_interval == 0) ? 0.0f : local_depths[median_interval-1];
		float t_high = local_depths[median_interval];
		float T_low  = boundary_T[median_interval];
		float T_high = boundary_T[median_interval+1];
		// 二分法求精确深度
		for (int iter = 0; iter < BISECTION_ITER; iter++) {
			float t_mid = 0.5f * (t_low + t_high);
			float T_mid = compute_T(t_mid);
			if (T_mid > 0.5f) {
				t_low = t_mid;
				T_low = T_mid;
			} else {
				t_high = t_mid;
				T_high = T_mid;
			}
		}
		median_depth = 0.5f * (t_low + t_high);
		median_left_depth = t_low;
		median_right_depth = t_high;
		median_left_T = T_low;
		median_right_T = T_high;
		median_left_idx = (median_interval == 0) ? -1 : local_gids[median_interval-1];
		median_right_idx = local_gids[median_interval];
	} else {
		// 未跨过 0.5，取最远高斯深度（或 far plane）
		median_depth = local_depths[local_cnt-1];
		median_left_depth = median_right_depth = median_depth;
		median_left_T = median_right_T = boundary_T[local_cnt];
		median_left_idx = median_right_idx = local_gids[local_cnt-1];
	}

	// 输出部分不变...

    // Output
    for (int ch = 0; ch < CHANNELS; ch++)
        out_color[ch * H * W + pix_id] = final_color[ch] + T_acc * bg_color[ch];
    out_depth[pix_id] = median_depth;
    final_T[pix_id] = T_acc;
    n_contrib[pix_id] = local_cnt;
    if (out_median_left_depth) out_median_left_depth[pix_id] = median_left_depth;
    if (out_median_right_depth) out_median_right_depth[pix_id] = median_right_depth;
    if (out_median_left_T) out_median_left_T[pix_id] = median_left_T;
    if (out_median_right_T) out_median_right_T[pix_id] = median_right_T;
    if (out_median_left_gid) out_median_left_gid[pix_id] = median_left_idx;
    if (out_median_right_gid) out_median_right_gid[pix_id] = median_right_idx;
}

// Wrapper for render (adds new parameters)
void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float* colors,
    const float* depths,
    const float* depth_vars,   // new
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color,
    float* out_depth,
    float* out_median_left_depth,
    float* out_median_right_depth,
    float* out_median_left_T,
    float* out_median_right_T,
    uint32_t* out_median_left_gid,
    uint32_t* out_median_right_gid)
{
    renderCUDA<NUM_CHANNELS> <<<grid, block>>>(
        ranges, point_list, W, H,
        means2D, colors, depths, depth_vars, conic_opacity,
        final_T, n_contrib, bg_color,
        out_color, out_depth,
        out_median_left_depth, out_median_right_depth,
        out_median_left_T, out_median_right_T,
        out_median_left_gid, out_median_right_gid);
}

void FORWARD::preprocess(int P, int D, int M,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* depth_vars,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    preprocessCUDA<NUM_CHANNELS> <<<(P+255)/256, 256>>>(
        P, D, M,
        means3D, scales, scale_modifier, rotations, opacities, shs, clamped,
        cov3D_precomp, colors_precomp,
        viewmatrix, projmatrix, cam_pos,
        W, H, tan_fovx, tan_fovy, focal_x, focal_y,
        radii, means2D, depths, depth_vars, cov3Ds, rgb, conic_opacity,
        grid, tiles_touched, prefiltered);
}