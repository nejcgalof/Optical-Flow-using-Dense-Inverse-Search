#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <stdio.h>  
#include "patch.hpp"

using namespace std;

namespace OpticalFlow
{
	Patch::Patch(image_parameters* image_param_in,
		fix_parameters* fix_param_in,
		int patchid_in)
		:
		image_param(image_param_in),
		fix_param(fix_param_in),
		patchid(patchid_in)
	{
		pc = new patchstate;
		pc->pdiff.resize(fix_param->num_points_patch, 1);
		pc->pweight.resize(fix_param->num_points_patch, 1);

		tmp.resize(fix_param->num_points_patch, 1);
		dxx_tmp.resize(fix_param->num_points_patch, 1);
		dyy_tmp.resize(fix_param->num_points_patch, 1);
	}

	Patch::~Patch()
	{
		delete pc;
	}

	void Patch::InitializePatch(Eigen::Map<const Eigen::MatrixXf> * im_ao_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dy_in, const Eigen::Vector2f pt_ref_in)
	{
		im_ao = im_ao_in;
		im_ao_dx = im_ao_dx_in;
		im_ao_dy = im_ao_dy_in;

		pt_ref = pt_ref_in;
		ResetPatch();

		getPatchStaticNNGrad(im_ao->data(), im_ao_dx->data(), im_ao_dy->data(), &pt_ref, &tmp, &dxx_tmp, &dyy_tmp);

		ComputeHessian();
	}

	void Patch::ComputeHessian()
	{
		pc->Hes(0, 0) = (dxx_tmp.array() * dxx_tmp.array()).sum();
		pc->Hes(0, 1) = (dxx_tmp.array() * dyy_tmp.array()).sum();
		pc->Hes(1, 1) = (dyy_tmp.array() * dyy_tmp.array()).sum();
		pc->Hes(1, 0) = pc->Hes(0, 1);
		if (pc->Hes.determinant() == 0)
		{
			pc->Hes(0, 0) += 1e-10;
			pc->Hes(1, 1) += 1e-10;
		}
	}

	void Patch::SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * im_bo_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dy_in)
	{
		im_bo = im_bo_in;
		im_bo_dx = im_bo_dx_in;
		im_bo_dy = im_bo_dy_in;

		ResetPatch();
	}

	void Patch::ResetPatch()
	{
		pc->hasconverged = 0;
		pc->hasoptstarted = 0;

		pc->pt_st = pt_ref;
		pc->pt_iter = pt_ref;

		pc->p_in.setZero();
		pc->p_iter.setZero();
		pc->delta_p.setZero();

		pc->cnt = 0;
		pc->invalid = false;
	}

	void Patch::OptimizeStart(const Eigen::Vector2f p_in_arg)
	{
		pc->p_in = p_in_arg;
		pc->p_iter = p_in_arg;

		// convert from input parameters to 2D query location(s) for patches
		pc->pt_iter = pt_ref + pc->p_iter;    // for optical flow the point displacement and the parameter vector are equivalent

		// save starting location, only needed for outlier check
		pc->pt_st = pc->pt_iter;

		//Check if initial position is already invalid
		if (pc->pt_iter[0] < image_param->tmp_lb || pc->pt_iter[1] < image_param->tmp_lb ||    // check if patch left valid image region
			pc->pt_iter[0] > image_param->tmp_ubw || pc->pt_iter[1] > image_param->tmp_ubh)
		{
			pc->hasconverged = 1;
			pc->pdiff = tmp;
			pc->hasoptstarted = 1;
		}
		else
		{
			pc->cnt = 0; // reset iteration counter
			pc->hasconverged = 0;

			getPatchStaticBil(im_bo->data(), &(pc->pt_iter), &(pc->pdiff));
			if ((pc->cnt > fix_param->iterations)) {
				pc->hasconverged = 1;
			}

			pc->hasoptstarted = 1;
			pc->invalid = false;
		}
	}

	void Patch::OptimizeIter(const Eigen::Vector2f p_in_arg, const bool untilconv)
	{
		if (!pc->hasoptstarted)
		{
			ResetPatch();
			OptimizeStart(p_in_arg);
		}
		int oldcnt = pc->cnt;

		// optimize patch until convergence, or do only one iteration if DIS visualization is used
		while (!(pc->hasconverged || (untilconv == false && (pc->cnt > oldcnt))))
		{
			pc->cnt++;

			// Projection onto sd_images
			pc->delta_p[0] = (dxx_tmp.array() * pc->pdiff.array()).sum();
			pc->delta_p[1] = (dyy_tmp.array() * pc->pdiff.array()).sum();

			pc->delta_p = pc->Hes.llt().solve(pc->delta_p); // solve linear system

			pc->p_iter -= pc->delta_p; // update flow vector

															   // compute patch locations based on new parameter vector
			pc->pt_iter = pt_ref + pc->p_iter;    // for optical flow the point displacement and the parameter vector are equivalent

			// check if patch(es) moved too far from starting location, if yes, stop iteration and reset to starting location
			if ((pc->pt_st - pc->pt_iter).norm() > fix_param->outlierthresh  // check if query patch moved more than >padval from starting location -> most likely outlier
				||
				pc->pt_iter[0] < image_param->tmp_lb || pc->pt_iter[1] < image_param->tmp_lb ||    // check patch left valid image region
				pc->pt_iter[0] > image_param->tmp_ubw || pc->pt_iter[1] > image_param->tmp_ubh)
			{
				pc->p_iter = pc->p_in; // reset
				pc->pt_iter = pt_ref + pc->p_iter;    // for optical flow the point displacement and the parameter vector are equivalent
				pc->hasconverged = 1;
				pc->hasoptstarted = 1;
			}

			getPatchStaticBil(im_bo->data(), &(pc->pt_iter), &(pc->pdiff));
			if ((pc->cnt > fix_param->iterations)) {
				pc->hasconverged = 1;
			}
		}
	}

	// Extract patch on integer position, and gradients, No Bilinear interpolation
	void Patch::getPatchStaticNNGrad(const float* img, const float* img_dx, const float* img_dy,
		const Eigen::Vector2f* mid_in,
		Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e,
		Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmp_dx_in_e,
		Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_dy_in_e)
	{
		float *tmp_in = tmp_in_e->data();
		float *tmp_dx_in = tmp_dx_in_e->data();
		float *tmp_dy_in = tmp_dy_in_e->data();

		Eigen::Vector2i pos;
		Eigen::Vector2i pos_it;

		pos[0] = round((*mid_in)[0]) + image_param->imgpadding;
		pos[1] = round((*mid_in)[1]) + image_param->imgpadding;

		int posxx = 0;

		int lb = -fix_param->patch_size / 2;
		int ub = fix_param->patch_size / 2 - 1;

		for (int j = lb; j <= ub; ++j)
		{
			for (int i = lb; i <= ub; ++i, ++posxx)
			{
				pos_it[0] = pos[0] + i;
				pos_it[1] = pos[1] + j;
				int idx = pos_it[0] + pos_it[1] * image_param->tmp_w;

				tmp_in[posxx] = img[idx];
				tmp_dx_in[posxx] = img_dx[idx];
				tmp_dy_in[posxx] = img_dy[idx];
			}
		}

		// PATCH NORMALIZATION
		if (fix_param->patch_normalization) // Subtract Mean
			tmp_in_e->array() -= (tmp_in_e->sum() / fix_param->num_points_patch);
	}

	// Extract patch on float position with bilinear interpolation, no gradients.
	void Patch::getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e)
	{
		float *tmp_in = tmp_in_e->data();

		Eigen::Vector2f resid;
		Eigen::Vector4f we; // bilinear weight vector
		Eigen::Vector4i pos;
		Eigen::Vector2i pos_it;

		// Compute the bilinear weight vector, for patch without orientation/scale change -> weight vector is constant for all pixels
		pos[0] = ceil((*mid_in)[0] + .00001f); // ensure rounding up to natural numbers
		pos[1] = ceil((*mid_in)[1] + .00001f);
		pos[2] = floor((*mid_in)[0]);
		pos[3] = floor((*mid_in)[1]);

		resid[0] = (*mid_in)[0] - (float)pos[2];
		resid[1] = (*mid_in)[1] - (float)pos[3];
		we[0] = resid[0] * resid[1];
		we[1] = (1 - resid[0])*resid[1];
		we[2] = resid[0] * (1 - resid[1]);
		we[3] = (1 - resid[0])*(1 - resid[1]);

		pos[0] += image_param->imgpadding;
		pos[1] += image_param->imgpadding;

		float * tmp_it = tmp_in;
		const float * img_a, *img_b, *img_c, *img_d, *img_e;

		img_e = img + pos[0] - fix_param->patch_size / 2;

		int lb = -fix_param->patch_size / 2;
		int ub = fix_param->patch_size / 2 - 1;

		for (pos_it[1] = pos[1] + lb; pos_it[1] <= pos[1] + ub; ++pos_it[1])
		{
			img_a = img_e + pos_it[1] * image_param->tmp_w;
			img_c = img_e + (pos_it[1] - 1) * image_param->tmp_w;
			img_b = img_a - 1;
			img_d = img_c - 1;


			for (pos_it[0] = pos[0] + lb; pos_it[0] <= pos[0] + ub; ++pos_it[0],
				++tmp_it, ++img_a, ++img_b, ++img_c, ++img_d)
			{
				(*tmp_it) = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d);
			}
		}
		// PATCH NORMALIZATION
		if (fix_param->patch_normalization) // Subtract Mean
			tmp_in_e->array() -= (tmp_in_e->sum() / fix_param->num_points_patch);
	}

}


