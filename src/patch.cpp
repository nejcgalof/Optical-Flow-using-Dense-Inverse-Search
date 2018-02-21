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

namespace OFC
{

	PatClass::PatClass(
		const camparam* cpt_in,
		const camparam* cpo_in,
		const optparam* op_in,
		const int patchid_in)
		:
		cpt(cpt_in),
		cpo(cpo_in),
		op(op_in),
		patchid(patchid_in)
	{
		pc = new patchstate;
		CreateStatusStruct(pc);

		tmp.resize(op->novals, 1);
		dxx_tmp.resize(op->novals, 1);
		dyy_tmp.resize(op->novals, 1);
	}

	void PatClass::CreateStatusStruct(patchstate * psin)
	{
		// get reference / template patch
		psin->pdiff.resize(op->novals, 1);
		psin->pweight.resize(op->novals, 1);
	}

	PatClass::~PatClass()
	{
		delete pc;
	}

	void PatClass::InitializePatch(Eigen::Map<const Eigen::MatrixXf> * im_ao_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dy_in, const Eigen::Vector2f pt_ref_in)
	{
		im_ao = im_ao_in;
		im_ao_dx = im_ao_dx_in;
		im_ao_dy = im_ao_dy_in;

		pt_ref = pt_ref_in;
		ResetPatch();

		getPatchStaticNNGrad(im_ao->data(), im_ao_dx->data(), im_ao_dy->data(), &pt_ref, &tmp, &dxx_tmp, &dyy_tmp);

		ComputeHessian();
	}

	void PatClass::ComputeHessian()
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

	void PatClass::SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * im_bo_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dy_in)
	{
		im_bo = im_bo_in;
		im_bo_dx = im_bo_dx_in;
		im_bo_dy = im_bo_dy_in;

		ResetPatch();
	}

	void PatClass::ResetPatch()
	{
		pc->hasconverged = 0;
		pc->hasoptstarted = 0;

		pc->pt_st = pt_ref;
		pc->pt_iter = pt_ref;

		pc->p_in.setZero();
		pc->p_iter.setZero();
		pc->delta_p.setZero();

		pc->delta_p_sqnorm = 1e-10;
		pc->delta_p_sqnorm_init = 1e-10;
		pc->mares = 1e20;
		pc->mares_old = 1e20;
		pc->cnt = 0;
		pc->invalid = false;
	}

	void PatClass::OptimizeStart(const Eigen::Vector2f p_in_arg)
	{
		pc->p_in = p_in_arg;
		pc->p_iter = p_in_arg;

		// convert from input parameters to 2D query location(s) for patches
		paramtopt();

		// save starting location, only needed for outlier check
		pc->pt_st = pc->pt_iter;

		//Check if initial position is already invalid
		if (pc->pt_iter[0] < cpt->tmp_lb || pc->pt_iter[1] < cpt->tmp_lb ||    // check if patch left valid image region
			pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh)
		{
			pc->hasconverged = 1;
			pc->pdiff = tmp;
			pc->hasoptstarted = 1;
		}
		else
		{
			pc->cnt = 0; // reset iteration counter
			pc->delta_p_sqnorm = 1e-10;
			pc->delta_p_sqnorm_init = 1e-10;  // set to arbitrary low value, s.t. that loop condition is definitely true on first iteration
			pc->mares = 1e5;          // mean absolute residual
			pc->mares_old = 1e20; // for rate of change, keep mares from last iteration in here. Set high so that loop condition is definitely true on first iteration
			pc->hasconverged = 0;

			OptimizeComputeErrImg();

			pc->hasoptstarted = 1;
			pc->invalid = false;
		}
	}

	void PatClass::OptimizeIter(const Eigen::Vector2f p_in_arg, const bool untilconv)
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
			paramtopt();

			// check if patch(es) moved too far from starting location, if yes, stop iteration and reset to starting location
			if ((pc->pt_st - pc->pt_iter).norm() > op->outlierthresh  // check if query patch moved more than >padval from starting location -> most likely outlier
				||
				pc->pt_iter[0] < cpt->tmp_lb || pc->pt_iter[1] < cpt->tmp_lb ||    // check patch left valid image region
				pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh)
			{
				pc->p_iter = pc->p_in; // reset
				paramtopt();
				pc->hasconverged = 1;
				pc->hasoptstarted = 1;
			}

			OptimizeComputeErrImg();
		}
	}

	inline void PatClass::paramtopt()
	{  
		pc->pt_iter = pt_ref + pc->p_iter;    // for optical flow the point displacement and the parameter vector are equivalent

	}

	void PatClass::LossComputeErrorImage(Eigen::Matrix<float, Eigen::Dynamic, 1>* patdest, Eigen::Matrix<float, Eigen::Dynamic, 1>* wdest, const Eigen::Matrix<float, Eigen::Dynamic, 1>* patin, const Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmpin)
	{
		float * pd = (float*)patdest->data(),
			*pa = (float*)patin->data(),
			*te = (float*)tmpin->data(),
			*pw = (float*)wdest->data();

		if (op->costfct == 0) // L2 cost function
		{
			for (int i = op->novals / 4; i--; ++pd, ++pa, ++te, ++pw)
			{
				(*pd) = (*pa) - (*te);  // difference image
				//(*pw) = __builtin_ia32_andnps(op->negzero, (*pd));
				/*
				r0 := ~a0 & b0
				r1 := ~a1 & b1
				r2 := ~a2 & b2
				r3 := ~a3 & b3
				*/
				/*for (int la = 0; la < 4; la++) {
					unsigned char *ngz = reinterpret_cast<unsigned char *>(op->negzero);
					unsigned char *pd2 = reinterpret_cast<unsigned char *>(pd);
					pw[la] = ~ ngz[la] & pd2[la];
				}*/
				unsigned char *ngz = reinterpret_cast<unsigned char *>(op->negzero);
				unsigned char *pd2 = reinterpret_cast<unsigned char *>(pd);
				*pw = ~(*ngz & *pd2);
				//_mm256_store_ps(pw, _mm256_andnot_ps(_mm256_load_ps(op->negzero), _mm256_load_ps(pd)));
				//_mm256_store_ps(pw, _mm256_andnot_ps(_mm256_load_ps( op->negzero),_mm256_load_ps(pd)));
			}
		}
		else {
			std::cout << "not implement" << std::endl;
		}
	}

	void PatClass::OptimizeComputeErrImg()
	{
		getPatchStaticBil(im_bo->data(), &(pc->pt_iter), &(pc->pdiff));

		// Get photometric patch error
		LossComputeErrorImage(&pc->pdiff, &pc->pweight, &pc->pdiff, &tmp);

		// Compute step norm
		pc->delta_p_sqnorm = pc->delta_p.squaredNorm();
		if (pc->cnt == 1)
			pc->delta_p_sqnorm_init = pc->delta_p_sqnorm;

		// Check early termination criterions
		pc->mares_old = pc->mares;
		pc->mares = pc->pweight.lpNorm<1>() / (op->novals);
		if (!((pc->cnt < op->max_iter) &  (pc->mares  > op->res_thresh) &
			((pc->cnt < op->min_iter) | (pc->delta_p_sqnorm / pc->delta_p_sqnorm_init >= op->dp_thresh)) &
			((pc->cnt < op->min_iter) | (pc->mares / pc->mares_old <= op->dr_thresh))))
			pc->hasconverged = 1;

	}

	// Extract patch on integer position, and gradients, No Bilinear interpolation
	void PatClass::getPatchStaticNNGrad(const float* img, const float* img_dx, const float* img_dy,
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

		pos[0] = round((*mid_in)[0]) + cpt->imgpadding;
		pos[1] = round((*mid_in)[1]) + cpt->imgpadding;

		int posxx = 0;

		int lb = -op->p_samp_s / 2;
		int ub = op->p_samp_s / 2 - 1;

		for (int j = lb; j <= ub; ++j)
		{
			for (int i = lb; i <= ub; ++i, ++posxx)
			{
				pos_it[0] = pos[0] + i;
				pos_it[1] = pos[1] + j;
				int idx = pos_it[0] + pos_it[1] * cpt->tmp_w;

				tmp_in[posxx] = img[idx];
				tmp_dx_in[posxx] = img_dx[idx];
				tmp_dy_in[posxx] = img_dy[idx];
			}
		}

		// PATCH NORMALIZATION
		if (op->patnorm>0) // Subtract Mean
			tmp_in_e->array() -= (tmp_in_e->sum() / op->novals);
	}

	// Extract patch on float position with bilinear interpolation, no gradients.
	void PatClass::getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e)
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

		pos[0] += cpt->imgpadding;
		pos[1] += cpt->imgpadding;

		float * tmp_it = tmp_in;
		const float * img_a, *img_b, *img_c, *img_d, *img_e;

		img_e = img + pos[0] - op->p_samp_s / 2;

		int lb = -op->p_samp_s / 2;
		int ub = op->p_samp_s / 2 - 1;

		for (pos_it[1] = pos[1] + lb; pos_it[1] <= pos[1] + ub; ++pos_it[1])
		{
			img_a = img_e + pos_it[1] * cpt->tmp_w;
			img_c = img_e + (pos_it[1] - 1) * cpt->tmp_w;
			img_b = img_a - 1;
			img_d = img_c - 1;


			for (pos_it[0] = pos[0] + lb; pos_it[0] <= pos[0] + ub; ++pos_it[0],
				++tmp_it, ++img_a, ++img_b, ++img_c, ++img_d)
			{
				(*tmp_it) = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d);
			}
		}
		// PATCH NORMALIZATION
		if (op->patnorm>0) // Subtract Mean
			tmp_in_e->array() -= (tmp_in_e->sum() / op->novals);
	}

}


