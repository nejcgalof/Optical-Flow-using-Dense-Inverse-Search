#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include "patch.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{
	Patch::Patch(image_parameters* image_param_in, fix_parameters* fix_param_in) : image_param(image_param_in), fix_param(fix_param_in)
	{
		pc = new patch_state;
		pc->patch_second.resize(fix_param->num_points_patch, 1);

		patch_grad.resize(fix_param->num_points_patch, 1);
		patch_grad_dx.resize(fix_param->num_points_patch, 1);
		patch_grad_dy.resize(fix_param->num_points_patch, 1);
	}

	Patch::~Patch()
	{
		delete pc;
	}

	void Patch::init_patch(Map<MatrixXf>* img_first_in, Map<MatrixXf>* img_first_dx_in, Map<MatrixXf>* img_first_dy_in, Vector2f patch_ref_in)
	{
		img_first = img_first_in;
		img_first_dx = img_first_dx_in;
		img_first_dy = img_first_dy_in;

		patch_first_pos = patch_ref_in;
		reset_patch();

		// Get gradient for this reference patch
		get_gradients_on_patch();

		// Compute hessian matrix on this patch
		compute_hessian_matrix();
	}

	void Patch::get_gradients_on_patch()
	{
		Vector2i pos;
		Vector2i pos_it;

		pos[0] = round(patch_first_pos[0]) + image_param->img_padding;
		pos[1] = round(patch_first_pos[1]) + image_param->img_padding;

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
				
				patch_grad[posxx] = img_first->data()[idx];
				patch_grad_dx[posxx] = img_first_dx->data()[idx];
				patch_grad_dy[posxx] = img_first_dy->data()[idx];
			}
		}
	}

	void Patch::compute_hessian_matrix()
	{
		// The "Hessian matrix" of a multivariable function, organizes all second partial derivatives into a matrix
		// Is square 2x2 matrix
		// [ x^2 x*y ]
		// [ x*y y^2 ]
		// H = sum (S*S) for each pixel; S is partial derivate of image gradient
		pc->hessian(0, 0) = (patch_grad_dx.array() * patch_grad_dx.array()).sum();
		pc->hessian(0, 1) = (patch_grad_dx.array() * patch_grad_dy.array()).sum();
		pc->hessian(1, 1) = (patch_grad_dy.array() * patch_grad_dy.array()).sum();
		pc->hessian(1, 0) = pc->hessian(0, 1);
		if (pc->hessian.determinant() == 0)
		{
			pc->hessian(0, 0) += 1e-10;
			pc->hessian(1, 1) += 1e-10;
		}
	}

	void Patch::set_target_image(Map<MatrixXf> * img_second_in, Map<MatrixXf> * img_second_dx_in, Map<MatrixXf> * img_second_dy_in)
	{
		img_second = img_second_in;
		img_second_dx = img_second_dx_in;
		img_second_dy = img_second_dy_in;

		reset_patch();
	}

	void Patch::reset_patch()
	{
		pc->converged = false;
		pc->optimal_started = false;

		pc->patch_second_start_pos = patch_first_pos;
		pc->patch_second_pos = patch_first_pos;

		pc->patch_input_pos.setZero();
		pc->u.setZero();
		pc->delta_u.setZero();

		pc->counter_iter = 0;
		pc->invalid = false;
	}

	// Test if first move is valid. Test if inverse search optimal started
	void Patch::inverse_search_start(Vector2f patch_in)
	{
		pc->patch_input_pos = patch_in;
		pc->u = patch_in;

		// Warp with u W(x,u) = (x+u,y+v)
		pc->patch_second_pos = patch_first_pos + pc->u;

		// save starting location, only needed for outlier check
		pc->patch_second_start_pos = pc->patch_second_pos;

		//Check if initial position is already invalid
		if (pc->patch_second_pos[0] < image_param->tmp_lb || pc->patch_second_pos[1] < image_param->tmp_lb
			|| pc->patch_second_pos[0] > image_param->tmp_ub_w || pc->patch_second_pos[1] > image_param->tmp_ub_h)
		{
			// if not
			pc->converged = true;
			pc->patch_second = patch_grad;
			pc->optimal_started = true;
		}
		else
		{
			pc->counter_iter = 0; // reset iteration counter
			pc->converged = false;

			get_patch_second_image();
			
			// If max iteration, patch converged - we stop
			if ((pc->counter_iter > fix_param->iterations)) {
				pc->converged = true;
			}

			pc->optimal_started = true;
			pc->invalid = false;
		}
	}

	void Patch::inverse_search(Vector2f patch_in)
	{
		if (!pc->optimal_started)
		{
			reset_patch();
			inverse_search_start(patch_in);
		}

		// Optimize patch until convergence
		while (!pc->converged)
		{
			pc->counter_iter++;

			// W(x,u+delta_u)
			// Compute the sum, which includes the image difference (second image) multiplication with the image gradients
			pc->delta_u[0] = (patch_grad_dx.array() * pc->patch_second.array()).sum();
			pc->delta_u[1] = (patch_grad_dy.array() * pc->patch_second.array()).sum();

			// Solve linear system Ax=b for delta u
			// A is LU decomposition of Hessian matrix (eigen need this to solve - also can use LLT etc ...)
			pc->delta_u = pc->hessian.lu().solve(pc->delta_u);

			// Update warp parameter u: In optical flow this become u<-u-delta_u
			pc->u = pc->u - pc->delta_u; // update flow vector
		   
			// Warp with warping vector u(u,v): W(x,u) = (x+u,y+v)
			pc->patch_second_pos = patch_first_pos + pc->u;

			// check if patch(es) moved too far from starting location, if yes, stop iteration and reset to starting location
			if ((pc->patch_second_start_pos - pc->patch_second_pos).norm() > fix_param->outlierthresh  // check if query patch moved more than >padval from starting location -> most likely outlier
				||
				pc->patch_second_pos[0] < image_param->tmp_lb || pc->patch_second_pos[1] < image_param->tmp_lb ||
				pc->patch_second_pos[0] > image_param->tmp_ub_w || pc->patch_second_pos[1] > image_param->tmp_ub_h)
			{
				pc->u = pc->patch_input_pos; // reset
				pc->patch_second_pos = patch_first_pos + pc->u;
				pc->converged = true;
				pc->optimal_started = true;
			}

			get_patch_second_image();
			
			// If max iteration, patch converged - we stop
			if ((pc->counter_iter > fix_param->iterations)) {
				pc->converged = true;
			}
		}
	}

	// Calculate image patch with linear interpolation on second image
	// Patch is not always on real position, this is reason for linear interpolation
	void Patch::get_patch_second_image()
	{
		Vector4f weight;
		Vector2i pos;

		// linear interpolation: calculate weights
		// l=floor(x)
		// k=floor(y)
		// a=x-l
		// b=y-k
		// (1-a)(1-b)*I(l,k)  <-weight[0] = (1-a)*(1-b)
		// + a*(1-b)*I(l+1,k) <-weight[1] = a*(1-b)
		// + b*(1-a)*I(l,k+1) <-weight[2] = b*(1-a)
		// + a*b*I(l+1,k+1) <-weight[3] = a*b

		float l = floor((pc->patch_second_pos)[0]);
		float k = floor((pc->patch_second_pos)[1]);

		float a = (pc->patch_second_pos)[0] - l;
		float b = (pc->patch_second_pos)[1] - k;
		weight[0] = (1 - a)*(1 - b);
		weight[1] = a * (1 - b);
		weight[2] = b * (1 - a);
		weight[3] = a * b;

		// Find nearest real position (x,y)
		pos[0] = ceil((pc->patch_second_pos)[0] + .00001f) + image_param->img_padding;
		pos[1] = ceil((pc->patch_second_pos)[1] + .00001f) + image_param->img_padding;

		// lb and up for patch - for moving
		int lb = -fix_param->patch_size / 2;
		int ub = fix_param->patch_size / 2 - 1;

		int data_it = 0;
		Vector2i pos_it;

		// start in left side of patch
		int ind_e = pos[0] - fix_param->patch_size / 2;

		//each row calculate
		for (pos_it[1] = pos[1] + lb; pos_it[1] <= pos[1] + ub; ++pos_it[1])
		{
			// get 4 neighbours of looking point a=right-up, b=left-up, c=right-down, d=left-down
			int ind_a = ind_e  + pos_it[1] * image_param->tmp_w;
			int ind_c = ind_e + (pos_it[1] - 1) * image_param->tmp_w;
			int ind_b = ind_a - 1;
			int ind_d = ind_c - 1;

			// in one line move to right - each column
			for (pos_it[0] = pos[0] + lb; pos_it[0] <= pos[0] + ub; ++pos_it[0])
			{
				pc->patch_second.data()[data_it] = weight[3] * img_second->data()[ind_a] + weight[2] * img_second->data()[ind_b] + weight[1] * img_second->data()[ind_c] + weight[0] * img_second->data()[ind_d];
				data_it++, ind_a++, ind_b++, ind_c++, ind_d++;
			}
		}

		// patch normalization
		if (fix_param->patch_normalization) { // Subtract Mean
			pc->patch_second.array() -= (pc->patch_second.sum() / fix_param->num_points_patch);
		}
	}
}


