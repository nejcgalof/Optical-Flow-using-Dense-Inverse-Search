#pragma once
#include "optical_flow.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{
	typedef struct
	{
		bool converged;
		bool optimal_started;

		// reference/second patch 
		Matrix<float, Dynamic, 1> patch_second; // image error to reference/second image

		Matrix<float, 2, 2> hessian; // Hessian for optimization
		Vector2f patch_input_pos;  // point position
		Vector2f u; // warping vector u
		Vector2f delta_u; // iteration update delta_u
		Vector2f patch_second_pos; // result of W(x,u) patch position 
		Vector2f patch_second_start_pos; // start positions

		int counter_iter = 0;
		bool invalid = false;
	} patch_state;

	class Patch
	{

	public:
		Patch(image_parameters* image_param_in, fix_parameters* fix_param_in);
		~Patch();

		void init_patch(Map<MatrixXf>* img_first_in, Map<MatrixXf>* img_first_dx_in, Map<MatrixXf>* img_first_dy_in, Vector2f pt_ref_in);
		void set_target_image(Map<MatrixXf>* img_second_in, Map<MatrixXf>* img_second_dx_in, Map<MatrixXf>* img_second_dy_in);

		void inverse_search(Vector2f patch_in);

		Vector2f get_patch_pos()  { return pc->patch_second_pos; }  // get current iteration patch position
		bool is_valid() { return (!pc->invalid); }
		Vector2f* get_u_vector() { return &(pc->u); } 

	private:

		void inverse_search_start(Vector2f patch_in);
		void reset_patch();
		void compute_hessian_matrix();
		
		// Extract gradient on this patch
		void get_gradients_on_patch();

		// Calculate image patch with linear interpolation on second image 
		void get_patch_second_image();

		Vector2f patch_first_pos; // reference point location (center of patch) paper: pixel x
		Matrix<float, Dynamic, 1> patch_grad;
		Matrix<float, Dynamic, 1> patch_grad_dx; 
		Matrix<float, Dynamic, 1> patch_grad_dy;

		Map<MatrixXf> * img_first, *img_first_dx, *img_first_dy;
		Map<MatrixXf> * img_second, *img_second_dx, *img_second_dy;

		image_parameters* image_param;
		fix_parameters* fix_param;

		patch_state * pc = nullptr; // current patch state
	};
}
