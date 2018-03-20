#pragma once
#include "patch.hpp"
#include "optical_flow.hpp" // For camera intrinsic and opt. parameter struct

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{

	class PatchGrid
	{

	public:
		PatchGrid(image_parameters* image_param_in, fix_parameters* fix_param_in);
		~PatchGrid();

		void init_grid(float* img_first_in, float* img_first_dx_in, float* img_first_dy_in);
		void set_target_image(float* img_second_in, float* img_second_dx_in, float* img_second_dy_in);
		void patch_init_from_prev_flow(float * flow_prev);

		void densification_and_create_dance_flow(float *dense_flow);

		// Optimizes grid to convergence of each patch
		void inverse_search();

		int get_num_all_patch() { return num_all_patch; }
		Vector2f GetRefPatchPos(int i) { return patch_reference[i]; } // Get reference  patch position
		Vector2f GetQuePatchPos(int i) { return patches[i]->GetPointPos(); } // Get target/query patch position
		Vector2f GetQuePatchDis(int i) { return patch_reference[i] - patches[i]->GetPointPos(); } // Get query patch displacement from reference patch

	private:

		float * img_first, *img_first_dx, *img_first_dy;
		float * img_second, *img_second_dx, *img_second_dy;

		Map<MatrixXf> * img_first_eg, *img_first_dx_eg, *img_first_dy_eg;
		Map<MatrixXf> * img_second_eg, *img_second_dx_eg, *img_second_dy_eg;

		image_parameters* image_param;
		fix_parameters* fix_param;

		int num_patch_width;
		int num_patch_height;
		int num_all_patch;

		vector<Patch*> patches; // Patch Objects
		vector<Vector2f> patch_reference; // Midpoints for reference patches
		vector<Vector2f> patch_init; // starting parameters for query patches
	};
}