#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{
	typedef struct
	{
		int width; // image width, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int height; // image height, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int img_padding; // image padding in pixels at all sides, images padded with replicated border, gradients padded with zero
		float tmp_lb; // lower bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ub_w; // upper width bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ub_h; // upper height bound for valid image region, pre-compute for image padding to avoid border check 
		int tmp_w; // width + 2*imgpadding
		int tmp_h; // height + 2*imgpadding
	} image_parameters;

	typedef struct
	{
		// Explicitly set parameters:
		int coarsest_scale;             // first (coarsest) scale
		int finest_scale;             // last (finest) scale
		int patch_size;         // patch size (edge length in pixels)  
		int iterations;         // max. iterations on one scale
		bool patch_normalization;          // Use patch mean-normalization
		float outlierthresh;          // displacement threshold (in px) before a patch is flagged as outlier
		int steps;                    // horizontal and vertical distance (in px) between patch centers
		int num_points_patch;         // number of points in patch (=p_samp_s*p_samp_s)
		float minerrval = 2.0f;       // 1/max(this, error) for pixel averaging weight
	} fix_parameters;

	class OpticalFlowClass
	{

	public:
		OpticalFlowClass(
			float** img_first_in, float** img_first_dx_in, float** img_first_dy_in,
			float** img_second_in, float** img_second_dx_in, float** img_second_dy_in,
			int img_padding_in,
			float * outflow, // Output-flow
			int width_in, int height_in,
			int coarsest_scale, int finest_scale,
			int iterations,
			int patch_size,
			float patch_overlap,
			bool patnorm_in);

	private:
		void DisplayDrawPatchBoundary(Mat img, Vector2f pt, float sc);

		float** img_first, ** img_first_dx, ** img_first_dy;
		float** img_second, ** img_second_dx, ** img_second_dy;

		fix_parameters fix_param;
		vector<image_parameters> image_param;
	};
}


