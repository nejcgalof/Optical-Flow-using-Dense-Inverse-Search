#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

namespace OpticalFlow
{
	typedef struct
	{
		int width; // image width, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int height; // image height, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int imgpadding; // image padding in pixels at all sides, images padded with replicated border, gradients padded with zero
		float tmp_lb; // lower bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ubw; // upper width bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ubh; // upper height bound for valid image region, pre-compute for image padding to avoid border check 
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
			float** im_ao_in, float** im_ao_dx_in, float** im_ao_dy_in,
			float ** im_bo_in, float ** im_bo_dx_in, float ** im_bo_dy_in,
			int imgpadding_in,
			float * outflow, // Output-flow
			float * initflow, // Initialization-flow
			int width_in, int height_in,
			int coarsest_scale, int finest_scale,
			int iterations,
			int patch_size,
			float patch_overlap,
			bool patnorm_in);

	private:
		void DisplayDrawPatchBoundary(cv::Mat img, Eigen::Vector2f pt, float sc);

		float ** im_ao, ** im_ao_dx, ** im_ao_dy;
		float ** im_bo, ** im_bo_dx, ** im_bo_dy;

		fix_parameters fix_param;
		vector<image_parameters> image_param;
	};
}


