#pragma once
#include "optical_flow.hpp" // For camera intrinsic and opt. parameter struct

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{
	typedef struct
	{
		bool hasconverged;
		bool hasoptstarted;

		// reference/template patch 
		Eigen::Matrix<float, Eigen::Dynamic, 1> patch_diff; // image error to reference image
		Eigen::Matrix<float, Eigen::Dynamic, 1> patch_weight; // absolute error image

		Eigen::Matrix<float, 2, 2> Hes; // Hessian for optimization
		Eigen::Vector2f p_in, p_iter, delta_p; // point position, displacement to starting position, iteration update
		Eigen::Vector2f pt_iter;
		Eigen::Vector2f pt_st;

		int cnt = 0;
		bool invalid = false;
	} patch_state;



	class Patch
	{

	public:
		Patch(image_parameters* image_param_in, fix_parameters* fix_param_in);
		~Patch();

		void InitializePatch(Map<MatrixXf> * im_ao_in, Map<MatrixXf> * im_ao_dx_in, Map<MatrixXf> * im_ao_dy_in, Vector2f pt_ref_in);
		void SetTargetImage(Map<MatrixXf> * im_bo_in, Map<MatrixXf> * im_bo_dx_in, Map<MatrixXf> * im_bo_dy_in);

		void OptimizeIter(const Eigen::Vector2f p_in_arg, const bool untilconv);

		Eigen::Vector2f GetPointPos()  { return pc->pt_iter; }  // get current iteration patch position (in this frame's opposite camera for OF, Depth)
		bool IsValid() { return (!pc->invalid); }
		float * GetpWeightPtr() { return (float*)pc->patch_weight.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class
		Eigen::Vector2f* GetParam() { return &(pc->p_iter); }   // get current iteration parameters

	private:

		void OptimizeStart(const Eigen::Vector2f p_in_arg);
		void ResetPatch();
		void ComputeHessian();
		
		// Extract patch on integer position, and gradients, No Bilinear interpolation
		void getPatchStaticNNGrad(const float* img, const float* img_dx, const float* img_dy, const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in, Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmp_dx_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_dy_in);
		// Extract patch on float position with bilinear interpolation, no gradients.  
		void getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e);

		Eigen::Vector2f pt_ref; // reference point location
		Eigen::Matrix<float, Eigen::Dynamic, 1> tmp;
		Eigen::Matrix<float, Eigen::Dynamic, 1> dxx_tmp; // x derivative, doubles as steepest descent image for OF, Depth, SF
		Eigen::Matrix<float, Eigen::Dynamic, 1> dyy_tmp; // y derivative, doubles as steepest descent image for OF, SF

		Map<MatrixXf> * im_ao, *im_ao_dx, *im_ao_dy;
		Map<MatrixXf> * im_bo, *im_bo_dx, *im_bo_dy;

		image_parameters* image_param;
		fix_parameters* fix_param;

		patch_state * pc = nullptr; // current patch state
	};
}
