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
		Vector2f patch_in;  // point position
		Vector2f p_iter; //displacement to starting position
		Vector2f delta_p; // iteration update
		Vector2f pt_iter; 
		Vector2f pt_st; // start positions

		int cnt = 0;
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

		Eigen::Vector2f GetPointPos()  { return pc->pt_iter; }  // get current iteration patch position (in this frame's opposite camera for OF, Depth)
		bool IsValid() { return (!pc->invalid); }
		float * GetpWeightPtr() { return (float*)pc->patch_weight.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class
		Eigen::Vector2f* GetParam() { return &(pc->p_iter); }   // get current iteration parameters

	private:

		void OptimizeStart(Vector2f patch_in);
		void reset_patch();
		void compute_hessian_matrix();
		
		// Extract patch on integer position, and gradients, No Bilinear interpolation
		void get_gradients_on_patch(float* img, float* img_dx, float* img_dy, Vector2f* mid_in, Matrix<float, Dynamic, 1>* grad_in, Matrix<float, Dynamic, 1>* grad_dx_in, Matrix<float, Dynamic, 1>* grad_dy_in);
		// Extract patch on float position with bilinear interpolation, no gradients.  
		void getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in, Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e);

		Vector2f patch_ref; // reference point location
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
