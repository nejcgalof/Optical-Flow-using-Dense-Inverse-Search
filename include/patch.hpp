#pragma once
#include "optical_flow.hpp" // For camera intrinsic and opt. parameter struct

namespace OFC
{
	typedef struct
	{
		bool hasconverged;
		bool hasoptstarted;

		// reference/template patch 
		Eigen::Matrix<float, Eigen::Dynamic, 1> pdiff; // image error to reference image
		Eigen::Matrix<float, Eigen::Dynamic, 1> pweight; // absolute error image


		Eigen::Matrix<float, 2, 2> Hes; // Hessian for optimization
		Eigen::Vector2f p_in, p_iter, delta_p; // point position, displacement to starting position, iteration update
														  // start positions, current point position, patch norm
		Eigen::Matrix<float, 1, 1> normtmp;
		Eigen::Vector2f pt_iter;
		Eigen::Vector2f pt_st;

		int cnt = 0;
		bool invalid = false;
	} patchstate;



	class PatClass
	{

	public:
		PatClass(const camparam* cpt_in,
			const optparam* op_in,
			const int patchid_in);

		~PatClass();

		void InitializePatch(Eigen::Map<const Eigen::MatrixXf> * im_ao_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dy_in, const Eigen::Vector2f pt_ref_in);
		void SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * im_bo_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dy_in);

		void OptimizeIter(const Eigen::Vector2f p_in_arg, const bool untilconv);

		inline const bool isConverged() const { return pc->hasconverged; }
		inline const bool hasOptStarted() const { return pc->hasoptstarted; }
		inline const Eigen::Vector2f GetPointPos() const { return pc->pt_iter; }  // get current iteration patch position (in this frame's opposite camera for OF, Depth)
		inline const bool IsValid() const { return (!pc->invalid); }
		inline const float * GetpWeightPtr() const { return (float*)pc->pweight.data(); } // Return data pointer to image error patch, used in efficient indexing for densification in patchgrid class

		inline const Eigen::Vector2f* GetParam() const { return &(pc->p_iter); }   // get current iteration parameters

		inline const Eigen::Vector2f* GetParamStart() const { return &(pc->p_in); }

	private:

		void OptimizeStart(const Eigen::Vector2f p_in_arg);
		void paramtopt();
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

		Eigen::Map<const Eigen::MatrixXf> * im_ao, *im_ao_dx, *im_ao_dy;
		Eigen::Map<const Eigen::MatrixXf> * im_bo, *im_bo_dx, *im_bo_dy;

		const camparam* cpt;
		const optparam* op;
		const int patchid;

		patchstate * pc = nullptr; // current patch state

	};
}
