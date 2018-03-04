#pragma once
#include "patch.hpp"
#include "optical_flow.hpp" // For camera intrinsic and opt. parameter struct

namespace OpticalFlow
{

	class PatchGrid
	{

	public:
		PatchGrid(image_parameters* image_param_in,
			fix_parameters* fix_param_in);

		~PatchGrid();

		void InitializeGrid(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in);
		void SetTargetImage(const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in);
		void InitializeFromCoarserOF(const float * flow_prev);

		void AggregateFlowDense(float *flowout) const;

		// Optimizes grid to convergence of each patch
		void Optimize();
		//Optimize each patch in grid for one iteration, visualize displacement vector, repeat
		//void OptimizeAndVisualize(const float sc_fct_tmp);  // needed for verbosity >= 3, DISVISUAL

		void SetComplGrid(PatchGrid *cg_in);

		inline const int GetNoPatches() const { return nopatches; }
		inline const int GetNoph() const { return noph; }
		inline const int GetNopw() const { return nopw; }

		inline const Eigen::Vector2f GetRefPatchPos(int i) const { return pt_ref[i]; } // Get reference  patch position
		inline const Eigen::Vector2f GetQuePatchPos(int i) const { return pat[i]->GetPointPos(); } // Get target/query patch position
		inline const Eigen::Vector2f GetQuePatchDis(int i) const { return pt_ref[i] - pat[i]->GetPointPos(); } // Get query patch displacement from reference patch

	private:

		const float * im_ao, *im_ao_dx, *im_ao_dy;
		const float * im_bo, *im_bo_dx, *im_bo_dy;

		Eigen::Map<const Eigen::MatrixXf> * im_ao_eg, *im_ao_dx_eg, *im_ao_dy_eg;
		Eigen::Map<const Eigen::MatrixXf> * im_bo_eg, *im_bo_dx_eg, *im_bo_dy_eg;

		image_parameters* image_param;
		fix_parameters* fix_param;

		int steps;
		int nopw;
		int noph;
		int nopatches;

		std::vector<OpticalFlow::Patch*> pat; // Patch Objects
		std::vector<Eigen::Vector2f> pt_ref; // Midpoints for reference patches
		std::vector<Eigen::Vector2f> p_init; // starting parameters for query patches, use only 1 for depth, 2 for OF, all 4 for scene flow  

		const PatchGrid * cg = nullptr;
	};


}