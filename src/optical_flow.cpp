#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "optical_flow.hpp"
#include "patch_grid.hpp"


using namespace std;

namespace OFC
{

	OFClass::OFClass(float** im_ao_in, float** im_ao_dx_in, float** im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
																									  // E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
																									  // im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
																									  // IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0, 
		float ** im_bo_in, float ** im_bo_dx_in, float ** im_bo_dy_in,
		const int imgpadding_in,
		float * outflow,
		const float * initflow,
		const int width_in, const int height_in,
		const int sc_f_in, const int sc_l_in,
		const int max_iter_in, const int min_iter_in,
		const float  dp_thresh_in,
		const float  dr_thresh_in,
		const float res_thresh_in,
		const int p_samp_s_in,
		const float patove_in,
		const int costfct_in,
		const int noc_in,
		const int patnorm_in)
		: im_ao(im_ao_in), im_ao_dx(im_ao_dx_in), im_ao_dy(im_ao_dy_in),
		im_bo(im_bo_in), im_bo_dx(im_bo_dx_in), im_bo_dy(im_bo_dy_in)
	{
		op.nop = 2;
		op.p_samp_s = p_samp_s_in;  // patch has even border length, center pixel is at (p_samp_s/2, p_samp_s/2) (ZERO INDEXED!) 
		op.outlierthresh = (float)op.p_samp_s / 2;
		op.patove = patove_in;
		op.sc_f = sc_f_in;
		op.sc_l = sc_l_in;
		op.max_iter = max_iter_in;
		op.min_iter = min_iter_in;
		op.dp_thresh = dp_thresh_in*dp_thresh_in; // saves the square to compare with squared L2-norm (saves sqrt operation)
		op.dr_thresh = dr_thresh_in;
		op.res_thresh = res_thresh_in;
		op.steps = 2;// std::max(1, (int)floor(op.p_samp_s*(1 - op.patove)));
		op.novals = noc_in * (p_samp_s_in)*(p_samp_s_in);
		op.costfct = costfct_in;
		op.noc = noc_in;
		op.patnorm = patnorm_in;
		op.noscales = op.sc_f - op.sc_l + 1;
		op.normoutlier_tmpbsq = new float[4]{ op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier };
		//op.normoutlier_tmp2bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq, op.twos);
		op.normoutlier_tmp2bsq = new float[4]{ op.normoutlier_tmpbsq[0] * op.twos[0],op.normoutlier_tmpbsq[1] * op.twos[1],op.normoutlier_tmpbsq[2] * op.twos[2], op.normoutlier_tmpbsq[3] * op.twos[3] };
		//op.normoutlier_tmp4bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq, op.fours);
		op.normoutlier_tmp4bsq = new float[4]{ op.normoutlier_tmpbsq[0] * op.fours[0],op.normoutlier_tmpbsq[1] * op.fours[1],op.normoutlier_tmpbsq[2] * op.fours[2], op.normoutlier_tmpbsq[3] * op.fours[3] };

		// Create grids on each scale
		vector<OFC::PatGridClass*> grid_fw(op.noscales);
		vector<OFC::PatGridClass*> grid_bw(op.noscales); // grid for backward OF computation, only needed if 'usefbcon' is set to 1.
		vector<float*> flow_fw(op.noscales);
		vector<float*> flow_bw(op.noscales);
		cpl.resize(op.noscales);
		cpr.resize(op.noscales);
		for (int sl = op.sc_f; sl >= op.sc_l; --sl)
		{
			int i = sl - op.sc_l;

			float sc_fct = pow(2, -sl); // scaling factor at current scale
			cpl[i].sc_fct = sc_fct;
			cpl[i].height = height_in * sc_fct;
			cpl[i].width = width_in * sc_fct;
			cpl[i].imgpadding = imgpadding_in;
			cpl[i].tmp_lb = -(float)op.p_samp_s / 2;
			cpl[i].tmp_ubw = (float)(cpl[i].width + op.p_samp_s / 2 - 2);
			cpl[i].tmp_ubh = (float)(cpl[i].height + op.p_samp_s / 2 - 2);
			cpl[i].tmp_w = cpl[i].width + 2 * imgpadding_in;
			cpl[i].tmp_h = cpl[i].height + 2 * imgpadding_in;
			cpl[i].curr_lv = sl;
			cpl[i].camlr = 0;


			cpr[i] = cpl[i];
			cpr[i].camlr = 1;

			flow_fw[i] = new float[op.nop * cpl[i].width * cpl[i].height];
			grid_fw[i] = new OFC::PatGridClass(&(cpl[i]), &(cpr[i]), &op);
		}
		// *** Main loop; Operate over scales, coarse-to-fine
		for (int sl = op.sc_f; sl >= op.sc_l; --sl)
		{
			int ii = sl - op.sc_l;

			// Initialize grid (Step 1 in Algorithm 1 of paper)
			grid_fw[ii]->InitializeGrid(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]);
			grid_fw[ii]->SetTargetImage(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]);

			// Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)                                          
			if (sl < op.sc_f)
			{
				grid_fw[ii]->InitializeFromCoarserOF(flow_fw[ii + 1]); // initialize from flow at previous coarser scale
			}
			else if (sl == op.sc_f && initflow != nullptr) // initialization given input flow
			{
				grid_fw[ii]->InitializeFromCoarserOF(initflow); // initialize from flow at coarser scale
			}


			// Dense Inverse Search. (Step 3 in Algorithm 1 of paper)                                          
			grid_fw[ii]->Optimize();

			// Densification. (Step 4 in Algorithm 1 of paper)                                                                    
			float *tmp_ptr = flow_fw[ii];
			if (sl == op.sc_l)
				tmp_ptr = outflow;

			grid_fw[ii]->AggregateFlowDense(tmp_ptr);

		}

		// Clean up
		for (int sl = op.sc_f; sl >= op.sc_l; --sl)
		{
			delete[] flow_fw[sl - op.sc_l];
			delete grid_fw[sl - op.sc_l];
		}
	}
}














