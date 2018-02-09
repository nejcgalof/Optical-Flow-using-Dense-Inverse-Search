#include <iostream>
#include <string>
#include <vector>
#include <optical_flow.hpp>


using namespace std;
using namespace cv;

namespace OFC
{
	OFClass::OFClass(vector<float*> & im_ao_in, vector<float*> & im_ao_dx_in, vector<float*> & im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
																											// E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
																											// im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
																											// IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0, 
		vector<float*> &  im_bo_in, vector<float*> &  im_bo_dx_in, vector<float*> &  im_bo_dy_in,
		int imgpadding_in,
		float * outflow,          // Output-flow:         has to be of size to fit the last  computed OF scale [width / 2^(last scale)   , height / 2^(last scale)]   , 1 channel depth / 2 for OF
		float * initflow,   // Initialization-flow: has to be of size to fit the first computed OF scale [width / 2^(first scale+1), height / 2^(first scale+1)], 1 channel depth / 2 for OF
		int width_in, int height_in,
		int sc_f_in, int sc_l_in,
		int max_iter_in, int min_iter_in,
		float dp_thresh_in,
		float dr_thresh_in,
		float res_thresh_in,
		int p_samp_s_in,
		float patove_in,
		bool usefbcon_in,
		int costfct_in,
		int noc_in,
		int patnorm_in,
		bool usetvref_in,
		float tv_alpha_in,
		float tv_gamma_in,
		float tv_delta_in,
		int tv_innerit_in,
		int tv_solverit_in,
		float tv_sor_in)
		: im_ao(im_ao_in), im_ao_dx(im_ao_dx_in), im_ao_dy(im_ao_dy_in),
		im_bo(im_bo_in), im_bo_dx(im_bo_dx_in), im_bo_dy(im_bo_dy_in) 
	{
		// CODE
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
		op.steps = std::max(1, (int)floor(op.p_samp_s*(1 - op.patove)));
		op.novals = noc_in * (p_samp_s_in)*(p_samp_s_in);
		op.usefbcon = usefbcon_in;
		op.costfct = costfct_in;
		op.noc = noc_in;
		op.patnorm = patnorm_in;
		op.noscales = op.sc_f - op.sc_l + 1;
		op.usetvref = usetvref_in;
		op.tv_alpha = tv_alpha_in;
		op.tv_gamma = tv_gamma_in;
		op.tv_delta = tv_delta_in;
		op.tv_innerit = tv_innerit_in;
		op.tv_solverit = tv_solverit_in;
		op.tv_sor = tv_sor_in;
		op.normoutlier_tmpbsq =  { op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier, op.normoutlier*op.normoutlier };
		op.normoutlier_tmp2bsq = (op.normoutlier_tmpbsq, op.twos);
		op.normoutlier_tmp4bsq = (op.normoutlier_tmpbsq, op.fours);
		
		//vector<OFC::PatGridClass*> grid_fw(op.noscales);
		vector<float*> flow_fw(op.noscales);
		cpl.resize(op.noscales);
		cpr.resize(op.noscales);
	}
}