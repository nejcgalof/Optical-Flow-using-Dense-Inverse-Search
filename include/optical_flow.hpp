#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace OFC
{
	typedef struct
	{
		int width;                // image width, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int height;               // image height, does not include '2*imgpadding', but includes original padding to ensure integer divisible image width and height
		int imgpadding;           // image padding in pixels at all sides, images padded with replicated border, gradients padded with zero, ADD THIS ONLY WHEN ADDRESSING THE IMAGE OR GRADIENT
		float tmp_lb;             // lower bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ubw;            // upper width bound for valid image region, pre-compute for image padding to avoid border check 
		float tmp_ubh;            // upper height bound for valid image region, pre-compute for image padding to avoid border check 
		int tmp_w;                // width + 2*imgpadding
		int tmp_h;                // height + 2*imgpadding
		float sc_fct;             // scaling factor at current scale  
		int curr_lv;              // current level
		int camlr;                // 0: left camera, 1: right camera, used only for depth, to restrict sideways patch motion
	} camparam;

	typedef struct
	{
		// Explicitly set parameters:
		int sc_f;             // first (coarsest) scale
		int sc_l;             // last (finest) scale
		int p_samp_s;         // patch size (edge length in pixels)  
		int max_iter;         // max. iterations on one scale
		int min_iter;         // min. iterations on one scale
		float dp_thresh;      // minimum rate of change of delta_p before descending one level, e.g. .1 :  change scales when norm(delta_p_last)/norm(delta_p_init) < .1
		float dr_thresh;      // minimum rate of change of residual within 3-iterations-window before descending one level, e.g. .8 :  res_new/res_old >  * .8, SET HIGH (1e10) TO DISABLE
		float res_thresh;     // if (mean absolute) residual falls below this threshold, terminate iterations on current scale, IGNORES MIN_ITER , SET TO LOW (1e-10) TO DISABLE
		int patnorm;          // Use patch mean-normalization
		int verbosity;        // Verbosity, 0: plot nothing, 1: final internal timing 2: complete iteration timing, (UNCOMMENTED -> 3: Display flow scales, 4: Display flow scale iterations)
		bool usefbcon;        // use forward-backward flow merging 
		int costfct;          // Cost function: 0: L2-Norm, 1: L1-Norm, 2: PseudoHuber-Norm 
		bool usetvref;        // TV parameters
		float tv_alpha;
		float tv_gamma;
		float tv_delta;
		int tv_innerit;
		int tv_solverit;
		float tv_sor;         // Successive-over-relaxation weight

							  // Automatically set parameters / fixed parameters
		int nop;                      // number of parameters per pixel, 1 for depth, 2 for optical flow, 4 for scene flow
		float patove;                 // point/line padding to all sides (px)
		float outlierthresh;          // displacement threshold (in px) before a patch is flagged as outlier
		int steps;                    // horizontal and vertical distance (in px) between patch centers
		int novals;                   // number of points in patch (=p_samp_s*p_samp_s) 
		int noc;                      // number of channels in image and gradients 
		int noscales;                 // total number of scales
		float minerrval = 2.0f;       // 1/max(this, error) for pixel averaging weight
		float normoutlier = 5.0f;     // norm error threshold for huber norm

									  // Helper variables
		vector<float> zero{ 0.0f, 0.0f, 0.0f, 0.0f };
		vector<float> negzero = { -0.0f, -0.0f, -0.0f, -0.0f };
		vector<float> half = { 0.5f, 0.5f, 0.5f, 0.5f };
		vector<float> ones = { 1.0f, 1.0f, 1.0f, 1.0f };
		vector<float> twos = { 2.0f, 2.0f, 2.0f, 2.0f };
		vector<float> fours = { 4.0f, 4.0f, 4.0f, 4.0f };
		vector<float> normoutlier_tmpbsq;
		vector<float> normoutlier_tmp2bsq;
		vector<float> normoutlier_tmp4bsq;

	} optparam;



	class OFClass
	{
	public:
		OFClass(vector<float*> & im_ao_in, vector<float*> & im_ao_dx_in, vector<float*> & im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
																								 // E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
																								 // im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
																								 // IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0, 
			vector<float*> &  im_bo_in, vector<float*> &  im_bo_dx_in, vector<float*> &  im_bo_dy_in,
			int imgpadding_in,
			float * outflow,          // Output-flow:         has to be of size to fit the last  computed OF scale [width / 2^(last scale)   , height / 2^(last scale)]   , 1 channel depth / 2 for OF
			float * initflow,   // Initialization-flow: has to be of size to fit the first computed OF scale [width / 2^(first scale+1), height / 2^(first scale+1)], 1 channel depth / 2 for OF
			int width_in,  int height_in,
			int sc_f_in, int sc_l_in,
			int max_iter_in, int min_iter_in,
			float dp_thresh_in,
			float dr_thresh_in,
			float res_thresh_in,
			int padval_in,
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
			float tv_sor_in);

	private:
		vector<float*> im_ao,  im_ao_dx,  im_ao_dy;
		vector<float*> im_bo,  im_bo_dx,  im_bo_dy;

		optparam op;                    // Struct for pptimization parameters
		std::vector<camparam> cpl, cpr; // Struct (for each scale) for camera/image parameter
	};
}