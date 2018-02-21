#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
using namespace std;
using namespace cv;

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
		int costfct;          // Cost function: 0: L2-Norm, 1: L1-Norm, 2: PseudoHuber-Norm 

							  // Automatically set parameters / fixed parameters
		float patove;                 // point/line padding to all sides (px)
		float outlierthresh;          // displacement threshold (in px) before a patch is flagged as outlier
		int steps;                    // horizontal and vertical distance (in px) between patch centers
		int novals;                   // number of points in patch (=p_samp_s*p_samp_s) 
		int noscales;                 // total number of scales
		float minerrval = 2.0f;       // 1/max(this, error) for pixel averaging weight
		float normoutlier = 5.0f;     // norm error threshold for huber norm

									  // Helper variables
		float* zero = new float[4]{ 0.0f, 0.0f, 0.0f, 0.0f };
		float* negzero = new float[4]{ -0.0f, -0.0f, -0.0f, -0.0f };
		float* half = new float[4]{ 0.5f, 0.5f, 0.5f, 0.5f };
		float* ones = new float[4]{1.0f, 1.0f, 1.0f, 1.0f};
		float* twos = new float[4]{ 2.0f, 2.0f, 2.0f, 2.0f };
		float* fours = new float[4]{ 4.0f, 4.0f, 4.0f, 4.0f };
		float* normoutlier_tmpbsq;
		float* normoutlier_tmp2bsq;
		float* normoutlier_tmp4bsq;
	} optparam;



	class OFClass
	{

	public:
		OFClass(float** im_ao_in, float** im_ao_dx_in, float** im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients. 
																								 // E.g. im_ao[sc_f_in] will be used as coarsest coarsest, im_ao[sc_l_in] as finest scale
																								 // im_ao[  (sc_l_in-1) : 0 ] can be left as nullptr pointers
																								 // IMPORTANT assumption: mod(width,2^sc_f_in)==0  AND mod(height,2^sc_f_in)==0, 
			float ** im_bo_in, float ** im_bo_dx_in, float ** im_bo_dy_in,
			const int imgpadding_in,
			float * outflow,          // Output-flow:         has to be of size to fit the last  computed OF scale [width / 2^(last scale)   , height / 2^(last scale)]   , 1 channel depth / 2 for OF
			const float * initflow,   // Initialization-flow: has to be of size to fit the first computed OF scale [width / 2^(first scale+1), height / 2^(first scale+1)], 1 channel depth / 2 for OF
			const int width_in, const int height_in,
			const int sc_f_in, const int sc_l_in,
			const int max_iter_in, const int min_iter_in,
			const float  dp_thresh_in,
			const float  dr_thresh_in,
			const float res_thresh_in,
			const int padval_in,
			const float patove_in,
			const int patnorm_in);

	private:

		// needed for verbosity >= 3, DISVISUAL
		void DisplayDrawPatchBoundary(cv::Mat img, Eigen::Vector2f pt, float sc);

		float ** im_ao, ** im_ao_dx, ** im_ao_dy;
		float ** im_bo, ** im_bo_dx, ** im_bo_dy;

		optparam op;                    // Struct for pptimization parameters
		vector<camparam> cpl;
		vector<camparam> cpr; // Struct (for each scale) for camera/image parameter
	};

}


