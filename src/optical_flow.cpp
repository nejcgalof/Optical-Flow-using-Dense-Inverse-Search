#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "optical_flow.hpp"
#include "patch_grid.hpp"

using namespace std;
using namespace cv;

namespace OFC
{

	OFClass::OFClass(
		float** im_ao_in, float** im_ao_dx_in, float** im_ao_dy_in, 
		float ** im_bo_in, float ** im_bo_dx_in, float ** im_bo_dy_in,
		int imgpadding_in,
		float * outflow,
		float * initflow,
		int width_in, int height_in,
		int sc_f_in, int sc_l_in,
		int iterations,
		int p_samp_s_in,
		float patove_in,
		bool patnorm_in)
		: im_ao(im_ao_in), im_ao_dx(im_ao_dx_in), im_ao_dy(im_ao_dy_in),
		im_bo(im_bo_in), im_bo_dx(im_bo_dx_in), im_bo_dy(im_bo_dy_in)
	{
		op.p_samp_s = p_samp_s_in;  // center pixel is at (p_samp_s/2, p_samp_s/2)
		op.outlierthresh = (float)op.p_samp_s/2;
		op.sc_f = sc_f_in;
		op.sc_l = sc_l_in;
		op.iterations = iterations,
		op.steps = 2;
		op.novals = (p_samp_s_in)*(p_samp_s_in);
		op.patnorm = patnorm_in;

		// Create grids on each scale
		vector<OFC::PatGridClass*> grid_fw(op.sc_f - op.sc_l + 1);
		vector<float*> flow_fw(op.sc_f - op.sc_l + 1);
		cpl.resize(op.sc_f - op.sc_l + 1);
		for (int sl = op.sc_f; sl >= op.sc_l; --sl)
		{
			int i = sl - op.sc_l;

			float sc_fct = pow(2, -sl); // scaling factor at current scale
			cpl[i].height = height_in * sc_fct;
			cpl[i].width = width_in * sc_fct;
			cpl[i].imgpadding = imgpadding_in;
			cpl[i].tmp_lb = -(float)op.p_samp_s / 2;
			cpl[i].tmp_ubw = (float)(cpl[i].width + op.p_samp_s / 2 - 2);
			cpl[i].tmp_ubh = (float)(cpl[i].height + op.p_samp_s / 2 - 2);
			cpl[i].tmp_w = cpl[i].width + 2 * imgpadding_in;
			cpl[i].tmp_h = cpl[i].height + 2 * imgpadding_in;

			flow_fw[i] = new float[2 * cpl[i].width * cpl[i].height];
			grid_fw[i] = new OFC::PatGridClass(&(cpl[i]), &op);
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

			// Display Grid on current scale
			float sc_fct_tmp = pow(2, sl); // upscale factor

			cv::Mat src(cpl[ii].height + 2 * cpl[ii].imgpadding, cpl[ii].width + 2 * cpl[ii].imgpadding, CV_32FC1, (void*)im_ao[sl]);
			cv::Mat img_ao_mat = src(cv::Rect(cpl[ii].imgpadding, cpl[ii].imgpadding, cpl[ii].width, cpl[ii].height));

			cv::Mat outimg;
			img_ao_mat.convertTo(outimg, CV_8UC1);
			cv::cvtColor(outimg, outimg, CV_GRAY2RGB);
			cv::resize(outimg, outimg, cv::Size(), sc_fct_tmp, sc_fct_tmp, cv::INTER_NEAREST);
			for (int i = 0; i < grid_fw[ii]->GetNoPatches(); ++i)
				DisplayDrawPatchBoundary(outimg, grid_fw[ii]->GetRefPatchPos(i), sc_fct_tmp);

			for (int i = 0; i < grid_fw[ii]->GetNoPatches(); ++i)
			{
				// Show displacement vector
				const Eigen::Vector2f pt_ref = grid_fw[ii]->GetRefPatchPos(i);
				const Eigen::Vector2f pt_ret = grid_fw[ii]->GetQuePatchPos(i);

				Eigen::Vector2f pta, ptb;
				cv::line(outimg, cv::Point((pt_ref[0] + .5)*sc_fct_tmp, (pt_ref[1] + .5)*sc_fct_tmp), cv::Point((pt_ret[0] + .5)*sc_fct_tmp, (pt_ret[1] + .5)*sc_fct_tmp), cv::Scalar(0, 255, 0), 2);
			}
			cv::namedWindow("Img_ao", cv::WINDOW_AUTOSIZE);
			cv::imshow("Img_ao", outimg);

			cv::waitKey(30);
			std::cout << "naprej" << std::endl;
		}


		// Clean up
		for (int sl = op.sc_f; sl >= op.sc_l; --sl)
		{
			delete[] flow_fw[sl - op.sc_l];
			delete grid_fw[sl - op.sc_l];
		}
	}

	void OFClass::DisplayDrawPatchBoundary(cv::Mat img, Eigen::Vector2f pt, float sc)
	{
		cv::line(img, cv::Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), cv::Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), cv::Scalar(0, 0, 255), 4);

		float lb = -op.p_samp_s / 2;
		float ub = op.p_samp_s / 2 - 1;

		cv::line(img, cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Scalar(0, 0, 255), 1);
	}
}














