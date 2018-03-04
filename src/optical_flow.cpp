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
using namespace Eigen;

namespace OpticalFlow
{

	OpticalFlowClass::OpticalFlowClass(
		float** im_ao_in, float** im_ao_dx_in, float** im_ao_dy_in, 
		float ** im_bo_in, float ** im_bo_dx_in, float ** im_bo_dy_in,
		int imgpadding_in,
		float * outflow,
		float * initflow,
		int width_in, int height_in,
		int coarsest_scale, int finest_scale,
		int iterations,
		int patch_size,
		float patch_overlap,
		bool patch_normalization)
		: im_ao(im_ao_in), im_ao_dx(im_ao_dx_in), im_ao_dy(im_ao_dy_in),
		im_bo(im_bo_in), im_bo_dx(im_bo_dx_in), im_bo_dy(im_bo_dy_in)
	{
		fix_param.patch_size = patch_size;  // center pixel (p_samp_s/2, p_samp_s/2)
		fix_param.outlierthresh = (float)fix_param.patch_size / 2; // Threshold (px) before a patch is outlier
		fix_param.coarsest_scale = coarsest_scale;
		fix_param.finest_scale = finest_scale;
		fix_param.iterations = iterations,
		fix_param.steps = std::max(1, (int)floor(fix_param.patch_size*(1 - patch_overlap))); // horizontal and vertical distance (in px) between patch centers
		fix_param.num_points_patch = (patch_size)*(patch_size); // number of points in patch (=p_samp_s*p_samp_s)
		fix_param.patch_normalization = patch_normalization;

		// Create grids on each scale
		vector<OpticalFlow::PatchGrid*> grid_fw(fix_param.coarsest_scale - fix_param.finest_scale + 1);
		vector<float*> flow_fw(fix_param.coarsest_scale - fix_param.finest_scale + 1);
		image_param.resize(fix_param.coarsest_scale - fix_param.finest_scale + 1);
		for (int sl = fix_param.coarsest_scale; sl >= fix_param.finest_scale; --sl)
		{
			int i = sl - fix_param.finest_scale;

			float sc_fct = pow(2, -sl); // scaling factor at current scale
			image_param[i].height = height_in * sc_fct;
			image_param[i].width = width_in * sc_fct;
			image_param[i].imgpadding = imgpadding_in;
			image_param[i].tmp_lb = -(float)fix_param.patch_size / 2;
			image_param[i].tmp_ubw = (float)(image_param[i].width + fix_param.patch_size / 2 - 2);
			image_param[i].tmp_ubh = (float)(image_param[i].height + fix_param.patch_size / 2 - 2);
			image_param[i].tmp_w = image_param[i].width + 2 * imgpadding_in;
			image_param[i].tmp_h = image_param[i].height + 2 * imgpadding_in;

			flow_fw[i] = new float[2 * image_param[i].width * image_param[i].height];
			grid_fw[i] = new OpticalFlow::PatchGrid(&(image_param[i]), &fix_param);
		}


		// *** Main loop; Operate over scales, coarse-to-fine
		for (int sl = fix_param.coarsest_scale; sl >= fix_param.finest_scale; --sl)
		{
			int ii = sl - fix_param.finest_scale;

			// Initialize grid (Step 1 in Algorithm 1 of paper)
			grid_fw[ii]->InitializeGrid(im_ao[sl], im_ao_dx[sl], im_ao_dy[sl]);
			grid_fw[ii]->SetTargetImage(im_bo[sl], im_bo_dx[sl], im_bo_dy[sl]);

			// Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)                                          
			if (sl < fix_param.coarsest_scale)
			{
				grid_fw[ii]->InitializeFromCoarserOF(flow_fw[ii + 1]); // initialize from flow at previous coarser scale
			}
			else if (sl == fix_param.coarsest_scale && initflow != nullptr) // initialization given input flow
			{
				grid_fw[ii]->InitializeFromCoarserOF(initflow); // initialize from flow at coarser scale
			}


			// Dense Inverse Search. (Step 3 in Algorithm 1 of paper)                                          
			grid_fw[ii]->Optimize();

			// Densification. (Step 4 in Algorithm 1 of paper)                                                                    
			float *tmp_ptr = flow_fw[ii];
			if (sl == fix_param.finest_scale)
				tmp_ptr = outflow;

			grid_fw[ii]->AggregateFlowDense(tmp_ptr);

			// Display Grid on current scale
			float sc_fct_tmp = pow(2, sl); // upscale factor

			Mat src(image_param[ii].height + 2 * image_param[ii].imgpadding, image_param[ii].width + 2 * image_param[ii].imgpadding, CV_32FC1, (void*)im_ao[sl]);
			cv::Mat img_ao_mat = src(cv::Rect(image_param[ii].imgpadding, image_param[ii].imgpadding, image_param[ii].width, image_param[ii].height));

			cv::Mat outimg;
			img_ao_mat.convertTo(outimg, CV_8UC1);
			cv::cvtColor(outimg, outimg, CV_GRAY2RGB);
			cv::resize(outimg, outimg, cv::Size(), sc_fct_tmp, sc_fct_tmp, cv::INTER_NEAREST);
			for (int i = 0; i < grid_fw[ii]->GetNoPatches(); ++i)
				DisplayDrawPatchBoundary(outimg, grid_fw[ii]->GetRefPatchPos(i), sc_fct_tmp);

			for (int i = 0; i < grid_fw[ii]->GetNoPatches(); ++i)
			{
				// Show displacement vector
				const Vector2f pt_ref = grid_fw[ii]->GetRefPatchPos(i);
				const Eigen::Vector2f pt_ret = grid_fw[ii]->GetQuePatchPos(i);

				Eigen::Vector2f pta, ptb;
				//cv::line(outimg, cv::Point((pt_ref[0] + .5)*sc_fct_tmp, (pt_ref[1] + .5)*sc_fct_tmp), cv::Point((pt_ret[0] + .5)*sc_fct_tmp, (pt_ret[1] + .5)*sc_fct_tmp), cv::Scalar(0, 255, 0), 2);
			}
			cv::namedWindow("Img_ao", cv::WINDOW_AUTOSIZE);
			cv::imshow("Img_ao", outimg);

			cv::waitKey(30);
			//std::cin.get();
			std::cout << "naprej" << std::endl;
		}


		// Clean up
		for (int sl = fix_param.coarsest_scale; sl >= fix_param.finest_scale; --sl)
		{
			delete[] flow_fw[sl - fix_param.finest_scale];
			delete grid_fw[sl - fix_param.finest_scale];
		}
	}

	void OpticalFlowClass::DisplayDrawPatchBoundary(cv::Mat img, Eigen::Vector2f pt, float sc)
	{
		//cv::line(img, cv::Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), cv::Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), cv::Scalar(255, 0, 0), 4);
		//cv::circle(img, cv::Point((pt[0])*sc, (pt[1])*sc), (int)(sc*((double)op.steps/4.0)), cv::Scalar(255, 0, 0), CV_FILLED);
		float lb = -fix_param.patch_size / 2;
		float ub = fix_param.patch_size / 2 - 1;

		cv::line(img, cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		cv::line(img, cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), cv::Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), cv::Scalar(0, 0, 255), 1);
		
	}
}














