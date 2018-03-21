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
		float** img_first_in, float** img_first_dx_in, float** img_first_dy_in,
		float ** img_second_in, float ** img_second_dx_in, float ** img_second_dy_in,
		int img_padding_in,
		float * out_flow,
		int width_in, int height_in,
		int coarsest_scale, int finest_scale,
		int iterations,
		int patch_size,
		float patch_overlap,
		bool patch_normalization,
		bool draw_grid)
		: img_first(img_first_in), img_first_dx(img_first_dx_in), img_first_dy(img_first_dy_in), img_second(img_second_in), img_second_dx(img_second_dx_in), img_second_dy(img_second_dy_in)
	{
		fix_param.patch_size = patch_size;  // center pixel (p_samp_s/2, p_samp_s/2)
		fix_param.outlierthresh = (float)fix_param.patch_size / 2; // Threshold (px) before a patch is outlier
		fix_param.coarsest_scale = coarsest_scale;
		fix_param.finest_scale = finest_scale;
		fix_param.iterations = iterations,
		fix_param.steps = max(1, (int)floor(fix_param.patch_size*(1 - patch_overlap))); // horizontal and vertical distance (in px) between patch centers
		fix_param.num_points_patch = (patch_size)*(patch_size); // number of points in patch (=p_samp_s*p_samp_s)
		fix_param.patch_normalization = patch_normalization;

		// Create grids, flows and image parameters on each scale
		vector<PatchGrid*> grids(fix_param.coarsest_scale - fix_param.finest_scale + 1);
		vector<float*> flows(fix_param.coarsest_scale - fix_param.finest_scale + 1);
		image_param.resize(fix_param.coarsest_scale - fix_param.finest_scale + 1);

		for (int scale = fix_param.coarsest_scale; scale >= fix_param.finest_scale; --scale)
		{
			int s = scale - fix_param.finest_scale;

			float scale_factor = pow(2, -scale); // scaling factor at current scale
			image_param[s].height = height_in * scale_factor;
			image_param[s].width = width_in * scale_factor;
			image_param[s].img_padding = img_padding_in;
			image_param[s].tmp_lb = -(float)fix_param.patch_size / 2;
			image_param[s].tmp_ub_w = (float)(image_param[s].width + fix_param.patch_size / 2 - 2);
			image_param[s].tmp_ub_h = (float)(image_param[s].height + fix_param.patch_size / 2 - 2);
			image_param[s].tmp_w = image_param[s].width + 2 * img_padding_in;
			image_param[s].tmp_h = image_param[s].height + 2 * img_padding_in;

			flows[s] = new float[2 * image_param[s].width * image_param[s].height];
			grids[s] = new PatchGrid(&(image_param[s]), &fix_param);
		}


		// Main loop Operate over scales, coarsext to finest
		for (int scale = fix_param.coarsest_scale; scale >= fix_param.finest_scale; --scale)
		{
			cout << "scale: " << scale << endl;
			int s = scale - fix_param.finest_scale;

			// Initialize grid (Step 1 in Algorithm 1 of paper)
			grids[s]->init_grid(img_first[scale], img_first_dx[scale], img_first_dy[scale]);
			grids[s]->set_target_image(img_second[scale], img_second_dx[scale], img_second_dy[scale]);

			// Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)                                          
			if (scale < fix_param.coarsest_scale)
			{
				grids[s]->patch_init_from_prev_flow(flows[s + 1]); // initialize from flow at previous coarser scale
			}

			// Dense Inverse Search. (Step 3 in Algorithm 1 of paper)                                          
			grids[s]->inverse_search(); // Inverse search for each patch

			// Densification. (Step 4 in Algorithm 1 of paper)                                                                    
			float *tmp_ptr = flows[s];
			if (scale == fix_param.finest_scale) {
				tmp_ptr = out_flow;
			}
			grids[s]->densification_and_create_dance_flow(tmp_ptr);

			// Display Grid on current scale
			if(draw_grid)
			{
				float sc_fct_tmp = pow(2, scale); // upscale factor

				Mat src(image_param[s].height + 2 * image_param[s].img_padding, image_param[s].width + 2 * image_param[s].img_padding, CV_32FC1, (void*)img_first[s]);
				Mat img_ao_mat = src(Rect(image_param[s].img_padding, image_param[s].img_padding, image_param[s].width, image_param[s].height));

				Mat outimg;
				img_ao_mat.convertTo(outimg, CV_8UC1);
				cvtColor(outimg, outimg, CV_GRAY2RGB);
				resize(outimg, outimg, Size(), sc_fct_tmp, sc_fct_tmp, INTER_NEAREST);

				// draw borders 
				for (int i = 0; i < grids[s]->get_num_all_patch(); ++i) {
					draw_patch_borders(outimg, grids[s]->get_patch_ref_pos(i), sc_fct_tmp);
				}

				for (int i = 0; i < grids[s]->get_num_all_patch(); ++i)
				{
					// Show displacement vector
					Vector2f pt_ref = grids[s]->get_patch_ref_pos(i);
					Vector2f pt_query = grids[s]->get_patch_query_pos(i);

					Vector2f pta, ptb;
					line(outimg, Point((pt_ref[0] + .5)*sc_fct_tmp, (pt_ref[1] + .5)*sc_fct_tmp), Point((pt_query[0] + .5)*sc_fct_tmp, (pt_query[1] + .5)*sc_fct_tmp), Scalar(0, 255, 0), 2);
				}

				namedWindow("grid", WINDOW_AUTOSIZE);
				imshow("grid", outimg);
				waitKey(30);
			}
		} // end of main loop

		// clean up
		for (int sl = fix_param.coarsest_scale; sl >= fix_param.finest_scale; --sl)
		{
			delete[] flows[sl - fix_param.finest_scale];
			delete grids[sl - fix_param.finest_scale];
		}
	}

	void OpticalFlowClass::draw_patch_borders(Mat img, Vector2f pt, float sc)
	{
		//line(img, Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), Point((pt[0] + .5)*sc, (pt[1] + .5)*sc), Scalar(255, 0, 0), 4);
		//circle(img, Point((pt[0])*sc, (pt[1])*sc), (int)(sc*((double)op.steps/4.0)), Scalar(255, 0, 0), CV_FILLED);
		float lb = -fix_param.patch_size / 2;
		float ub = fix_param.patch_size / 2 - 1;

		line(img, Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), Scalar(0, 0, 255), 1);
		line(img, Point(((pt[0] + ub) + .5)*sc, ((pt[1] + lb) + .5)*sc), Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), Scalar(0, 0, 255), 1);
		line(img, Point(((pt[0] + ub) + .5)*sc, ((pt[1] + ub) + .5)*sc), Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), Scalar(0, 0, 255), 1);
		line(img, Point(((pt[0] + lb) + .5)*sc, ((pt[1] + ub) + .5)*sc), Point(((pt[0] + lb) + .5)*sc, ((pt[1] + lb) + .5)*sc), Scalar(0, 0, 255), 1);
	}
}













