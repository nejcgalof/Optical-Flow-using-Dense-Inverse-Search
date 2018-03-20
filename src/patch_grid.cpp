#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "patch.hpp"
#include "patch_grid.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace OpticalFlow
{

	PatchGrid::PatchGrid(image_parameters* image_param_in, fix_parameters* fix_param_in) : image_param(image_param_in), fix_param(fix_param_in)
	{
		// Generate grid on current scale
		num_patch_width = ceil((float)image_param->width / (float)fix_param->steps);
		num_patch_height = ceil((float)image_param->height / (float)fix_param->steps);
		int offsetw = floor((image_param->width - (num_patch_width - 1)*fix_param->steps) / 2);
		int offseth = floor((image_param->height - (num_patch_height - 1)*fix_param->steps) / 2);

		num_all_patch = num_patch_width*num_patch_height;
		patch_reference.resize(num_all_patch);
		patch_init.resize(num_all_patch);
		patches.reserve(num_all_patch);

		img_first_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);
		img_first_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);
		img_first_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);

		img_second_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);
		img_second_dx_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);
		img_second_dy_eg = new Map<MatrixXf>(nullptr, image_param->height, image_param->width);

		int patch_id = 0;
		for (int x = 0; x < num_patch_width; ++x)
		{
			for (int y = 0; y < num_patch_height; ++y)
			{
				patch_reference[patch_id][0] = x * fix_param->steps + offsetw;
				patch_reference[patch_id][1] = y * fix_param->steps + offseth;
				patch_init[patch_id].setZero();

				patches.push_back(new Patch(image_param, fix_param));
				patch_id++;
			}
		}
	}

	PatchGrid::~PatchGrid()
	{
		delete img_first_eg;
		delete img_first_dx_eg;
		delete img_first_dy_eg;

		delete img_second_eg;
		delete img_second_dx_eg;
		delete img_second_dy_eg;

		for (int i = 0; i < num_all_patch; ++i) {
			delete patches[i];
		}
	}

	void PatchGrid::init_grid(float* img_first_in, float* img_first_dx_in, float* img_first_dy_in)
	{
		img_first = img_first_in;
		img_first_dx = img_first_dx_in;
		img_first_dy = img_first_dy_in;
		img_first_eg = new Map<MatrixXf>(img_first, image_param->height, image_param->width);
		img_first_dx_eg = new Map<MatrixXf>(img_first_dx, image_param->height, image_param->width);
		img_first_dy_eg = new Map<MatrixXf>(img_first_dy, image_param->height, image_param->width);

		for (int i = 0; i < num_all_patch; ++i)
		{
			patches[i]->init_patch(img_first_eg, img_first_dx_eg, img_first_dy_eg, patch_reference[i]);
			patch_init[i].setZero();
		}
	}

	void PatchGrid::set_target_image(float* img_second_in, float* img_second_dx_in, float* img_second_dy_in)
	{
		img_second = img_second_in;
		img_second_dx = img_second_dx_in;
		img_second_dy = img_second_dy_in;

		img_second_eg = new Map<MatrixXf>(img_second, image_param->height, image_param->width);
		img_second_dx_eg = new Map<MatrixXf>(img_second_dx, image_param->height, image_param->width);
		img_second_dy_eg = new Map<MatrixXf>(img_second_dy, image_param->height, image_param->width);

		for (int i = 0; i < num_all_patch; ++i) {
			patches[i]->set_target_image(img_second_eg, img_second_dx_eg, img_second_dy_eg);
		}
	}

	void PatchGrid::inverse_search()
	{
		// Inverse search for each patch
		for (int i = 0; i < num_all_patch; ++i)
		{
			patches[i]->inverse_search(patch_init[i]); // optimize until convergence  
		}
	}

	void PatchGrid::patch_init_from_prev_flow(float * flow_prev)
	{
		for (int ip = 0; ip < num_all_patch; ++ip)
		{
			int x = floor(patch_reference[ip][0] / 2);
			int y = floor(patch_reference[ip][1] / 2);
			int i = y * (image_param->width / 2) + x;

			patch_init[ip](0) = flow_prev[2 * i] * 2;
			patch_init[ip](1) = flow_prev[2 * i + 1] * 2;
		}
	}

	void PatchGrid::densification_and_create_dance_flow(float *dense_flow)
	{
		float* weight = new float[image_param->width * image_param->height];

		memset(dense_flow, 0, sizeof(float) * (2 * image_param->width * image_param->height)); // Fill with 0
		memset(dense_flow, 0, sizeof(float) * (image_param->width * image_param->height)); // Fill with 0

		float intensity_difference = 0.5f;  // denotes the intensity difference between template patch and warped image at this pixel

		// update displacements vectors u
		// create a dense flow in each pixel by applying weighted averaging to all patches overlapping at x in the reference image
		for (int ip = 0; ip < num_all_patch; ++ip)
		{
			if (patches[ip]->IsValid())
			{
				Vector2f* u = patches[ip]->GetParam(); // flow displacement of this patch
				Vector2f new_u; // new flow

				int lb = -fix_param->patch_size / 2;
				int ub = fix_param->patch_size / 2 - 1;

				// Inside a patch
				for (int y = lb; y <= ub; ++y)
				{
					for (int x = lb; x <= ub; ++x)
					{
						int xt = (x + patch_reference[ip][0]);
						int yt = (y + patch_reference[ip][1]);

						// if inside image
						if (xt >= 0 && yt >= 0 && xt < image_param->width && yt < image_param->height)
						{
							int i = yt*image_param->width + xt;

							// Us(x) = sum( (1/2)*u) // 1/2 for pixel averaging weight
							new_u = (*u) * intensity_difference;
							weight[i] += intensity_difference; // for pixel averaging weight

							dense_flow[2 * i] += new_u[0];
							dense_flow[2 * i + 1] += new_u[1];
						}
					}
				}
			}
		}
		
		// normalize each pixel by dividing displacement by aggregated weights from all patches
		for (int yi = 0; yi < image_param->height; ++yi)
		{
			for (int xi = 0; xi < image_param->width; ++xi)
			{
				int i = yi*image_param->width + xi;
				if (weight[i]>0)
				{    
					dense_flow[2 * i] /= weight[i];
					dense_flow[2 * i + 1] /= weight[i];
				}
			}
		}

		delete[] weight;
	}
}