#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "patch.hpp"
#include "patch_grid.hpp"

using namespace std;

namespace OpticalFlow
{

	PatchGrid::PatchGrid(
		image_parameters* image_param_in,
		fix_parameters* fix_param_in)
		:
		image_param(image_param_in),
		fix_param(fix_param_in)
	{

		// Generate grid on current scale
		steps = fix_param->steps;
		nopw = ceil((float)image_param->width / (float)steps);
		noph = ceil((float)image_param->height / (float)steps);
		int offsetw = floor((image_param->width - (nopw - 1)*steps) / 2);
		int offseth = floor((image_param->height - (noph - 1)*steps) / 2);

		nopatches = nopw*noph;
		pt_ref.resize(nopatches);
		p_init.resize(nopatches);
		pat.reserve(nopatches);

		im_ao_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);
		im_ao_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);
		im_ao_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);

		im_bo_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);
		im_bo_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);
		im_bo_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr, image_param->height, image_param->width);

		int patchid = 0;
		for (int x = 0; x < nopw; ++x)
		{
			for (int y = 0; y < noph; ++y)
			{
				int i = x*noph + y;

				pt_ref[i][0] = x * steps + offsetw;
				pt_ref[i][1] = y * steps + offseth;
				p_init[i].setZero();

				pat.push_back(new OpticalFlow::Patch(image_param, fix_param, patchid));
				patchid++;
			}
		}
	}

	PatchGrid::~PatchGrid()
	{
		delete im_ao_eg;
		delete im_ao_dx_eg;
		delete im_ao_dy_eg;

		delete im_bo_eg;
		delete im_bo_dx_eg;
		delete im_bo_dy_eg;

		for (int i = 0; i< nopatches; ++i)
			delete pat[i];
	}

	void PatchGrid::SetComplGrid(PatchGrid *cg_in)
	{
		cg = cg_in;
	}


	void PatchGrid::InitializeGrid(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in)
	{
		im_ao = im_ao_in;
		im_ao_dx = im_ao_dx_in;
		im_ao_dy = im_ao_dy_in;

		new (im_ao_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao, image_param->height, image_param->width); // new placement operator
		new (im_ao_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dx, image_param->height, image_param->width);
		new (im_ao_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dy, image_param->height, image_param->width);

		for (int i = 0; i < nopatches; ++i)
		{
			pat[i]->InitializePatch(im_ao_eg, im_ao_dx_eg, im_ao_dy_eg, pt_ref[i]);
			p_init[i].setZero();
		}

	}

	void PatchGrid::SetTargetImage(const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in)
	{
		im_bo = im_bo_in;
		im_bo_dx = im_bo_dx_in;
		im_bo_dy = im_bo_dy_in;

		new (im_bo_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo, image_param->height, image_param->width); // new placement operator
		new (im_bo_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dx, image_param->height, image_param->width); // new placement operator
		new (im_bo_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dy, image_param->height, image_param->width); // new placement operator

		for (int i = 0; i < nopatches; ++i)
			pat[i]->SetTargetImage(im_bo_eg, im_bo_dx_eg, im_bo_dy_eg);

	}

	void PatchGrid::Optimize()
	{
		for (int i = 0; i < nopatches; ++i)
		{
			pat[i]->OptimizeIter(p_init[i], true); // optimize until convergence  
		}
	}

	void PatchGrid::InitializeFromCoarserOF(const float * flow_prev)
	{
		for (int ip = 0; ip < nopatches; ++ip)
		{
			int x = floor(pt_ref[ip][0] / 2); // better, but slower: use bil. interpolation here
			int y = floor(pt_ref[ip][1] / 2);
			int i = y*(image_param->width / 2) + x;

			p_init[ip](0) = flow_prev[2 * i] * 2;
			p_init[ip](1) = flow_prev[2 * i + 1] * 2;
		}
	}

	void PatchGrid::AggregateFlowDense(float *flowout) const
	{
		float* we = new float[image_param->width * image_param->height];

		memset(flowout, 0, sizeof(float) * (2 * image_param->width * image_param->height));
		memset(we, 0, sizeof(float) * (image_param->width * image_param->height));

		for (int ip = 0; ip < nopatches; ++ip)
		{

			if (pat[ip]->IsValid())
			{
				const Eigen::Vector2f* fl = pat[ip]->GetParam(); // flow displacement of this patch
				Eigen::Vector2f flnew;

				const float * pweight = pat[ip]->GetpWeightPtr(); // use image error as weight

				int lb = -fix_param->patch_size / 2;
				int ub = fix_param->patch_size / 2 - 1;

				for (int y = lb; y <= ub; ++y)
				{
					for (int x = lb; x <= ub; ++x, ++pweight)
					{
						int yt = (y + pt_ref[ip][1]);
						int xt = (x + pt_ref[ip][0]);

						if (xt >= 0 && yt >= 0 && xt < image_param->width && yt < image_param->height)
						{

							int i = yt*image_param->width + xt;

							float absw = 1.0f / (float)(std::max(fix_param->minerrval, *pweight));

							flnew = (*fl) * absw;
							we[i] += absw;

							flowout[2 * i] += flnew[0];
							flowout[2 * i + 1] += flnew[1];
						}
					}
				}
			}
		}

		// if complementary (forward-backward merging) is given, integrate negative backward flow as well
		if (cg)
		{
			Eigen::Vector4f wbil; // bilinear weight vector
			Eigen::Vector4i pos;

			for (int ip = 0; ip < cg->nopatches; ++ip)
			{
				if (cg->pat[ip]->IsValid())
				{
					const Eigen::Vector2f* fl = (cg->pat[ip]->GetParam()); // flow displacement of this patch
					Eigen::Vector2f flnew;

					const Eigen::Vector2f rppos = cg->pat[ip]->GetPointPos(); // get patch position after optimization
					const float * pweight = cg->pat[ip]->GetpWeightPtr(); // use image error as weight

					Eigen::Vector2f resid;

					// compute bilinear weight vector
					pos[0] = ceil(rppos[0] + .00001); // make sure they are rounded up to natural number
					pos[1] = ceil(rppos[1] + .00001); // make sure they are rounded up to natural number
					pos[2] = floor(rppos[0]);
					pos[3] = floor(rppos[1]);

					resid[0] = rppos[0] - pos[2];
					resid[1] = rppos[1] - pos[3];
					wbil[0] = resid[0] * resid[1];
					wbil[1] = (1 - resid[0])*resid[1];
					wbil[2] = resid[0] * (1 - resid[1]);
					wbil[3] = (1 - resid[0])*(1 - resid[1]);

					int lb = -fix_param->patch_size / 2;
					int ub = fix_param->patch_size / 2 - 1;


					for (int y = lb; y <= ub; ++y)
					{
						for (int x = lb; x <= ub; ++x, ++pweight)
						{

							int yt = y + pos[1];
							int xt = x + pos[0];
							if (xt >= 1 && yt >= 1 && xt < (image_param->width - 1) && yt < (image_param->height - 1))
							{
								float absw = 1.0f / (float)(std::max(fix_param->minerrval, *pweight));
								flnew = (*fl) * absw;

								int idxcc = xt + yt   *image_param->width;
								int idxfc = (xt - 1) + yt   *image_param->width;
								int idxcf = xt + (yt - 1)*image_param->width;
								int idxff = (xt - 1) + (yt - 1)*image_param->width;

								we[idxcc] += wbil[0] * absw;
								we[idxfc] += wbil[1] * absw;
								we[idxcf] += wbil[2] * absw;
								we[idxff] += wbil[3] * absw;

								flowout[2 * idxcc] -= wbil[0] * flnew[0];   // use reversed flow 
								flowout[2 * idxcc + 1] -= wbil[0] * flnew[1];

								flowout[2 * idxfc] -= wbil[1] * flnew[0];
								flowout[2 * idxfc + 1] -= wbil[1] * flnew[1];

								flowout[2 * idxcf] -= wbil[2] * flnew[0];
								flowout[2 * idxcf + 1] -= wbil[2] * flnew[1];

								flowout[2 * idxff] -= wbil[3] * flnew[0];
								flowout[2 * idxff + 1] -= wbil[3] * flnew[1];
							}
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
				if (we[i]>0)
				{    
					flowout[2 * i] /= we[i];
					flowout[2 * i + 1] /= we[i];
				}
			}
		}

		delete[] we;
	}

}