#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>
#include <color_coding.hpp>
#include <IO_flow.hpp>

using namespace cv;

// Construct images and gradient pyramides
void construct_pyramide(cv::Mat & img_fmat, cv::Mat * img_fmat_pyr, cv::Mat * img_dx_fmat_pyr, cv::Mat * img_dy_fmat_pyr, float ** img_pyr, float ** img_dx_pyr, float ** img_dy_pyr, int lv_f, int img_padding)
{
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		if (i == 0) // At finest scale: copy directly
		{
			cv::Mat dx, dy, dx2, dy2, dmag;
			cv::Sobel(img_fmat, dx, CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			cv::Sobel(img_fmat, dy, CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			dx2 = dx.mul(dx);
			dy2 = dy.mul(dy);
			dmag = dx2 + dy2;
			cv::sqrt(dmag, dmag);
			// Set magnitude in finnest scale
			img_fmat_pyr[i] = dmag.clone();
		}
		else { // for other: downscale previous scale by .5  - just downsize 
			cv::resize(img_fmat_pyr[i - 1], img_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
		}
		img_fmat_pyr[i].convertTo(img_fmat_pyr[i], CV_32FC1);

		// Calculate gradient dx and dy
		cv::Sobel(img_fmat_pyr[i], img_dx_fmat_pyr[i], CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
		cv::Sobel(img_fmat_pyr[i], img_dy_fmat_pyr[i], CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
		img_dx_fmat_pyr[i].convertTo(img_dx_fmat_pyr[i], CV_32F);
		img_dy_fmat_pyr[i].convertTo(img_dy_fmat_pyr[i], CV_32F);
	}

	// pad images
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		copyMakeBorder(img_fmat_pyr[i], img_fmat_pyr[i], img_padding, img_padding, img_padding, img_padding, cv::BORDER_REPLICATE);  // Replicate border for image padding
		img_pyr[i] = (float*)img_fmat_pyr[i].data;
		copyMakeBorder(img_dx_fmat_pyr[i], img_dx_fmat_pyr[i], img_padding, img_padding, img_padding, img_padding, cv::BORDER_CONSTANT, 0); // Zero padding for gradients
		img_dx_pyr[i] = (float*)img_dx_fmat_pyr[i].data;
		copyMakeBorder(img_dy_fmat_pyr[i], img_dy_fmat_pyr[i], img_padding, img_padding, img_padding, img_padding, cv::BORDER_CONSTANT, 0);
		img_dy_pyr[i] = (float*)img_dy_fmat_pyr[i].data;
	}
}

void CreateFolder(const char* path)
{
	if (!CreateDirectory(path, NULL))
	{
		return;
	}
}

int main(int argc, char** argv )
{
	for (int img_i = 1; img_i < 50; img_i++) {

		// PARAMETERS INPUT
		string folder = "alley_1/";
		string write_folder = "OF_" + folder;
		CreateFolder(write_folder.c_str());
		string first = std::to_string(img_i);
		string second = std::to_string(img_i+1);
		string first_image = folder+"frame_" + std::string(4 - first.length(), '0').append(first) + ".png";
		string second_image = folder+"frame_" + std::string(4 - second.length(), '0').append(second) + ".png";
		cout << "start " << first_image << endl;
		const char *imgfile_first = first_image.c_str();
		const char *imgfile_second = second_image.c_str();

		int iterations = 1000; // Max. iterations
		int patch_size = 8; // Rectangular patch size in (pixel)
		const int coarsest_scale = 3; // Coarsest scale in multi-scale pyramid
		int finest_scale = 0; // Finest scale in multi-scale pyramide
		float patch_overlap = 0.7; // Patch overlap on each scale (percent) - 0.7
		bool patch_normalization = true; // Mean - normalize patches
		bool draw_grid = false; // draw patch grid and flows

		cv::Mat img_first_mat, img_second_mat, img_tmp;
		img_first_mat = cv::imread(imgfile_first, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
		img_second_mat = cv::imread(imgfile_second, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file   
		if (!img_first_mat.data || !img_second_mat.data)
		{
			printf("No image data \n");
			return -1;
		}

		/*cv::Mat flowinit;
		flowinit.create(img_first_mat.size().height, img_first_mat.size().width, CV_32FC2);
		ReadFlowFile(flowinit, "frame_0001.flo");
		Mat dst;
		draw_optical_flow(flowinit, dst);
		SaveFlowFile(flowinit, "frame_0002.flo");
		imshow("flowinit", dst);
		waitKey(30);*/
		// PARAMETERS
		cv::Size sz = img_first_mat.size();
		int width_org = sz.width;   // unpadded original image size
		int height_org = sz.height;

		// Add border for divisible for all scales
		int padw = 0, padh = 0;
		int scale_factor = pow(2, coarsest_scale); // Division by this number on coarsest scale
		int div = sz.width % scale_factor;
		if (div > 0) { // If not divide , set padding
			padw = scale_factor - div;
		}
		div = sz.height % scale_factor;
		if (div > 0) { // If not divide , set padding
			padh = scale_factor - div;
		}
		if (padh > 0 || padw > 0)
		{
			copyMakeBorder(img_first_mat, img_first_mat, floor((float)padh / 2.0f), ceil((float)padh / 2.0f), floor((float)padw / 2.0f), ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
			copyMakeBorder(img_second_mat, img_second_mat, floor((float)padh / 2.0f), ceil((float)padh / 2.0f), floor((float)padw / 2.0f), ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
		}
		sz = img_first_mat.size(); // set new size

		// Generate scale pyramids
		cv::Mat img_first_fmat, img_second_fmat;
		img_first_mat.convertTo(img_first_fmat, CV_32F); // convert to float
		img_second_mat.convertTo(img_second_fmat, CV_32F);

		float* img_first_pyr[coarsest_scale + 1];
		float* img_first_dx_pyr[coarsest_scale + 1];
		float* img_first_dy_pyr[coarsest_scale + 1];
		float* img_second_pyr[coarsest_scale + 1];
		float* img_second_dx_pyr[coarsest_scale + 1];
		float* img_second_dy_pyr[coarsest_scale + 1];

		cv::Mat img_first_fmat_pyr[coarsest_scale + 1];
		cv::Mat img_second_fmat_pyr[coarsest_scale + 1];
		cv::Mat img_first_dx_fmat_pyr[coarsest_scale + 1];
		cv::Mat img_first_dy_fmat_pyr[coarsest_scale + 1];
		cv::Mat img_second_dx_fmat_pyr[coarsest_scale + 1];
		cv::Mat img_second_dy_fmat_pyr[coarsest_scale + 1];

		// Construct images and gradient pyramides
		construct_pyramide(img_first_fmat, img_first_fmat_pyr, img_first_dx_fmat_pyr, img_first_dy_fmat_pyr, img_first_pyr, img_first_dx_pyr, img_first_dy_pyr, coarsest_scale, patch_size);
		construct_pyramide(img_second_fmat, img_second_fmat_pyr, img_second_dx_fmat_pyr, img_second_dy_fmat_pyr, img_second_pyr, img_second_dx_pyr, img_second_dy_pyr, coarsest_scale, patch_size);

		// Run optical flow algorithm
		float sc_fct = pow(2, finest_scale);
		cv::Mat flowout(sz.height / sc_fct, sz.width / sc_fct, CV_32FC2); // Optical Flow


		OpticalFlow::OpticalFlowClass ofc(img_first_pyr, img_first_dx_pyr, img_first_dy_pyr,
			img_second_pyr, img_second_dx_pyr, img_second_dy_pyr,
			patch_size,  // extra image padding to avoid border violation check
			(float*)flowout.data,   // pointer to n-band output float array
			sz.width, sz.height,
			coarsest_scale, finest_scale, iterations, patch_size, patch_overlap, patch_normalization, draw_grid);

		// Resize to original scale, if not run to finest level
		if (finest_scale != 0)
		{
			flowout *= sc_fct;
			cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct, cv::INTER_LINEAR);
		}
		// If image was padded, remove padding before saving to file
		flowout = flowout(cv::Rect((int)floor((float)padw / 2.0f), (int)floor((float)padh / 2.0f), width_org, height_org));

		Mat dst2;
		draw_optical_flow(flowout, dst2);
		imwrite("OF_"+(string)imgfile_first, dst2);
		imshow("output_color_OF", dst2);
		waitKey(30);
		cout << "finish " << first_image << endl;
	}

    return 0;
}