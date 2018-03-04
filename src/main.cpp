#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>
#include <color_coding.hpp>
#include <IO_flow.hpp>

using namespace cv;

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

int main(int argc, char** argv )
{

    /*if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    //image = imread( argv[1], 1 );*/


	char *imgfile_first = "frame_0001.png";
	char *imgfile_second = "frame_0002.png";
	//char *imgfile_ao = "frame10.png";
	//char *imgfile_bo = "frame11.png";

	cv::Mat img_first_mat, img_second_mat, img_tmp;
	img_first_mat = cv::imread(imgfile_first, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	img_second_mat = cv::imread(imgfile_second, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file   
    if ( !img_first_mat.data  || !img_second_mat.data)
    {
        printf("No image data \n");
        return -1;
    }
	/*cv::Mat flowinit;
	flowinit.create(img_ao_mat.size().height, img_ao_mat.size().width, CV_32FC2);
	ReadFlowFile(flowinit, "frame_0001.flo");
	Mat dst;
	draw_optical_flow(flowinit, dst);
	SaveFlowFile(flowinit, "frame_0002.flo");

	imshow("flowinit", dst);*/
	// PARAMETERS
	cv::Size sz = img_first_mat.size();
	int width_org = sz.width;   // unpadded original image size
	int height_org = sz.height;
	
	// PARAMETERS

	int iterations = 1000; // Max. iterations
	int patch_size = 8; // Rectangular patch size in (pixel)
	const int coarsest_scale = 3; // Coarsest scale in multi-scale pyramid
	int finest_scale = 0; // Finest scale in multi-scale pyramide
	float patch_overlap = 0.7; // Patch overlap on each scale (percent) - 0.7
	bool patch_normalization = true; // Mean - normalize patches

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
	if (padh>0 || padw>0)
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
		coarsest_scale, finest_scale, iterations, patch_size, patch_overlap, patch_normalization);

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
	imwrite("vc_001.png", dst2);
	imshow("MOJE", dst2);
    waitKey(0);

    return 0;
}