#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <optical_flow.hpp>
#include <color_coding.hpp>
#include <IO_flow.hpp>

using namespace cv;


void ConstructImgPyramide(cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr,float** img_ao_pyr, float** img_ao_dx_pyr, float** img_ao_dy_pyr, int lv_f, int lv_l, bool getgrad, int imgpadding, int padw, int padh)
{
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		if (i == 0) // At finest scale: copy directly, for all other: downscale previous scale by .5
		{
			img_ao_fmat_pyr[i] = img_ao_fmat.clone();
			cv::Mat dx, dy, dx2, dy2, dmag;
			cv::Sobel(img_ao_fmat, dx, CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			cv::Sobel(img_ao_fmat, dy, CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			dx2 = dx.mul(dx);
			dy2 = dy.mul(dy);
			dmag = dx2 + dy2;
			cv::sqrt(dmag, dmag);
			img_ao_fmat_pyr[i] = dmag.clone();
		}
		else
			cv::resize(img_ao_fmat_pyr[i - 1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);

		img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], CV_32FC1);

		if (getgrad)
		{
			cv::Sobel(img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			cv::Sobel(img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
			img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
		}
	}

	// pad images
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		copyMakeBorder(img_ao_fmat_pyr[i], img_ao_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_REPLICATE);  // Replicate border for image padding
		img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

		if (getgrad)
		{
			copyMakeBorder(img_ao_dx_fmat_pyr[i], img_ao_dx_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_CONSTANT, 0); // Zero padding for gradients
			copyMakeBorder(img_ao_dy_fmat_pyr[i], img_ao_dy_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_CONSTANT, 0);

			img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
			img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;
		}
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


	char *imgfile_ao = "frame_0001.png";
	char *imgfile_bo = "frame_0002.png";
	//char *imgfile_ao = "frame10.png";
	//char *imgfile_bo = "frame11.png";

	cv::Mat img_ao_mat, img_bo_mat, img_tmp;
	img_ao_mat = cv::imread(imgfile_ao, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	img_bo_mat = cv::imread(imgfile_bo, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file   
    if ( !img_ao_mat.data  || !img_bo_mat.data)
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
	cv::Size sz = img_ao_mat.size();
	int width_org = sz.width;   // unpadded original image size
	int height_org = sz.height;
	
	// PARAMETERS

	int maxiter = 16; // Max. iterations
	int miniter = 16; // Min. iterations
	float mindprate = 0.05; // Early stopping parameters
	float mindrrate = 0.95; // Early stopping parameters
	float minimgerr = 0.0; // Early stopping parameters
	int patchsz = 8; // Rectangular patch size in (pixel)
	int lv_f = std::max(0, (int)std::floor(log2((2.0f*(float)width_org) / ((float)20 * (float)patchsz)))); // Coarsest scale in multi-scale pyramid
	int lv_l = std::max(lv_f - 2, 0); // Finest scale in multi-scale pyramide
	float poverl = 0.3; // Patch overlap on each scale (percent)
	int patnorm = 1; // Mean - normalize patches

	// Add border for divisible for all scales
	int padw = 0, padh = 0;
	int scale_factor = pow(2, lv_f); // Division by this number on coarsest scale
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
		copyMakeBorder(img_ao_mat, img_ao_mat, floor((float)padh / 2.0f), ceil((float)padh / 2.0f), floor((float)padw / 2.0f), ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
		copyMakeBorder(img_bo_mat, img_bo_mat, floor((float)padh / 2.0f), ceil((float)padh / 2.0f), floor((float)padw / 2.0f), ceil((float)padw / 2.0f), cv::BORDER_REPLICATE);
	}
	sz = img_ao_mat.size(); // set new size

	// Generate scale pyramids
	cv::Mat img_ao_fmat, img_bo_fmat;
	img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
	img_bo_mat.convertTo(img_bo_fmat, CV_32F);

	float* img_ao_pyr = new float[lv_f + 1];
	float* img_bo_pyr = new float[lv_f + 1];
	float* img_ao_dx_pyr = new float[lv_f + 1];
	float* img_ao_dy_pyr = new float[lv_f + 1];
	float* img_bo_dx_pyr = new float[lv_f + 1];
	float* img_bo_dy_pyr = new float[lv_f + 1];

	Mat* img_ao_fmat_pyr = new Mat[lv_f + 1];
	Mat* img_bo_fmat_pyr = new Mat[lv_f + 1];
	Mat* img_ao_dx_fmat_pyr = new Mat[lv_f + 1];
	Mat* img_ao_dy_fmat_pyr = new Mat[lv_f + 1];
	Mat* img_bo_dx_fmat_pyr = new Mat[lv_f + 1];
	Mat* img_bo_dy_fmat_pyr = new Mat[lv_f + 1];

	ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, &img_ao_pyr, &img_ao_dx_pyr, &img_ao_dy_pyr, lv_f, lv_l, true, patchsz, padw, padh);
	ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, &img_bo_pyr, &img_bo_dx_pyr, &img_bo_dy_pyr, lv_f, lv_l, true, patchsz, padw, padh);

	// Run optical flow algorithm
	float sc_fct = pow(2, lv_l);
	cv::Mat flowout(sz.height / sc_fct, sz.width / sc_fct, CV_32FC2); // Optical Flow


	OFC::OFClass ofc(&img_ao_pyr, &img_ao_dx_pyr, &img_ao_dy_pyr,
		&img_bo_pyr, &img_bo_dx_pyr, &img_bo_dy_pyr,
		patchsz,  // extra image padding to avoid border violation check
		(float*)flowout.data,   // pointer to n-band output float array
		nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
		sz.width, sz.height,
		lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, patnorm);

	// Resize to original scale, if not run to finest level
	if (lv_l != 0)
	{
		flowout *= sc_fct;
		cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct, cv::INTER_LINEAR);
	}
	// If image was padded, remove padding before saving to file
	flowout = flowout(cv::Rect((int)floor((float)padw / 2.0f), (int)floor((float)padh / 2.0f), width_org, height_org));

	Mat dst2;
	draw_optical_flow(flowout, dst2);
	imshow("MOJE", dst2);
    waitKey(0);

    return 0;
}