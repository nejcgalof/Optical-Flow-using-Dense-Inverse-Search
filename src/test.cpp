#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void ConstructImgPyramide(cv::Mat & img_ao_fmat, vector<Mat> & img_ao_fmat_pyr, vector<Mat> & img_ao_dx_fmat_pyr, vector<Mat> & img_ao_dy_fmat_pyr, vector<float*> & img_ao_pyr, vector<float*> & img_ao_dx_pyr,  vector<float*> & img_ao_dy_pyr, int lv_f, int lv_l, bool getgrad, int imgpadding, int padw, int padh)
{
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		if (i == 0) // At finest scale: copy directly, for all other: downscale previous scale by .5
		{
			// use gradient magnitude image as input
			cv::Mat dx, dy, dx2, dy2, dmag;
			cv::Sobel(img_ao_fmat, dx, CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			cv::Sobel(img_ao_fmat, dy, CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
			dx2 = dx.mul(dx);
			dy2 = dy.mul(dy);
			dmag = dx2 + dy2;
			cv::sqrt(dmag, dmag);
			img_ao_fmat_pyr[i] = dmag.clone();
		}
		else {
			cv::resize(img_ao_fmat_pyr[i - 1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
		}

		img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], CV_32FC1);

		// Get gradient
		cv::Sobel(img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
		cv::Sobel(img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 3, 1 / 8.0, 0, cv::BORDER_DEFAULT);
		img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
		img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
	}

	// pad images
	for (int i = 0; i <= lv_f; ++i)  // Construct image and gradient pyramides
	{
		copyMakeBorder(img_ao_fmat_pyr[i], img_ao_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_REPLICATE);  // Replicate border for image padding
		img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

		// for gradients
		copyMakeBorder(img_ao_dx_fmat_pyr[i], img_ao_dx_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_CONSTANT, 0); // Zero padding for gradients
		copyMakeBorder(img_ao_dy_fmat_pyr[i], img_ao_dy_fmat_pyr[i], imgpadding, imgpadding, imgpadding, imgpadding, cv::BORDER_CONSTANT, 0);

		img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
		img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;
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

	cv::Mat img_ao_mat, img_bo_mat, img_tmp;
	img_ao_mat = cv::imread(imgfile_ao, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	img_bo_mat = cv::imread(imgfile_bo, CV_LOAD_IMAGE_GRAYSCALE);   // Read the file   
    if ( !img_ao_mat.data  || !img_bo_mat.data)
    {
        printf("No image data \n");
        return -1;
    }

	// PARAMETERS
	cv::Size sz = img_ao_mat.size();
	int width_org = sz.width;   // unpadded original image size
	int height_org = sz.height;
	
	// PARAMETERS

	int maxiter = 12; // Max. iterations
	int miniter = 12; // Min. iterations
	float mindprate = 0.05; // Early stopping parameters
	float mindrrate = 0.95; // Early stopping parameters
	float minimgerr = 0.0; // Early stopping parameters
	int patchsz = 8; // Rectangular patch size in (pixel)
	int lv_f = std::max(0, (int)std::floor(log2((2.0f*(float)width_org) / (5 * (float)patchsz)))); // Coarsest scale in multi-scale pyramid
	int lv_l = std::max(lv_f - 2, 0); // Finest scale in multi-scale pyramide
	float poverl = 0.4; // Patch overlap on each scale (percent)
	bool usefbcon = false; // Use forward-backward consistency
	int patnorm = 1; // Mean - normalize patches
	int costfct = 0; // Cost function (here: 0/L2)  Alternatives: 1/L1, 2/Huber, 10/NCC
	bool usetvref = true; // Use TV refinement
	float tv_alpha = 10.0; // TV parameter alpha
	float tv_gamma = 10.0; // TV parameter gamma
	float tv_delta = 5.0; // TV parameter delta
	int tv_innerit = 1; // Number of TV outer iterations
	int tv_solverit = 3; // Number of TV solver iterations 
	float tv_sor = 1.6; // TV SOR value 

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
	vector<float*> img_ao_pyr(lv_f + 1);
	vector<float*> img_bo_pyr(lv_f + 1);
	vector<float*> img_ao_dx_pyr(lv_f + 1);
	vector<float*> img_ao_dy_pyr(lv_f + 1);
	vector<float*> img_bo_dx_pyr(lv_f + 1);
	vector<float*> img_bo_dy_pyr(lv_f + 1);

	vector<Mat> img_ao_fmat_pyr(lv_f + 1);
	vector<Mat> img_bo_fmat_pyr(lv_f + 1);
	vector<Mat> img_ao_dx_fmat_pyr(lv_f + 1);
	vector<Mat> img_ao_dy_fmat_pyr(lv_f + 1);
	vector<Mat> img_bo_dx_fmat_pyr(lv_f + 1);
	vector<Mat> img_bo_dy_fmat_pyr(lv_f + 1);

	ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, true, patchsz, padw, padh);
	ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, true, patchsz, padw, padh);


	imshow("Display Image1", img_ao_mat);
	imshow("Display Image2", img_bo_mat);
	imshow("pyr ao 1", img_ao_fmat_pyr[0]);
	//imshow("pyr ao 2", img_ao_fmat_pyr[1]);
	imshow("pyr aodx 1", img_ao_dx_fmat_pyr[0]);
	//imshow("pyr aodx 2", img_ao_dx_fmat_pyr[1]);
	imshow("pyr aody 1", img_ao_dy_fmat_pyr[0]);
	//imshow("pyr aody 2", img_ao_dy_fmat_pyr[1]);
    waitKey(0);

    return 0;
}