#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "..\include\IO_flow.hpp"

using namespace cv;
using namespace std;

// what is .flo file: http://vision.middlebury.edu/flow/code/flow-code/README.txt
void ReadFlowFile(cv::Mat& img, const char* filename)
{
	FILE *stream = fopen(filename, "rb");
	if (stream == 0)
		cout << "ReadFile: could not open %s" << endl;

	int width, height;
	float tag;
	int nc = img.channels();
	float* tmp = new float[nc];

	if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
		(int)fread(&width, sizeof(int), 1, stream) != 1 ||
		(int)fread(&height, sizeof(int), 1, stream) != 1)
		cout << "ReadFile: problem reading file %s" << endl;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if ((int)fread(tmp, sizeof(float), nc, stream) != nc)
				cout << "ReadFile(%s): file is too short" << endl;

			if (nc == 1) // depth
				img.at<float>(y, x) = tmp[0];
			else if (nc == 2) // Optical Flow
			{
				img.at<cv::Vec2f>(y, x)[0] = tmp[0];
				img.at<cv::Vec2f>(y, x)[1] = tmp[1];
			}
			else if (nc == 4) // Scene Flow
			{
				img.at<cv::Vec4f>(y, x)[0] = tmp[0];
				img.at<cv::Vec4f>(y, x)[1] = tmp[1];
				img.at<cv::Vec4f>(y, x)[2] = tmp[2];
				img.at<cv::Vec4f>(y, x)[3] = tmp[3];
			}
		}
	}

	if (fgetc(stream) != EOF)
		cout << "ReadFile(%s): file is too long" << endl;

	fclose(stream);
}

// Save optical flow as .flo file
void SaveFlowFile(cv::Mat& img, const char* filename)
{
	cv::Size szt = img.size();
	int width = szt.width, height = szt.height;
	int nc = img.channels();
	float* tmp = new float[nc];

	FILE *stream = fopen(filename, "wb");
	if (stream == 0)
		cout << "WriteFile: could not open file" << endl;

	// write the header
	fprintf(stream, "PIEH");
	if ((int)fwrite(&width, sizeof(int), 1, stream) != 1 ||
		(int)fwrite(&height, sizeof(int), 1, stream) != 1)
		cout << "WriteFile: problem writing header" << endl;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (nc == 1) // depth
				tmp[0] = img.at<float>(y, x);
			else if (nc == 2) // Optical Flow
			{
				tmp[0] = img.at<cv::Vec2f>(y, x)[0];
				tmp[1] = img.at<cv::Vec2f>(y, x)[1];
			}
			else if (nc == 4) // Scene Flow
			{
				tmp[0] = img.at<cv::Vec4f>(y, x)[0];
				tmp[1] = img.at<cv::Vec4f>(y, x)[1];
				tmp[2] = img.at<cv::Vec4f>(y, x)[2];
				tmp[3] = img.at<cv::Vec4f>(y, x)[3];
			}

			if ((int)fwrite(tmp, sizeof(float), nc, stream) != nc)
				cout << "WriteFile: problem writing data" << endl;
		}
	}
	fclose(stream);
}
