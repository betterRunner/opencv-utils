#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;


int main()
{
	Mat image1 = imread("cat.jpeg",0);
	Mat hsv1, histogram1;

	int channel[] = {0};
	int hist_size[] = {256};
	float range[] = {0, 256};
	const float* ranges[] = {range};
	cv::calcHist(&image1, 1, channel, cv::Mat(), histogram1, 1, hist_size, ranges, true, false);

	double histogram_max1;
	minMaxLoc(histogram1, 0, &histogram_max1, 0, 0);
	int scale = 2, hist_height = 256;
	Mat hist_img1 = Mat::zeros(hist_height, 256 * scale, CV_8UC3);
	for(int m=0; m<256; m++)
	{
		float bin_val = histogram1.at<float>(m);
		int intensity = cvRound(bin_val * hist_height / histogram_max1);
		rectangle(hist_img1, Point(m*scale, hist_height-1), Point((m+1)*scale-1, hist_height-intensity), CV_RGB(255,25,255));
	}
	imshow("image", image1);
	imshow("hist1", hist_img1);
	waitKey(0);

	return 0;
}


