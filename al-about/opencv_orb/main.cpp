#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace cv;
using namespace std;

int main()
{
	string path = "guy/";
	Mat m1 = imread(path+"guy2.jpeg", IMREAD_GRAYSCALE);
	Mat m2 = imread(path+"guy3.jpeg", IMREAD_GRAYSCALE);
	imshow("1", m1);
	imshow("2", m2);

	Ptr<ORB> orb = ORB::create(8000);
	Mat desp1, desp2;
	std::vector<KeyPoint> key1, key2;
	orb->detectAndCompute(m1, Mat(), key1, desp1);
	orb->detectAndCompute(m2, Mat(), key2, desp2);
	drawKeypoints(m1, key1, m1);
	drawKeypoints(m2, key2, m2);
	imshow("3", m1);
	imshow("4", m2);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector<DMatch> matches;
	std::vector<DMatch> good_matches;
	double max_dist = 0, min_dist = 10000;
	matcher->match(desp1, desp2, matches);
	for(int m=0; m<desp1.rows; m++)
	{
		double dist = matches[m].distance;
		if(dist < min_dist) min_dist = dist;
		if(dist > max_dist) max_dist = dist;
	}
	for(int m=0; m<desp1.rows; m++)
	{
		if(matches[m].distance < 0.6*max_dist)
		{
			good_matches.push_back(matches[m]);
		}
	}

	Mat m_matches;
	drawMatches(m1, key1, m2, key2, good_matches, m_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("5", m_matches);

	waitKey(0);

	return 0;
}




