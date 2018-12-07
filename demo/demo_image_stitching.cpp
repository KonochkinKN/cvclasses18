/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
	cv::Mat lol;
	//cv::Mat piece = cv::imread("piece.jpg");
	//cv::Mat source = cv::imread("source.jpg");
	cv::Mat frame0 = cv::imread("frame2.jpg");
	cv::Mat frame1 = cv::imread("frame1.jpg");
	
	std::cout << "Frame #0 is " << (frame0.empty() ? "not " : "") << "ok" << std::endl;
	std::cout << "Frame #1 is " << (frame1.empty() ? "not " : "") << "ok" << std::endl;

	cvlib::Stitcher stitcher;
	stitcher.apply(frame0, lol);
	stitcher.apply(frame1, lol);
	
	//imshow("warped", lol);
	
	cv::waitKey(0);
	
    return 0;
}
