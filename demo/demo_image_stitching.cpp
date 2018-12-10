/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"

int demo_image_stitching_old(int argc, char* argv[])
{
	cv::Mat lol;
	cv::Mat frame0 = cv::imread("frame2.jpg");
	cv::Mat frame1 = cv::imread("frame1.jpg");
	
	std::cout << "Frame #0 is " << (frame0.empty() ? "not " : "") << "ok" << std::endl;
	std::cout << "Frame #1 is " << (frame1.empty() ? "not " : "") << "ok" << std::endl;

	cvlib::Stitcher stitcher;
	stitcher.apply(frame0);
	stitcher.apply(frame1);
	
	cv::waitKey(0);
	
    return 0;
}

int demo_image_stitching(int argc, char* argv[])
{
	cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
	cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
	
	cvlib::Stitcher stitcher;
	
    int pressed_key = 0;
    utils::fps_counter fps;
	while (pressed_key != 27) // ESC
    {
        cap >> frame;
		stitcher.apply(frame);
		cv::imshow(demo_wnd, stitcher.getPanoram());

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
        }
		
        utils::put_fps_text(frame, fps);
        cv::imshow(main_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
	
    return 0;
}
