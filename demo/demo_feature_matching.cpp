/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void on_ratio_changed(int value, void* ptr)
{
	((cvlib::descriptor_matcher*)(ptr))->set_ratio(float(1 - value / 100));
}

int demo_feature_matching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    auto detector = cvlib::corner_detector_fast::create();
	auto descriptor = cv::ORB::create();
    auto matcher = cvlib::descriptor_matcher(1.2f);

    /// \brief helper struct for tidy code
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

	int ratio = 10;
	int ssd = 100;
    cv::createTrackbar("SSD ratio", demo_wnd, &ratio, 50, on_ratio_changed, (void*)&matcher);
    cv::createTrackbar("SSD", demo_wnd, &ssd, 255);

    img_features ref;
    img_features test;
    std::vector<std::vector<cv::DMatch>> pairs;

    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> test.img;

        detector->detect(test.img, test.corners);
        cv::drawKeypoints(test.img, test.corners, main_frame);
        cv::imshow(main_wnd, main_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            ref.img = test.img.clone();
			detector->detect(ref.img, ref.corners);
			descriptor->compute(ref.img, ref.corners, ref.descriptors);
        }

        if (ref.corners.empty())
        {
            continue;
        }

        descriptor->compute(test.img, test.corners, test.descriptors);
        matcher.radiusMatch(test.descriptors, ref.descriptors, pairs, (float)ssd + 1.0f);
        cv::drawMatches(test.img, test.corners, ref.img, ref.corners, pairs, demo_frame);

        utils::put_fps_text(demo_frame, fps);
        cv::imshow(demo_wnd, demo_frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
