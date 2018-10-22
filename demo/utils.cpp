/* Utilites implemetation for demo applications of Computer Vision Library.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include <opencv2/opencv.hpp>

#include <vector>

#include "utils.hpp"

namespace utils
{
moving_average::moving_average(int length /*= 30*/) : history_length_(length)
{
    history_.resize(history_length_, 0.0);
}

double moving_average::add_new_value(double val)
{
    if (!is_initialized_)
    {
        if (current_index_ < history_length_)
        {
            history_[current_index_] = val;
            accumulator_ += val;
            return accumulator_ / current_index_++;
        }
        else
        {
            is_initialized_ = true;
        }
    }

    if (current_index_ == history_length_)
    {
        current_index_ = 0;
    }

    accumulator_ -= history_[current_index_];
    history_[current_index_++] = val;
    accumulator_ += val;

    return accumulator_ / history_length_;
}

fps_counter::fps_counter(int length /*= 30*/) : last_tick_count_(cv::getTickCount()), accum_(length)
{
}

double fps_counter::process_new_frame()
{
    const int64 new_tick_count = cv::getTickCount();
    const double diff = cv::getTickFrequency() / (new_tick_count - last_tick_count_);
    last_tick_count_ = new_tick_count;
    return accum_.add_new_value(diff);
}

void put_fps_text(cv::Mat& image, fps_counter& fps, cv::Scalar color /*= (255, 0, 0)*/)
{
    const auto txtFont = CV_FONT_HERSHEY_SIMPLEX;
    const auto fontScale = 0.5;
    const auto thickness = 1;
    static const cv::Size textSize = cv::getTextSize("fps: 19.12756", txtFont, fontScale, thickness, nullptr);
    static const cv::Point textOrgPoint = {image.size().width / 2 - textSize.width / 2, image.size().height - 5};
    static const cv::Point rectPoint1 = {image.size().width / 2 - textSize.width / 2 - 2, image.size().height - 17};
    static const cv::Point rectPoint2 = {image.size().width / 2 + textSize.width / 2 + 2, image.size().height};

    std::stringstream ss;
    ss.precision(5);

    ss << "fps: " << std::fixed << fps.process_new_frame();

	cv::Rect fpsRect = cv::Rect(rectPoint1, rectPoint2);
	cv::rectangle(image, fpsRect, cv::Scalar(0,0,0), -1);

    cv::putText(image, ss.str(), textOrgPoint, txtFont, fontScale, color, thickness, 8, false);
}

void put_kp_count_text(cv::Mat& image, size_t kp_count, cv::Scalar color /*= (255, 255, 255)*/)
{
    const auto txtFont = CV_FONT_HERSHEY_SIMPLEX;
    const auto fontScale = 0.5;
    const auto thickness = 1;
    static const cv::Size textSize = cv::getTextSize("KP: 10000", txtFont, fontScale, thickness, nullptr);
    static const cv::Point textOrgPoint = {0, 12};
    static const cv::Point rectPoint1 = {0, 0};
    static const cv::Point rectPoint2 = {textSize.width + 2, 15};

    std::stringstream ss;
    ss << "KP: " << kp_count;

	cv::Rect fpsRect = cv::Rect(rectPoint1, rectPoint2);
	cv::rectangle(image, fpsRect, cv::Scalar(0,0,0), -1);

    cv::putText(image, ss.str(), textOrgPoint, txtFont, fontScale, color, thickness, 8, false);
}

} // namespace utils
