/* Image stitcher algorithm implementation.
 * @file
 * @date 2018-12-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>

namespace cvlib
{
void Stitcher::apply(const cv::Mat& src)
{
	// std::cout << "BEGIN\n";

    if (src.empty()) return;
	if (mPanoram.empty())
	{
		cv::medianBlur(src, mPanoram, 3);
		//mPanoram = src.clone();
		return;
	}
	// std::cout << "GOT PANORAM\n";

	cv::Mat medSrc;
	cv::medianBlur(src, medSrc, 3);

	//auto detector = cvlib::corner_detector_fast::create();
	auto detector = cv::ORB::create();
	auto extractor = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_L2, false);

    std::vector<cv::KeyPoint> cornersSrc;
    std::vector<cv::KeyPoint> cornersPanoram;
    cv::Mat descriptorsSrc;
    cv::Mat descriptorsPanoram;
    std::vector<cv::DMatch> matches;
	
	// std::cout << "INIT\n";

	detector->detect(medSrc, cornersSrc);
	detector->detect(mPanoram, cornersPanoram);
	
	if (cornersSrc.empty() || cornersPanoram.empty()) return;
	
	// std::cout << "DETECT\n";
	
	extractor->compute(medSrc, cornersSrc, descriptorsSrc);
	extractor->compute(mPanoram, cornersPanoram, descriptorsPanoram);
	
	// std::cout << "COMPUTE\n";

	matcher.match(descriptorsSrc, descriptorsPanoram, matches);

	// std::cout << "MATCHES: " << matches.size() <<"\n";
	
	double maxDist = 0; double minDist = 100;
	for (int i = 0; i < matches.size(); i++)
	{ 
		double dist = matches[i].distance;
		if (dist < minDist) minDist = dist;
		if (dist > maxDist) maxDist = dist;
	}
	
	if (minDist < 1) minDist = 1;

	// std::cout << "MIN DIST: " << minDist <<"\n";
	// std::cout << "MAX DIST: " << maxDist <<"\n";

	std::vector<cv::DMatch> goodMatches;
	for(int i = 0; i < matches.size(); i++)
	{ 
		if (matches[i].distance <= 3*minDist)
			goodMatches.push_back(matches[i]);
	}

	// std::cout << "GOOD MATCHES: " << goodMatches.size() << std::endl;
	
	if (goodMatches.size() < 4) return;

	std::vector<cv::Point2f> frame;
	std::vector<cv::Point2f> scene;

	for( int i = 0; i < goodMatches.size(); i++ )
	{
		frame.push_back(cornersSrc[goodMatches[i].queryIdx].pt);
		scene.push_back(cornersPanoram[goodMatches[i].trainIdx].pt);
	}

	cv::Rect frameRect = cv::boundingRect(frame);
	cv::Rect sceneRect = cv::boundingRect(scene);

	int dw = (frameRect.x + frameRect.width/2) - (sceneRect.x + sceneRect.width/2);
	int dh = (frameRect.y + frameRect.height/2) - (sceneRect.y + sceneRect.height/2);
	
	int saW = 0; // shared area Width
	int saH = 0; // shared area Height

	if (dw>0) // scene to the right
		saW = sceneRect.x + sceneRect.width/2 + src.cols - (frameRect.x + frameRect.width/2);
	else if (dw<0)// scene to the left
		saW = frameRect.x + frameRect.width/2 + mPanoram.cols - (sceneRect.x + sceneRect.width/2);
	else
		saW = std::min(mPanoram.cols, src.cols);

	if (dh>0) // scene to the down
		saH = sceneRect.y + sceneRect.height/2 + src.rows - (frameRect.y + frameRect.height/2);
	else if (dh<0)// scene to the up
		saH = frameRect.y + frameRect.height/2 + mPanoram.rows - (sceneRect.y + sceneRect.height/2);
	else
		saH = std::min(mPanoram.rows, src.rows);

	if (saW > std::min(mPanoram.cols, src.cols)) saW = std::min(mPanoram.cols, src.cols);
	if (saH > std::min(mPanoram.rows, src.rows)) saH = std::min(mPanoram.rows, src.rows);

	int width = mPanoram.cols + src.cols - saW;
	int height = mPanoram.rows + src.rows - saH;

	cv::Mat result(height, width, mPanoram.type(), cv::Scalar::all(0));
	cv::Rect sceneRoi( (dw>0)*(result.cols - mPanoram.cols), (dh>0)*(result.rows - mPanoram.rows), mPanoram.cols, mPanoram.rows);
	mPanoram.copyTo(result(sceneRoi));

	cv::Rect frameRoi( (dw<0)*(result.cols - src.cols), (dh<0)*(result.rows - src.rows), src.cols, src.rows);
	medSrc.copyTo(result(frameRoi));
	
	int sharedRoiX = (dw>0) ? sceneRoi.x : frameRoi.x;
	int sharedRoiY = (dh>0) ? sceneRoi.y : frameRoi.y;
	cv::Rect sharedRoi(sharedRoiX, sharedRoiY, saW, saH);
	
	int sharedRoiSceneX = (dw<0) ? mPanoram.cols - saW : 0;
	int sharedRoiSceneY = (dh<0) ? mPanoram.rows - saH : 0;
	cv::Rect sharedRoiScene(sharedRoiSceneX, sharedRoiSceneY, saW, saH);
	
	int sharedRoiFrameX = (dw>0) ? src.cols - saW : 0;
	int sharedRoiFrameY = (dh>0) ? src.rows - saH : 0;
	cv::Rect sharedRoiFrame(sharedRoiFrameX, sharedRoiFrameY, saW, saH);
	
	cv::Mat sharedArea = mPanoram(sharedRoiScene).clone()/2;
	sharedArea += medSrc(sharedRoiFrame).clone()/2;
	sharedArea.copyTo(result(sharedRoi));

	mPanoram.release();
	mPanoram = result.clone();
}

cv::Mat Stitcher::getPanoram() const
{
	return mPanoram;
}

} // namespace cvlib
