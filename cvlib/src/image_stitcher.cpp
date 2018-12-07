/* Image stitcher algorithm implementation.
 * @file
 * @date 2018-12-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>

namespace cvlib
{
void Stitcher::apply(const cv::Mat& src, cv::Mat& dst)
{
	std::cout << "BEGIN\n";

    if (src.empty()) return;
	if (mPanoram.empty())
	{
		mPanoram = src.clone();
		dst = mPanoram.clone();
		return;
	}
	
	std::cout << "GOT PANORAM\n";

	auto detector = cvlib::corner_detector_fast::create();
	auto extractor = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_L2, false);

    std::vector<cv::KeyPoint> cornersSrc;
    std::vector<cv::KeyPoint> cornersPanoram;
    cv::Mat descriptorsSrc;
    cv::Mat descriptorsPanoram;
    std::vector<cv::DMatch> matches;
	
	std::cout << "INIT\n";

	detector->detect(src, cornersSrc);
	detector->detect(mPanoram, cornersPanoram);
	
	std::cout << "DETECT\n";
	
	extractor->compute(src, cornersSrc, descriptorsSrc);
	extractor->compute(mPanoram, cornersPanoram, descriptorsPanoram);
	
	std::cout << "COMPUTE\n";

	matcher.match(descriptorsSrc, descriptorsPanoram, matches);

	std::cout << "MATCH: " << matches.size() <<"\n";
	
	double maxDist = 0; double minDist = 100;
	for (int i = 0; i < matches.size(); i++)
	{ 
		double dist = matches[i].distance;
		if (dist < minDist) minDist = dist;
		if (dist > maxDist) maxDist = dist;
	}
	
	if (minDist < 1) minDist = 1;

	std::cout << "MIN DIST: " << minDist <<"\n";
	std::cout << "MAX DIST: " << maxDist <<"\n";

	std::vector<cv::DMatch> goodMatches;
	for(int i = 0; i < matches.size(); i++)
	{ 
		if (matches[i].distance <= 3*minDist)
			goodMatches.push_back(matches[i]);
	}

	std::cout << "GOOD MATCHES: " << goodMatches.size() << std::endl;
	
	std::vector<cv::Point2f> frame;
	std::vector<cv::Point2f> scene;

	for( int i = 0; i < goodMatches.size(); i++ )
	{
		frame.push_back(cornersSrc[goodMatches[i].queryIdx].pt);
		scene.push_back(cornersPanoram[goodMatches[i].trainIdx].pt);
	}

	cv::Rect frameRect = cv::boundingRect(frame);
	cv::Mat H = cv::findHomography(frame, scene, cv::RANSAC);
	
	std::cout << "H:\n";
	for (int i = 0; i < H.size().width; i++)
	{
		for (int j = 0; j < H.size().height; j++)
		{
			std::cout << (double)H.at<double>(i,j) << "\t";
		}
		std::cout << "\n";
	}

	std::vector<cv::Point2f> frameCorners(4);
	
	frameCorners[0] = cv::Point2f(frameRect.x, frameRect.y);
	frameCorners[1] = cv::Point2f(frameRect.x + frameRect.width, frameRect.y);
	frameCorners[2] = cv::Point2f(frameRect.x + frameRect.width, frameRect.y + frameRect.height);
	frameCorners[3] = cv::Point2f(frameRect.x, frameRect.y + frameRect.height);
	
	std::vector<cv::Point2f> sceneCorners(4);
	cv::perspectiveTransform(frameCorners, sceneCorners, H);
	
	std::cout << "OBJECT:\n";
	std::cout << frameCorners[0].x << " " << frameCorners[0].y << "\n";
	std::cout << frameCorners[1].x << " " << frameCorners[1].y << "\n";
	std::cout << frameCorners[2].x << " " << frameCorners[2].y << "\n";
	std::cout << frameCorners[3].x << " " << frameCorners[3].y << "\n";
	std::cout << "SCENE:\n";
	std::cout << sceneCorners[0].x << " " << sceneCorners[0].y << "\n";
	std::cout << sceneCorners[1].x << " " << sceneCorners[1].y << "\n";
	std::cout << sceneCorners[2].x << " " << sceneCorners[2].y << "\n";
	std::cout << sceneCorners[3].x << " " << sceneCorners[3].y << "\n";
	
	int dw = -(int)H.at<double>(0,2);
	int dh = -(int)H.at<double>(1,2);
	
	cv::Rect sceneRect(sceneCorners[0], sceneCorners[2]);
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

	int width = mPanoram.cols + src.cols - saW;
	int height = mPanoram.rows + src.rows - saH;

	cv::Mat result(height, width, mPanoram.type(), cv::Scalar::all(0));

	cv::Rect sceneRoi( (dw>0)*(result.cols - mPanoram.cols), (dh>0)*(result.rows - mPanoram.rows), mPanoram.cols, mPanoram.rows);
	mPanoram.copyTo(result(sceneRoi));

	cv::Rect frameRoi( (dw<0)*(result.cols - src.cols), (dh<0)*(result.rows - src.rows), src.cols, src.rows);
	src.copyTo(result(frameRoi));
	
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
	sharedArea += src(sharedRoiFrame).clone()/2;
	sharedArea.copyTo(result(sharedRoi));

	cv::resize(result, result, cv::Size(1280, 720));
	cv::imshow("Panoram", result);
	
	// DEMO
	cv::Mat imgMatches;
	cv::drawMatches( src, cornersSrc, mPanoram, cornersPanoram,
			   goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
			   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::line( imgMatches, frameCorners[0], frameCorners[1], cv::Scalar( 255, 255, 0), 4 );
	cv::line( imgMatches, frameCorners[1], frameCorners[2], cv::Scalar( 255, 255, 0), 4 );
	cv::line( imgMatches, frameCorners[2], frameCorners[3], cv::Scalar( 255, 255, 0), 4 );
	cv::line( imgMatches, frameCorners[3], frameCorners[0], cv::Scalar( 255, 255, 0), 4 );
	
	cv::line( imgMatches, sceneCorners[0] + cv::Point2f( src.cols, 0), sceneCorners[1] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	cv::line( imgMatches, sceneCorners[1] + cv::Point2f( src.cols, 0), sceneCorners[2] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	cv::line( imgMatches, sceneCorners[2] + cv::Point2f( src.cols, 0), sceneCorners[3] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	cv::line( imgMatches, sceneCorners[3] + cv::Point2f( src.cols, 0), sceneCorners[0] + cv::Point2f( src.cols, 0), cv::Scalar( 0, 255, 0), 4 );

	cv::resize(imgMatches, imgMatches, cv::Size(1280, 720));
	cv::imshow( "Good Matches & Object detection", imgMatches );
}

} // namespace cvlib
