/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <cstring>
#include <iostream>
 
/* detection area
 *	. . X U X . .
 *	. X . . . X .
 *	X . . . . . X
 *	L . . C . . R
 *	X . . . . . X
 *	. X . . . X .
 *	. . X D X . .
 */
 
 /* description area
  *  . . . . . . . . .
  *  . 7 . . 0 . . 1 .
  *  . . . . . . . . .
  *  . . . . . . . . .
  *  . 6 . . C . . 2 .
  *  . . . . . . . . .
  *  . . . . . . . . .
  *  . 5 . . 4 . . 3 .
  *  . . . . . . . . .
  */
  
  /* region directions
   *  7 0 1
   *  6 . 2
   *  5 4 3
   */

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

// Check 4 main pixels (Up, Down, Left and Right) and if may-be-corner, check other 12 pixels
void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear(); cv::Mat mat = image.getMat();
	if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	const int thresh = 20;
	const int WS[16] = {0,3, 0,-3,1,2,3, 3, 2, 1,-1,-2,-3,-3,-2,-1}; // Width Shifts
	const int HS[16] = {3,0,-3, 0,3,2,1,-1,-2,-3,-3,-2,-1, 1, 2, 3}; // Height Shifts
	for (int w = 3; w < mat.size().width - 3; w++) { // 3 - radius of segment
		for (int h = 3; h < mat.size().height - 3; h++) {
			int cntP = 0, cntM = 0, arrI[4], cntI = 0; bool isOk = true;
			double diff, pixC = (double)mat.at<unsigned char>(h,w); // central pixel
			for (int i = 0; i < 16 && (i<4||cntM>2||cntP>2) && cntI < 4; i++) {
				diff = pixC-(double)mat.at<unsigned char>(h+HS[i],w+WS[i]);
				if (diff > thresh) cntP++; else if (-diff > thresh) cntM++;
				else {arrI[cntI] = i*(1+3*(i<4))-(i>3)*(4-(i-1)/3); cntI++;}}
			for (int i = 0; i < cntI && cntI < 4; i++) for(int j = i+1; j < cntI && isOk; j++){
				diff = std::abs(arrI[i] - arrI[j]); if (diff > 3 && diff < 13) isOk = false; }
			if ((cntP >= 12 || cntM >= 12) && isOk) keypoints.push_back(cv::KeyPoint((float)w,(float)h,3.0f)); }}
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
	const int r = 4; // descriptor area radius
    const int desc_length = 12;
	const int ws[8] = {0,1,1,1,0,-1,-1,-1}; // width shifts
	const int hs[8] = {-1,-1,0,1,1,1,0,-1}; // height shifts
	const int cw[9] = {4,4,7,7,7,4,1,1,1}; // central widths
	const int ch[9] = {4,1,1,4,7,7,7,4,1}; // central heights

    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);
	cv::Mat mat = image.getMat();
	if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	int w = mat.size().width, h = mat.size().height;
	cv::hconcat(mat.col(0), mat, mat); // expand image
	cv::hconcat(mat, mat.col(w), mat);
	cv::vconcat(mat.row(0), mat, mat);
	cv::vconcat(mat, mat.row(h), mat);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
		int x = (int)pt.pt.x, y = (int)pt.pt.y;
		cv::Mat area = mat(cv::Range(y - r + 1, y + r), cv::Range(x - r + 1, x + r));
		double min, max;
        cv::minMaxLoc(area, &min, &max);
		cv::Mat mmean, mstd;
		cv::meanStdDev(area, mmean, mstd);
		float mean = (float)mmean.at<double>(0);
		float std = (float)mstd.at<double>(0);
		
		int directions[9]; // {main + 8 others}
		// find main direction
		directions[0] = 0;
		int pixC = (int)area.at<unsigned char>(ch[0],cw[0]);
		int main_diff = std::abs(pixC - (int)area.at<unsigned char>(ch[1],cw[1]));
		for (int i = 2; i < 9; i++)
		{
			int diff = std::abs(pixC - (int)area.at<unsigned char>(ch[i],cw[i]));
			if (main_diff < diff)
			{
				main_diff = diff;
				directions[0] = i-1;
			}
		}
		
		// find other directions with respect to main direction
		for (int j = 1; j < 9; j++)
		{
			directions[j] = 0;
			pixC = (int)area.at<unsigned char>(ch[j],cw[j]);
			main_diff = std::abs(pixC - (int)area.at<unsigned char>(ch[j]+hs[0],cw[0]+ws[0]));
			for (int i = 1; i < 8; i++)
			{
				int diff = std::abs(pixC - (int)area.at<unsigned char>(ch[j]+hs[i],cw[j]+ws[i]));
				if (main_diff < diff)
				{
					main_diff = diff;
					directions[j] = i;
				}
			}
			directions[j] = (directions[j] + directions[0])%8;
		}
		
		// fill descriptor
		for (int i = 0; i < 8; i++)
		{
			*ptr = (directions[i+1]+1)*1000;
			ptr++;
		}
		*ptr = int(mean/max*(1e4));
		ptr++;
		*ptr = int(std/max*(1e4));
		ptr++;
		*ptr = (int)max;
		ptr++;
		*ptr = (int)min;
		ptr++;
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
