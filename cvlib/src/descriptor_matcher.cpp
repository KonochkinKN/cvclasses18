/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    cv::RNG rnd;
    for (int i = 0; i < q_desc.rows; ++i)
    {
        // \todo implement Ratio of SSD check.
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{	
	if (trainDescCollection.empty()) return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];
    matches.resize(q_desc.rows);
	
	for (int i = 0; i < q_desc.rows; ++i)
    {
		int nearestPoint1 = 0; // first nearest
		int nearestPoint2 = 0; // second nearest
		double minDist1 = maxDistance;
		double minDist2 = maxDistance;
		std::vector<int> goodMatches;  // good matches for knnMatch
		std::vector<double> distances; // and their distances

		// search for a good matches (radiusMatch)
		for (int j = 0; j < t_desc.rows; ++j)
		{
			double dist = cv::norm(q_desc.row(i) - t_desc.row(j), cv::NORM_L2);
			if (dist < maxDistance)
			{
				goodMatches.push_back(j);
				distances.push_back(dist);
				if (minDist1 > dist)
				{
					nearestPoint2 = nearestPoint1;
					minDist2 = minDist1;
					nearestPoint1 = j;
					minDist1 = dist;
				}
				else if (minDist2 > dist)
				{
					nearestPoint2 = j;
					minDist2 = dist;
				}
			}
		}

		// check SSD ratio
		if (minDist1 / minDist2 <= ratio_ && goodMatches.size() > 1)
			matches[i].emplace_back(i, nearestPoint1, (float)minDist1);
    }
}

} // namespace cvlib
