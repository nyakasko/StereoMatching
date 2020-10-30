#pragma once

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale);

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length,
  const double& center_x, const double& center_y, const double& offset);
  
void StereoEstimation_DP(
	const int& window_size,
    int height,
    int width,
    int lambda,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, const double& scale, const int& dmin);  
