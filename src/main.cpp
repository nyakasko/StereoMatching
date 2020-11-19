#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"
#include <omp.h>

int main(int argc, char** argv) {

    ////////////////
    // Parameters //
    ////////////////

    // camera setup parameters
    const double focal_length = 1247;
    const double baseline = 213;

    // stereo estimation parameters
    const int dmin = 200;
    int window_size = 5;
    double weight = 500;
    const double scale = 3;
    const double center_x = 2928.3;
    const double center_y = 940.545;
    const double offset = 553.54;
    ///////////////////////////
    // Commandline arguments //
    ///////////////////////////

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE, WINDOW_SIZE, WEIGHT" << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    const std::string output_file = argv[3];
    if (argv[4]) window_size = atoi(argv[4]);
    if (argv[5]) weight = atoi(argv[5]);

    if (!image1.data) {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data) {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "occlusion weights = " << weight << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "scaling of disparity images to show = " << scale << std::endl;
    std::cout << "output filename = " << argv[3] << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    int height = image1.size().height;
    int width = image1.size().width;
    ////////////////////
    // Reconstruction //
    ////////////////////

    // Naive disparity image
    cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    // DP disparity image
    cv::Mat dp_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    int lambda = weight;

    StereoEstimation_DP(
        window_size,
        height,
        width,
        lambda,
        image1, image2,
        dp_disparities, scale, dmin);

    // save and display images
    std::stringstream out2;
    out2 << output_file << "_dp.png";
    cv::imwrite(out2.str(), dp_disparities);

    cv::namedWindow("DP", cv::WINDOW_AUTOSIZE);
    cv::imshow("DP", dp_disparities);
    cv::waitKey(0);

    //StereoEstimation_Naive(
    //    window_size, dmin, height, width,
    //    image1, image2,
    //    naive_disparities, scale);

    ////////////
    // Output //
    ////////////

    // reconstruction
    Disparity2PointCloud(
        output_file,
        height, width, dp_disparities,
        window_size, dmin, baseline, focal_length, center_x, center_y, offset);

    // save and display images
    std::stringstream out1;
    out1 << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);

    cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive", naive_disparities);

    cv::waitKey(0);

    return 0;
}

void StereoEstimation_DP(
    const int& window_size,
    int height,
    int width,
    int lambda,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, const double& scale, const int& dmin) {

    int half_window_size = window_size / 2;
    for (int sor = half_window_size; sor < height - half_window_size; ++sor)
    {
        cv::Mat C = cv::Mat::zeros(width, width, CV_32F);
        cv::Mat M = cv::Mat::ones(width, width, CV_8UC1);
        std::cout
            << "Calculating disparities for the DP approach...  "
            << std::ceil(((sor - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;

        C.at<float>(0, 0) = lambda;
        M.at<uchar>(0, 0) = 3;

        #pragma omp parallel for
        for (int i = half_window_size + 1; i < width - half_window_size; ++i) {
            C.at<float>(i - half_window_size, 0) = C.at<float>(i - half_window_size - 1, 0) + lambda; // maybe i*occlusion?
            M.at<uchar>(i - half_window_size, 0) = 2; // left occlusion
            C.at<float>(0, i - half_window_size) += C.at<float>(0, i - half_window_size - 1) + lambda;
            M.at<uchar>(0, i - half_window_size) = 3; // right occlusion
        }
        #pragma omp parallel for
        for (int i = half_window_size + 1; i < width - half_window_size; ++i) {
            for (int j = half_window_size + 1; j < width - half_window_size; ++j) {
                // TODO: sum up matching cost (ssd) in a window
                int val = 0;
                for (int u = -half_window_size; u <= half_window_size; ++u) {
                    for (int v = -half_window_size; v <= half_window_size; ++v)
                    {
                        int val_left = image1.at<uchar>(sor + u, i + v);
                        int val_right = image2.at<uchar>(sor + u, j + v);

                        val += (val_left - val_right) * (val_left - val_right);
                    }
                }
                int min1 = C.at<float>(i - half_window_size - 1, j - half_window_size - 1) + val;
                int min2 = C.at<float>(i - half_window_size - 1, j - half_window_size) + lambda; // left occlusion
                int min3 = C.at<float>(i - half_window_size, j - half_window_size - 1) + lambda; // right occlusion
                int min = std::min({ min1, min2, min3 });
                C.at<float>(i - half_window_size, j - half_window_size) = min; // Update the cost matrix

                if (min == min1) M.at<uchar>(i - half_window_size, j - half_window_size) = 1;
                else if (min == min2) M.at<uchar>(i - half_window_size, j - half_window_size) = 2;  // left occlusion
                else if (min == min3) M.at<uchar>(i - half_window_size, j - half_window_size) = 3;  // right occlusion

            }
        }

        int index_i = width - half_window_size - 1;
        int index_j = width - half_window_size - 1;
        int k = width - half_window_size - 1;
        while (index_i > half_window_size && index_j > half_window_size) {
            if (M.at<uchar>(index_i - half_window_size, index_j - half_window_size) == 1) {

                dp_disparities.at<uchar>(sor - half_window_size, k - half_window_size) = index_j - index_i;
                index_i--;
                index_j--;
                k--;
            }
            if (M.at<uchar>(index_i - half_window_size, index_j - half_window_size) == 2) {
                // dp_disparities.at<uchar>(sor - half_window_size, index_j - half_window_size) = 0;
                index_i--;  // left occlusion
            }
            if (M.at<uchar>(index_i - half_window_size, index_j - half_window_size) == 3) {
                index_j--;  // right occlusion
            }
        }
    }
}

void StereoEstimation_Naive(
    const int& window_size,
    const int& dmin,
    int height,
    int width,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, const double& scale)
{
    int half_window_size = window_size / 2;

    for (int i = half_window_size; i < height - half_window_size; ++i) {

        std::cout
            << "Calculating disparities for the naive approach... "
            << std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
            << std::flush;
#pragma omp parallel for
        for (int j = half_window_size; j < width - half_window_size; ++j) {
            int min_ssd = INT_MAX;
            int disparity = 0;

            for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
                int ssd = 0;

                // TODO: sum up matching cost (ssd) in a window
                for (int u = -half_window_size; u <= half_window_size; ++u) {
                    for (int v = -half_window_size; v <= half_window_size; ++v)
                    {
                        int val_left = image1.at<uchar>(i + u, j + v);
                        int val_right = image2.at<uchar>(i + u, j + v + d);

                        ssd += (val_left - val_right) * (val_left - val_right);
                    }
                }


                if (ssd < min_ssd) {
                    min_ssd = ssd;
                    disparity = d;
                }
            }

            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity) * scale;
        }
    }

    std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

void Disparity2PointCloud(
    const std::string& output_file,
    int height, int width, cv::Mat& disparities,
    const int& window_size,
    const int& dmin, const double& baseline, const double& focal_length, const double& center_x, const double& center_y, const double& offset)
{
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());

    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;
            // TODO
            double Z = (baseline * focal_length) / (int(disparities.at<uchar>(i, j)) + dmin);
            double X = (i - center_x) * Z / focal_length;
            double Y = (j - center_y) * Z / focal_length;
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}