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
    const int dmin = 67;
    int window_size = 3;
    double weight = 500;
    const double scale = 3;

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
    cv::Mat naive_disparities = cv::Mat::zeros(height - window_size, width - window_size, CV_8UC1);
    // cv::Mat naive_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    // DP disparity image
    cv::Mat dp_disparities = cv::Mat::zeros(height, width, CV_8UC1);
    int lambda = 10;
    StereoEstimation_DP(
        height,
        width,
        lambda,
        image1, image2,
        dp_disparities, scale);
    // save / display images
    std::stringstream out2;
    out2 << output_file << "_dp.png";
    cv::imwrite(out2.str(), dp_disparities);

    cv::namedWindow("DP", cv::WINDOW_AUTOSIZE);
    cv::imshow("DP", dp_disparities);
    cv::waitKey(0);

    StereoEstimation_Naive(
        window_size, dmin, height, width,
        image1, image2,
        naive_disparities, scale);

    //////////
    //Output//
    //////////

    // reconstruction
    Disparity2PointCloud(
        output_file,
        height, width, naive_disparities,
        window_size, dmin, baseline, focal_length);

    // save / display images
    std::stringstream out1;
    // out1 << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);

    cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive", naive_disparities);

    cv::waitKey(0);

    return 0;
}

void StereoEstimation_DP(
    int height,
    int width,
    int lambda,
    cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, const double& scale){
    for (int sor = 0; sor < height; ++sor)
    {
        cv::Mat C = cv::Mat::zeros(width, width, CV_8UC1);
        cv::Mat M = cv::Mat::ones(width, width, CV_8UC1);
        std::cout
            << "Calculating disparities for the DP approach... "
            << std::ceil((sor / static_cast<double> (height)) * 100) << "%\r"
            << std::flush;
        for (int i = 1; i < width; ++i) {
            C.at<uchar>(i, 1) = C.at<uchar>(i - 1, 1) + lambda;
        }
        for (int j = 1; j < width; ++j) {
            C.at<uchar>(1, j) += lambda;
        }

        for (int i = 1; i < width; ++i) {
            for (int j = 1; j < width; ++j) {
                int val = image1.at<uchar>(sor, i) - image2.at<uchar>(sor, j);
                int dissim = val * val;
                int min1 = C.at<uchar>(i - 1, j - 1) + dissim;
                int min2 = C.at<uchar>(i - 1, j) + lambda;
                int min3 = C.at<uchar>(i, j - 1) + lambda;
                int min = std::min({ min1, min2, min3 });

                if (min == min1) M.at<uchar>(i, j) = 1;
                else if (min == min2) M.at<uchar>(i, j) = 2;
                else if (min == min3) M.at<uchar>(i, j) = 3;
            }
        }

        int index_i = width - 1;
        int index_j = width - 1;
        while (index_i > 0 && index_j > 0) {
            if (M.at<uchar>(index_i, index_j) == 1) {
                dp_disparities.at<uchar>(sor, index_j) = std::abs(index_i - index_j) * scale;
                index_i--;
                index_j--;
            }
            if (M.at<uchar>(index_i, index_j) == 2) {

                index_j--;
            }
            if (M.at<uchar>(index_i, index_j) == 3) {
                dp_disparities.at<uchar>(sor, index_j) = 0;
                index_i--;
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
    const int& dmin, const double& baseline, const double& focal_length)
{
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());
    for (int i = 0; i < height - window_size; ++i) {
        std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((i) / static_cast<double>(height - window_size + 1)) * 100) << "%\r" << std::flush;
        for (int j = 0; j < width - window_size; ++j) {
            if (disparities.at<uchar>(i, j) == 0) continue;

            // TODO
            const double Z = (baseline * focal_length) / disparities.at<uchar>(i, j);
            const double X = ( i * Z ) / disparities.at<uchar>(i, j);
            const double Y = ( j * Z ) / disparities.at<uchar>(i, j);
            
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }

    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}
 