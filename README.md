# OpenCV_Stereo_Matching
 Disparity image generation with naive and dynamic programming approaches and 3D pointcloud generation from the disparity image
 
 [A Maximum Likelihood Stereo Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S1077314296900405)
 
 Input images:
 
<img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/view0.png" width="300" height="300">   <img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/view1.png" width="300" height="300">
 
 1. Naive implementation with SSD and a sliding window (kernel = 7)
 
       <img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/output_naive_kernel7.png" width="400" height="400">
 
 2. Dynamic programming implementation with sliding window (kernel = 5, weight = 2000)

       <img src="https://github.com/nyakasko/StereoMatching/blob/main/data/output_dp_kernel_5_weight_2000.png" width="400" height="400">
