# OpenCV_naive_stereo
 Disparity image generation with naive and dynamic programming approaches
 
 Input images:
 
<img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/view0.png" width="300" height="300">   <img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/view1.png" width="300" height="300">
 
 1. Naive implementation with SSD and a sliding window (kernel = 7)
 
       <img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/output_naive_kernel7.png" width="400" height="400">
 
 2. Dynamic programming implementation with sliding window (kernel = 3)

       <img src="https://github.com/nyakasko/OpenCV_Stereo_Matching/blob/main/data/output_dp_kernel3.png" width="400" height="400">
