# Automated-Object-Detection-with-AI
Automated Object Detection in Experimental Data Using Combination of Unsupervised and Supervised Methods.  
Our proposed method is in main.ipynb.

## kmeans.py
- implemented k-means for binary classification

## main.ipynb
- implemented SVM, Logistic Regression.
- implemented combined method, with k_mean_ddl (distance discount factor).

## extraction.m
- to extract and format a training dataset
- select_region.m is a helper function.

## General Step for combined method:
1. Prepare dataset
  - Find suitable data: A suitable data should be a movie/video that can be converted to frames. We expect the object in movie to move gradually, such as a beating heart. Quick movements and relocation of objects are not tested. 
  - Convesion: Use tools to convert the video into frames, such as ffmpeg. The following example convert video to frames with fps=30. It is best to lossless formats as input data such as PNG, PGM.
  ```ffmpeg -i input.mp4 -framerate 30 out%d.png```
  - Processing: With images, we can load in the data with Python. 
    - If the data is colored, it should be converted to grayscale when loading. (h * w * 3 => h * w)
    - Assume we load in n images with shape h * w. Then the data should have shape [h, w, n]
    - Use minmax normalization along the n axis. Since we have n images, for the same pixel location (x,y) we can have n values with each value extracted from each image. So in another word, we should apply normalization to each pixel location individually through all n images. See code for details.
    
    
2. Get training data
  - In this dataset, we want to prepare the training data using k_mean_ddl.
  - We should run k_mean_ddl with different discount factors to see different effects. 
  - We want to select confident pixels that definitely belong to one class. For example, we should be confident that the pixels in the middle of a beating heart should always be classified as heart pixel, while pixels around corners would almost always be classified as background. We run k_mean_ddl twice with different discount factors to get two masks for the confident heart pixels and background pixels. We can extract values with the masks, and attach labels to form a dataset. Note remove_noise function can also be used here to assist selection.
  - At the end of this step, we should have a dataset with labels in the shape of [m, n+1], m is data points extracted, n is the number of image, 1 is the label. 
  - This step is interchangable with selecting pixels manually using matlab. 

3. Apply supervised methods
  - Now that we have labeled data, we can use different supervised methods, such as SVM, Logistic Regression or construct a MLP.



