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

## General Step:
1. Prepare dataset.
2. Run k_mean_ddl with different discount factors to see the effect

## Prepare Data
- frames should be extracted from videos with the same height and width.
- only grayscale images is used in our experiment, so conversion is needed for rgb images.
- Always normalize the data before using.
- After preparation, data would normally be in shape [h, w, total_frames] (or [h*w, total_frames]).


