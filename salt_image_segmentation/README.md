# TGS Salt Identification with U-Net model  
## Overview  
This project using [this challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)'s dataset. I approach this problem using [U-Net model](https://arxiv.org/abs/1505.04597) but with different sizes of the input image (the core of the model remains the same).  

## About files  
- [TGS_Salt.ipynb](https://github.com/thanhtvt/ML-DL-projects/blob/main/salt_image_segmentation/TGS_Salt.ipynb): Build, execute and evaluate model
- [best_model.h5](https://github.com/thanhtvt/ML-DL-projects/blob/main/salt_image_segmentation/best_model.h5): Saved the best model (highest validation accuracy) after training successfully.  

## Some predictions  
![prediction](https://github.com/thanhtvt/ML-DL-projects/blob/main/salt_image_segmentation/prediction.jpg)  
  
## References
- [Image Segmentation với U-Net](https://nttuan8.com/bai-12-image-segmentation-voi-u-net/)
- [Understanding Semantic Segmentation with UNET](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)  
- [Implementation of deep learning framework - Unet, using Keras](https://github.com/zhixuhao/unet)