# TGS Salt Identification  
    
<a href="https://www.kaggle.com/jonnyjackk/segmentation-with-single-resnet-unet-lb-0-81"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>  

# Overview  
**TGS Salt Identification Challenge** is a competition for salt identification in the TGS Salt Identification Challenge. The challenge is to predict the presence of salt in a given seismic images.  
  
**The dataset** is a set of images captured at various locations, which contains 4000 images in the training dataset and 18000 images in the test set. Each image is 101 x 101 pixels, each pixel is classified as either salt or sediment (not salt).
  
**Model used** is a U-Net model with ResNet backbone. Details about it can be found in the [kaggle notebook](https://www.kaggle.com/jonnyjackk/segmentation-with-single-resnet-unet-lb-0-81).  
  
![model](https://github.com/thanhtvt/ML-DL-projects/blob/main/salt_image_segmentation/images/Unet_Resnet.png)
  
# Result    
[My approach](https://gusty-lentil-0c8.notion.site/Report-6a6958c13990489ab46036075168d03a) achieved the highest mean IoU score of 0.82 on the private leaderboard and 0.79 on the public leaderboard. I also added my best model's weight in each phase [here](https://drive.google.com/drive/folders/1zkCM7x4k5f1IJDZaSGz32C0ojTT4BMPv?usp=sharing) so you can continue to train from those or replicate my approach.  
   
Some of my predictions are shown below.  
  
![prediction](https://github.com/thanhtvt/ML-DL-projects/blob/main/salt_image_segmentation/images/prediction.jpg)  
