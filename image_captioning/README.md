# Image Captioning  
## Overview  
Image Captioning is the process of generating textual description of an image. It uses both NLP and CV to generate captions.  
The model used in this repo:  

![model](https://github.com/thanhtvt/ML-DL-projects/blob/main/image_captioning/model.png)  
  
## Some notes
- This repo use Flickr8k dataset and some necessary .txt files, all of those can be downloaded from [here](https://drive.google.com/file/d/1Jr9K1L-grIC5UQjff4UPN5A29aJL86cn/view?usp=sharing)  
- This notebook has detail descriptions and explanation about what is going on. I also refer to some useful resources for our practice there.  
- This notebook still has a lot of ways to improve. Some of which can be:
  * Doing more **hyper-parameter tuning**.
  * Using validation set to evaluate model more accurate, avoid **overfitting**.
  * Using **[BLEU Score](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)** to evaluate and measure the performance of the model
  * ...
  
## References  
- [Ứng dụng thêm mô tả cho ảnh (image captioning)](https://nttuan8.com/bai-15-ung-dung-them-mo-ta-cho-anh-image-captioning/)
- [Image Captioning with Keras](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)
- [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)

