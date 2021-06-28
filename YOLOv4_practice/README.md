# YOLOv4 Cloud Practice  
## Overview  
Darknet YOLOv4 is faster and more accurate than real-time neural networks Google TensorFlow EfficientDet and FaceBook Pytorch/Detectron RetinaNet/MaskRCNN on Microsoft COCO dataset.  
Some resources:  
- [Paper about YOLOv4](https://arxiv.org/abs/2004.10934)  
- [YOLO's author blog](https://alexeyab84.medium.com/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe)  
![Example](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/example.jpeg)

## About files  
- [yolov4_custom_training.ipynb](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/yolov4_custom_training.ipynb): Train Custom YOLOv4 Model in Google Colab
- [yolov4_pre-trained.ipynb](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/yolov4_pre-trained.ipynb): Running Pre-trained YOLOv4 on over 80 Classes in Google Colab  
- [yolov4_training](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/yolov4_training): A directory that contains my essential files for training (upload this folder to Google Drive)

## Some notes
- I use [OIDv4_Toolkit](https://github.com/theAIGuysCode/OIDv4_ToolKit) for gathering and labeling custom dataset. You guys can view tutorial on how to use that tool [here](https://www.youtube.com/watch?v=_4A9inxGqRM)
- During training, my Colab Cloud crashed 2 times (T.T) so I was impatient for waiting for the results, which made the accuracy was not at my expectation.

## Some results  
### Running pre-trained YOLOv4
![Results](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/results.gif)  

### Train custom YOLOv4 model
![Results](https://github.com/thanhtvt/ML-DL-projects/blob/main/YOLOv4_practice/predictions.jpg)

## References  
- [Huấn luyện YOLO darknet trên google colab](https://phamdinhkhanh.github.io/2020/03/10/DarknetGoogleColab.html#421-tool-bounding-box)  
- [YOLOv4 Cloud Tutorial](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)
