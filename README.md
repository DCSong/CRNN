# CRNN-DenseNet
a DenseNet-CRNN implement with PyTorch
<An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition>

## 概况
CRNN是一种经典的用于文本/场景文本识别架构
这里我们用CRNN来进行印刷体文本行的识别
提供了CRNN的两种实现：原始版本的VGG实现和DenseNet作为CNN架构的实现(后缀v2) 

文件夹结构和其他说明请见 **源代码说明.txt**

<a href="url"><img src="https://github.com/DCSong/CRNN-DenseNet/blob/master/README_imgs/CRNN.jpg" align="middle" height=40% width=40% ></a>


## 训练
训练数据集包含两部分
1. 网络开源的数据集Synthetic Chinese String Dataset （360w）
2. 程序合成（40w）


开源数据集介绍和下载请见:
- https://github.com/xiaofengShi/CHINESE-OCR
- https://github.com/YCG09/chinese_ocr 

训练时，运行main.py或main_v2.py，batch_size和epochs等参数可在Config.py里修改

合成数据集的效果：

<a href="url"><img src="https://github.com/DCSong/CRNN-DenseNet/blob/master/README_imgs/genData.jpg" align="middle" height=80% width=80% ></a>


## 测试
这里提供了两个版本的训练参数，checkpoint.pth.tar和checkpoint_v2.pth.tar

测试时，将需要测试的文本行图片放在test_images文件夹中，直接运行infer.py或infer_v2.py就可以进行测试


## 效果
<a href="url"><img src="https://github.com/DCSong/CRNN-DenseNet/blob/master/README_imgs/result2.png" align="middle" height=70% width=70% ></a>
---

<a href="url"><img src="https://github.com/DCSong/CRNN-DenseNet/blob/master/README_imgs/result4.png" align="middle" height=70% width=70% ></a>
---

<a href="url"><img src="https://github.com/DCSong/CRNN-DenseNet/blob/master/README_imgs/result3.png" align="middle" height=70% width=70% ></a>


## 参考
[1] https://www.cnblogs.com/sierkinhane/p/9715582.html

[2] https://github.com/Sierkinhane/crnn_chinese_characters_rec

[3] https://github.com/xiaofengShi/CHINESE-OCR

[4] https://github.com/YCG09/chinese_ocr
