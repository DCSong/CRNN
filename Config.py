# -*- coding: UTF-8 -*- 
width = 280
height = 32
image_shape = [height, width]
channels = 1
max_text_len = 10  # fix to 10 chars
num_classes = 6129  # 6128 + blank
ctc_blank = 0

batch_size = 360
epochs = 6

TEST_IMG_ROOT = './test_images'      # the root of images that you want to infer