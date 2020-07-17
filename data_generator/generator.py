# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import glob
import numpy as np
import os
import cv2


'''
1. 从文字库随机选择10个字符
2. 生成图片
3. 随机使用函数
'''

# 从文字库中随机选择n个字符
def sto_choice_from_info_str(quantity=10):
    start = random.randint(0, len(info_str)-11)
    end = start + 10
    random_word = info_str[start: end]
    return random_word

# 随机选择字体颜色
def random_word_color():
    font_color_choice = [[54, 54, 54], [54, 54, 54], [105, 105, 105]]
    # font_color_choice = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    font_color = random.choice(font_color_choice)

    noise = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
    font_color = (np.array(font_color) + noise).tolist()

    #print('font_color：',font_color)

    return tuple(font_color)

# 生成一张图片
def create_an_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path + bground_choice)
    # print('background:', bground_choice)
    # print(bground.size[0], bground.size[1])
    x, y = random.randint(0, bground.size[0] - width), random.randint(0, bground.size[1] - height)
    bground = bground.crop((x, y, x + width, y + height))
    return bground


# 模糊函数
def darken_func(image):
    #.SMOOTH
    #.SMOOTH_MORE
    #.GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数
    filter_ = random.choice(
        [ImageFilter.SMOOTH,
        ImageFilter.SMOOTH_MORE]
        # ImageFilter.GaussianBlur(radius=1.3)
    )
    image = image.filter(filter_)
    #image = img.resize((290,32))
    return image


# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size):
    width, height = bground_size
    #print(bground_size)
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size * 10)
    y = random.randint(0, int((height - font_size) / 4))

    return x, y

def random_font_size():
    font_size = random.randint(20, 25)

    return font_size

def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font

def main(save_path, num, file):
    # 随机选取10个字符
    random_word = sto_choice_from_info_str(10)
    # 生成一张背景图片，已经剪裁好，宽高为32*280
    raw_image = create_an_image('./data_generator/background/', 280, 32)

    # 随机选取字体大小
    font_size = random_font_size()
    # 随机选取字体
    font_name = random_font('./data_generator/font/')
    # 随机选取字体颜色
    font_color = random_word_color()

    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = random_x_y(raw_image.size, font_size)

    # 将文本贴到背景图片
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)

    # 随机选取作用函数和数量作用于图片
    #random_choice_in_process_func()
    raw_image = darken_func(raw_image)
    #raw_image = raw_image.rotate(0.3)
    # 保存文本信息和对应图片名称
    #with open(save_path[:-1]+'.txt', 'a+', encoding='utf-8') as file:
    file.write('genData' + str(num)+ '.png ' + random_word + '\n')
    raw_image.save(save_path + 'genData' + str(num)+ '.png')


if __name__ == '__main__':
    # 处理具有工商信息和电影对白语义信息的语料库，去除空格等不必要符号
    with open('./data_generator/text.txt', 'r', encoding='utf-8') as file:
        info_list = [part.strip().replace('\t', '').replace('\n', '').replace(' ', '').replace('\'', '') for part in file.readlines()]
        info_str = ''.join(info_list)

    # 图片标签存放文件
    file  = open('./data_generator/data_set/labels.txt', 'w', encoding='utf-8')
    
    total = 400      # 生成数量
    for num in range(total):
        main('./data_generator/data_set/images/', num, file)
        if num % 1000 == 0:
            print('[%d/%d]'%(num, total))
    file.close()


