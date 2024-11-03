"""
1. 使用 Global Mapper 对原始数据集做适当的标注
    1. 标注线条样式
        步骤：
        选定标注文件(.shp) -> 右键Options -> Line Styles -> Use Same Style for All Features
        -> 给对应所有线条应用相同的样式 -> 粗细：10 px; 颜色：(128, 128, 128) for Hills, (192, 192, 192) for Guyots

        目前不同实体类型以颜色区分，后续如果有需求对更多不同实体类型分开处理，可以考虑为每种实体标注线条建立一个 .shp，然后按照以上步骤，统一处理
        - Mariana: 一个 xxx-zsx.shp 是原始的标注线条，其中两个线条被删除，因为超出了图像的边界，作为补偿，重新对原实体位置做了标注，即另一个 .shp 文件包含的内容
                   所有标注的实体都被视为 hills
        - Hawaii: 一个 .shp 文件，之前有给 hills 和 guyots 的标注线条设置过不同样式，但现在工作区中的文件丢失了样式
    2. 调整光照：发光点的经纬度设置为 (90, 90)


2. 导出图片：一张不带标注，一张带标注 -> 确保两者分辨率相同，以检查标记是否超出范围

3. 使用该脚本，分别为每个数据集生成特征和标签
    该脚本不能实现一键生成，因为需要人工筛查，一般参考这里的代码放到 python 命令行中运行，运行位置为某个数据集的根目录

"""


import json
import cv2
import numpy as np
from pathlib import Path
import random

# 未标注的图： data.jpg，带标注的图：data_annotated.jpg
img = cv2.imread('./data/DeepBlue/hawaii/data.jpg')
img_ann = cv2.imread('./data/DeepBlue/hawaii/data_annotated.jpg')


# 根据颜色，提取出不同实体的标注线条，得到某种实体对应的标注黑白图像
hills_edge = np.array([128, 128, 128])
hills_mask = np.all(img_ann == hills_edge, axis=-1)
guyots_edge = np.array([192, 192, 192])
guyots_mask = np.all(img_ann == guyots_edge, axis=-1)

hills_img = np.ones((*img_ann.shape[:2], 1))
hills_img[hills_mask] = 255
guyots_img = np.ones((*img_ann.shape[:2], 1))
guyots_img[guyots_mask] = 255


# 让线条变得连续
kernel = np.ones((3, 3), np.uint8)
hills_img = cv2.dilate(hills_img, kernel, iterations=1)
guyots_img = cv2.dilate(guyots_img, kernel, iterations=1)


# 可选，暂时将不同实体的标注黑白图像保存，用于检查
cv2.imwrite('hills.png', hills_img)
cv2.imwrite('guyots.jpg', guyots_img)


def random_crop_coords(num_crops, height, width, sub_width=1024, sub_height=1024):
    """
    随机分割出 `num_crops` 个分辨率为 (sub_width, sub_height) 的子图
    确定一张子图的具体方式是依据该函数提供的四个顶点坐标
    """
    sub_images_coords = []
    for _ in range(num_crops):
        x = random.randint(0, width - sub_width)
        y = random.randint(0, height - sub_height)
        sub_images_coords.append((y, y + sub_height, x, x + sub_width))
    return sub_images_coords


# 设种子值，可复现，确保每次划分出的数据集相同
random.seed(42)
init_num = 1000  # 对 Hawaii, hills 比较少，设置的 2000
imgs_coords = random_crop_coords(init_num, *img_ann.shape[:2])


# 将不同实体的标签保存到相应目录中，标签文件的编号是随机划分子图的顺序
root_dir = Path('.')
outputs_dir = root_dir / 'labels'
outputs_dir.mkdir(exist_ok=True, parents=True)
# for entity_name, entity_img in zip(['guyots', 'hills'], [guyots_img, hills_img]):  # 此句用来生成两种实体标签
for entity_name, entity_img in zip(['hills'], [hills_img]):
    entity_labels_dir = outputs_dir / entity_name
    entity_labels_dir.mkdir(exist_ok=True, parents=True)
    for i, coord in enumerate(imgs_coords):
        label = entity_img[coord[0]: coord[1], coord[2]: coord[3]]
        cv2.imwrite(entity_labels_dir / f'crop{i}.jpg', label)


# 将标签文件下载到本地 -> 人工检查哪些标签图像中符合要求：有较完整的标注线条，记录下这些图像的编号，即 crop{i} 中的 i
# 使用以下两个变量分别保存这些编号
# guyots_idx = []
hills_idx = []

# guyots_coords = [imgs_coords[i] for i in guyots_idx]
hills_coords = [imgs_coords[i] for i in hills_idx]


# 自动删除编号不在列表中的其余文件
# for entity_name, entity_idx in zip(['guyots', 'hills'], [guyots_idx, hills_idx]):
for entity_name, entity_idx in zip(['hills'], [hills_idx]):
    entity_labels_dir = outputs_dir / entity_name
    for i in range(init_num):
        if i not in entity_idx:
            (entity_labels_dir / f'crop{i}.jpg').unlink()


# 最后用列表中的编号，生成特征文件
inputs_dir = root_dir / 'features'
inputs_dir.mkdir(exist_ok=True, parents=True)
# for entity_name, entity_idx, entity_coords in [('guyots', guyots_idx, guyots_coords),
#                                                ('hills', hills_idx, hills_coords)]:
for entity_name, entity_idx, _ in [
    # ('guyots', guyots_idx, guyots_coords),
    ('hills', hills_idx, hills_coords)
]:
    input_dir = inputs_dir / entity_name
    input_dir.mkdir(exist_ok=True, parents=True)
    for i in entity_idx:
        coord = imgs_coords[i]
        feature = img[coord[0]: coord[1], coord[2]: coord[3]]
        cv2.imwrite(input_dir / f'crop{i}.jpg', feature)


# 保存元数据，方便复现
d = {
    'seed': 42,
    # 'guyots': {
    #     'indices': guyots_idx,
    #     'coords': guyots_coords,
    #     'init_num': init_num,
    # },
    'hills': {
        'indices': hills_idx,
        'coords': hills_coords,
        'init_num': init_num,
    }
}
(root_dir / 'meta.json').write_text(json.dumps(d), encoding='utf-8')




