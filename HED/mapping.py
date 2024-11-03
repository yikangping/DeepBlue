import re

import os
import cv2
from script import imgs_coords
from script import img_ann

path = ('./final_output/txt')
files = os.listdir(path)
coordinate={"N": 25.00025, "S": 16.99975, "W": -162.00025, "E": -152.99975}

def pixel_to_latlon(lat_nw, lon_nw, lat_ne, lon_ne, lat_sw, lon_sw, lat_se, lon_se, width, height, x, y):
    """
    将任意像素点 (x, y) 转换为经纬度坐标.

    参数:
    lat_NW, lon_NW: 西北角的纬度和经度
    lat_NE, lon_NE: 东北角的纬度和经度
    lat_SW, lon_SW: 西南角的纬度和经度
    lat_SE, lon_SE: 东南角的纬度和经度
    width: 图片的宽度（像素）
    height: 图片的高度（像素）
    x: 目标像素的 x 坐标 (从左到右)
    y: 目标像素的 y 坐标 (从上到下)

    返回:
    lat, lon: 像素点的纬度和经度
    """
    # 计算当前 y 行的纬度 (通过插值)
    lat = lat_nw + (y / height) * (lat_sw - lat_nw)

    # 计算当前 y 行的左边和右边的经度 (通过插值)
    lon_left = lon_nw + (y / height) * (lon_sw - lon_nw)
    lon_right = lon_ne + (y / height) * (lon_se - lon_ne)

    # 计算当前 x 列的经度 (通过插值)
    lon = lon_left + (x / width) * (lon_right - lon_left)

    return lat, lon




def main():
    for file in files:
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            crop_id = file.strip(".txt").strip('crop')
            s = f.read()
            if not os.path.exists('./final_output/result'):
                os.mkdir('./final_output/result')
            with open("./final_output/result/crop" + crop_id + ".txt", "a", encoding='utf-8') as f1:
                l = s.split('\n')
                for st in l:
                    tuples = re.findall(r'\((\d+), (\d+)\)', st)
                    if tuples:
                        f1.write('[')
                        for x, y in tuples:
                            x_real = int(x) + imgs_coords[int(crop_id)][1]
                            y_real = int(y) + imgs_coords[int(crop_id)][0]
                            lat, lon = pixel_to_latlon(coordinate['N'], coordinate['W'], coordinate['N'], coordinate['E'],
                                                    coordinate['S'], coordinate['W'], coordinate['S'], coordinate['E'],
                                                    img_ann.shape[1], img_ann.shape[0], x_real, y_real)
                            f1.write(f'({lat} ,{lon})->')
                        f1.write(']')
                        f1.write('\n')
if __name__ == '__main__':
    main()