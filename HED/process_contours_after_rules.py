import cv2
import numpy as np
import os
from PIL import Image


def compute_curvature(points):
    """计算曲线上所有点的曲率。返回一个曲率数组"""
    curvatures = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        p3 = np.array(points[i + 1])
        v1 = p2 - p1
        v2 = p3 - p2
        curvature = np.linalg.norm(np.cross(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
        curvatures[i] = curvature
    return curvatures

def filter_contours(contours, convex_hulls, min_circularity=0.15, max_variance=0.1, min_mean_curvature=0.1, min_area=400, max_aspect_ratio=3):
    """根据圆度、曲率方差、平均曲率、最小面积、凸化面积比和长条形过滤筛选轮廓"""
    filtered_contours = []
    for contour, hull in zip(contours, convex_hulls):
        original_area = cv2.contourArea(contour)
        convex_area = cv2.contourArea(hull)
        
        points = contour[:, 0, :]
        curvatures = compute_curvature(points)
        mean_curvature = np.mean(curvatures)
        variance_curvature = np.var(curvatures)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * convex_area / (perimeter**2) if perimeter > 0 else 0
        
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width / (height + 0.1), height / (width + 0.1))

        if (circularity >= min_circularity and
            variance_curvature <= max_variance and
            mean_curvature >= min_mean_curvature and
            original_area >= min_area and
            aspect_ratio <= max_aspect_ratio):
            filtered_contours.append(hull)
    
    return filtered_contours

# 定义路径
wjj = os.path.join(r'HED/data/DeepBlue', 'test')

# 定义保存文件夹路径，确保中文路径为 Unicode 格式
original_contours_folder = "HED/output2/original"
blank_contours_folder = "HED/output2/blank"

# 如果文件夹不存在，则创建它们
if not os.path.exists(original_contours_folder):
    os.makedirs(original_contours_folder)

if not os.path.exists(blank_contours_folder):
    os.makedirs(blank_contours_folder)

# 遍历 wjj 文件夹下的所有文件
file_list = os.listdir(wjj)
for file_name in file_list:
    # 获取文件名和扩展名
    wj, hzm = os.path.splitext(file_name)

    # 读取彩色图像，拼接图像路径
    image_path = os.path.join(wjj, file_name)

    # 检查图像路径是否存在
    if not os.path.exists(image_path):
        print(f"图像路径不存在: {image_path}")
        continue

    # 读取彩色图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 检查图像是否成功读取
    if image is None:
        print(f"无法读取图像，请检查路径: {image_path}")
        continue

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    median_blur = cv2.medianBlur(gaussian_blur, 5)
    bilateral_filter = cv2.bilateralFilter(median_blur, 9, 75, 75)
    _, binary_image = cv2.threshold(bilateral_filter, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    print('去除噪音')
    contours, hierarchy = cv2.findContours(morph_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    height, width = morph_image.shape
    # print(height,width)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area*4 < height*width:
            filtered_contours.append(contour)
    contours = filtered_contours
    print('跳过1', len(contours))
    convex_hulls = [cv2.convexHull(cnt) for cnt in contours]
    print('跳过2')
    filtered_contours = []
    for i,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area*4 < height*width:
            filtered_contours.append(contour)
    contours = filtered_contours
    contours = filter_contours(contours, convex_hulls)
    print('跳过23')
    temp_image = np.zeros_like(image, dtype=np.uint8)
    for i, contour in enumerate(contours):
        # 绘制单个轮廓
        cv2.drawContours(temp_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    gray_temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(r'D:\my_pro\save_temp\\'+wj+'.png',gray_temp_image)
    # 从填充的灰度图像中提取新的轮廓
    contours, _ = cv2.findContours(gray_temp_image,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area*4 < height*width:
            filtered_contours.append(contour)
    filtered_contours = []
    for contour in contours:
        keep = True
        for point in contour:
            x, y = point[0]
            # print('x,y 是',x, y)
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                keep = False
                break    
        if keep:
            filtered_contours.append(contour)
    contours = filtered_contours
    contours = filter_contours(contours, filtered_contours)
    # 再次进行形态学去噪（开运算）
    morph_image_final = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

    # 保存去噪后的黑白图像到 first_process 文件夹
    blank_with_contours_path = os.path.join(blank_contours_folder, wj + '.png')
    Image.fromarray(morph_image_final).save(blank_with_contours_path)

    # 在原图上画过滤后的红色边界
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 0, 255), 2)  # 红色描边，边界厚度为2

    # 将 OpenCV 图像转换为 Pillow 图像
    output_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    # 保存带红色边界的图像到 '第一次处理的效果' 文件夹
    original_with_contours_path = os.path.join(original_contours_folder, wj + '.png')
    output_image_pil.save(original_with_contours_path)

    print(f"处理完成: {file_name}")
    print(f"黑白图像已保存到: {blank_with_contours_path}")
    print(f"红色边界图像已保存到: {original_with_contours_path}")
