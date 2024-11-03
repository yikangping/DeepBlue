import cv2
import os
import numpy as np

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

def filter_contours(contours, convex_hulls, min_circularity=0.1, max_variance=1, min_mean_curvature=0.1, min_area=600, max_aspect_ratio=3):
    """根据圆度、曲率方差、平均曲率、最小面积、凸化面积比和长条形过滤筛选轮廓"""
    filtered_contours = []
    for contour, hull in zip(contours, convex_hulls):
        # 计算原始轮廓和凸化后轮廓的面积
        original_area = cv2.contourArea(contour)
        convex_area = cv2.contourArea(hull)
        
        # 计算曲率、圆度等属性进行进一步筛选
        points = contour[:, 0, :]
        curvatures = compute_curvature(points)
        mean_curvature = np.mean(curvatures)
        variance_curvature = np.var(curvatures)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * convex_area / (perimeter**2) if perimeter > 0 else 0
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width / (height+0.1), height / (width+0.1))

        # 过滤条件
        if (
            circularity >= min_circularity and
            variance_curvature <= max_variance and
            mean_curvature >= min_mean_curvature and
            original_area >= min_area 
            and aspect_ratio <= max_aspect_ratio
        ):  # 新增长条形过滤条件
            filtered_contours.append(hull)
    
    return filtered_contours
'''
def filter_contours(contours, convex_hulls, min_circularity=0.05, max_variance=0.05, min_mean_curvature=0.1, min_area=600, area_factor=4):
    """根据圆度、曲率方差、平均曲率、最小面积和凸化面积比筛选轮廓"""
    filtered_contours = []
    for contour, hull in zip(contours, convex_hulls):
        # 计算原始轮廓和凸化后轮廓的面积
        original_area = cv2.contourArea(contour)
        convex_area = cv2.contourArea(hull)
        
        # 如果凸化后的面积超过原始面积的三倍，过滤掉该轮廓
        if convex_area > area_factor * original_area:
            continue

        # 计算曲率、圆度等属性进行进一步筛选
        points = contour[:, 0, :]
        curvatures = compute_curvature(points)
        mean_curvature = np.mean(curvatures)
        variance_curvature = np.var(curvatures)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * original_area / (perimeter**2) if perimeter > 0 else 0
        
        if (
            circularity >= min_circularity 
            and variance_curvature <= max_variance 
            and mean_curvature >= min_mean_curvature 
            and original_area >= min_area
        ):
            filtered_contours.append(hull)
    
    return filtered_contours
'''
    
def save_contour_to_txt(contours, txt_save_path):
    """将边界信息保存到txt文件"""
    with open(txt_save_path, 'w') as f:
        for contour in contours:
            f.write('[')
            for point in contour:
                f.write(f"({point[0][0]}, {point[0][1]}) ->")
            f.write(']\n')

def edge2contours(edge_path, txt_save_path, image_save_path, original_image_path, blank_image_output_path, line_thickness=5):
    """读取图像，提取轮廓，保存到txt并在原图和二次处理图上绘制轮廓，保存到新建的空白图片"""
    edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(original_image_path)
    
    if edge is None:
        print(f"Error: Could not read the edge image file at {edge_path}. Please check the file path and file integrity.")
        return
    if original_image is None:
        print(f"Error: Could not read the original image file at {original_image_path}. Please check the file path and file integrity.")
        return

    # 二值化图像
    ret, binary = cv2.threshold(edge, 1, 255, cv2.THRESH_BINARY)

    # 提取外部轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 只进行一次凸化
    convex_hulls = [cv2.convexHull(cnt) for cnt in contours]

    # 过滤轮廓，加入凸化面积比的筛选
    filtered_contours = filter_contours(contours, convex_hulls)

    # 保存轮廓信息到txt文件
    save_contour_to_txt(filtered_contours, txt_save_path)

    # 在原图上绘制轮廓
    cv2.drawContours(original_image, filtered_contours, -1, (100, 0, 255), line_thickness)
    
    # 保存带有轮廓的原图
    cv2.imwrite(image_save_path, original_image)

    # 新建空白图像，大小与原图相同
    blank_image = np.zeros_like(original_image)

    # 在空白图像上绘制轮廓
    cv2.drawContours(blank_image, filtered_contours, -1, (255, 255, 255), line_thickness)

    # 保存带有轮廓的空白图像
    cv2.imwrite(blank_image_output_path, blank_image)

if __name__ == "__main__":
    edge_dir = r"./data/DeepBlue/model_output"
    target_dir_txt = r"./final_output/txt"
    target_dir_img = r"./final_output/contour_images"
    original_image_dir = r"./data/DeepBlue/test"
    output_image_dir = r"./final_output/contour_on_original_images"  # 保存带有轮廓的原图的路径
    blank_image_output_dir = r"./final_output/contour_on_blank_images"  # 保存带有轮廓的空白图片的路径

    if not os.path.exists(target_dir_txt):
        os.makedirs(target_dir_txt)
    if not os.path.exists(target_dir_img):
        os.makedirs(target_dir_img)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(blank_image_output_dir):
        os.makedirs(blank_image_output_dir)

    file_list = os.listdir(edge_dir)
    for file_name in file_list:
        print(file_name)
        save_name = os.path.splitext(file_name)[0]
        edge_path = os.path.join(edge_dir, file_name)
        txt_save_path = os.path.join(target_dir_txt, save_name + ".txt")
        image_save_path = os.path.join(output_image_dir, save_name + "_contour_on_original.jpg")
        blank_image_output_path = os.path.join(blank_image_output_dir, save_name + "_contour_on_blank.png")
        
        original_image_path = os.path.join(original_image_dir, save_name + ".jpg")
        
        # 处理并绘制轮廓
        edge2contours(edge_path, txt_save_path, image_save_path, original_image_path, blank_image_output_path, line_thickness=5)
