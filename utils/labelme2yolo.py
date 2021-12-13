import json
import os
import cv2
import math
import tqdm

img_folder_path = r'/home/zxl/repos/TruckDetection/data/yolov5/打完标/val'  # 图片存放文件夹
folder_path = r"/home/zxl/repos/TruckDetection/data/yolov5/打完标/val"  # 标注数据的文件地址
txt_folder_path = r"/home/zxl/repos/TruckDetection/data/yolov5/l"  # 转换后的txt标签文件存放的文件夹


# 保存为相对坐标形式 :label x_center y_center w h
def relative_coordinate_txt(img_name, json_d, img_path):
    try:
        src_img = cv2.imread(img_path)
        cv2.imwrite(img_path, src_img)
        src_img = cv2.imread(img_path)
        h, w = src_img.shape[:2]
    except:
        print(img_path)
        return
    txt_name = img_name.split(".")[0] + ".txt"
    txt_path = os.path.join(txt_folder_path, txt_name)
    # print(txt_path)
    with open(txt_path, 'w') as f:
        for item in json_d["shapes"]:
            # print(item['points'])
            # print(item['label'])
            point = item['points']
            x_center = (point[0][0] + point[1][0]) / 2
            y_center = (point[0][1] + point[1][1]) / 2
            width = math.fabs(point[1][0] - point[0][0])
            hight = math.fabs(point[1][1] - point[0][1])
            item['label'] = 0
            # print(x_center)
            f.write(" {} ".format(item['label']))
            f.write(" {} ".format(x_center / w))
            f.write(" {} ".format(y_center / h))
            f.write(" {} ".format(width / w))
            f.write(" {} ".format(hight / h))
            f.write(" \n")


# 保存为绝对坐标形式 :label x1 y1 x2 y2
def absolute_coordinate_txt(img_name, json_d, img_path):
    src_img = cv2.imread(img_path)
    h, w = src_img.shape[:2]
    txt_name = img_name.split(".")[0] + ".txt"
    txt_path = os.path.join(txt_folder_path, txt_name)
    with open(txt_path, 'w') as f:
        for item in json_d["shapes"]:
            # print(item['points'])
            # print(item['label'])
            point = item['points']
            x1 = point[0][0]
            y1 = point[0][1]
            x2 = point[1][0]
            y2 = point[1][1]
            f.write(" {} ".format(item['label']))
            f.write(" {} ".format(x1))
            f.write(" {} ".format(y1))
            f.write(" {} ".format(x2))
            f.write(" {} ".format(y2))
            f.write(" \n")


if __name__ == '__main__':
    for jsonfile in tqdm.tqdm(os.listdir(folder_path)):
        if ".json" not in jsonfile:
            continue
        temp_path = os.path.join(folder_path, jsonfile)
        # 如果是一个子目录就继续
        if os.path.isdir(temp_path):
            continue
        jsonfile_path = temp_path
        try:
            with open(jsonfile_path, "r", encoding='utf-8') as f:
                json_d = json.load(f)
                # img_name = json_d['imagePath'].split("\\")[-1].split(".")[0] + ".jpg"
                img_name = jsonfile.replace("json", "jpg")
                img_path = os.path.join(img_folder_path, img_name)
                relative_coordinate_txt(img_name, json_d, img_path)
                # absolute_coordinate_txt(img_name, json_d, img_path)
        except Exception as e:
            print(e, '\n', jsonfile_path)
