import os
from PIL import Image
from tqdm.std import tqdm
import random

THRESHOLD = 0.4
RandomBox = 3
task = 'val'
BoxLabelPath = os.path.join('/home/zxl/repos/TruckDetection/data/yolov5/crop/label', task)
BoxPredictPath = os.path.join('/home/zxl/repos/TruckDetection/data/yolov5/crop/predict', task)
ImgPath = os.path.join('/home/zxl/repos/TruckDetection/data/yolov5/images', task)
SavePath = os.path.join('/home/zxl/repos/TruckDetection/data/yolov5/crop/cropped', task)


def merge():
    files1 = os.listdir(BoxPredictPath)
    files2 = os.listdir(BoxLabelPath)
    files = list(set(files1).union(files2))  # 求并集

    contents = []

    for file in tqdm(files):
        tmp = []
        try:
            with open(os.path.join(BoxLabelPath, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    content = line.strip('\n')

                    if content == '':
                        continue

                    content = content.split(' ')
                    while '' in content:
                        content.remove('')
                    for i in range(1, len(content)):
                        content[i] = float(content[i])
                    content[0] = 1
                    tmp.append(content)
        except:
            pass
        try:
            with open(os.path.join(BoxPredictPath, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    content = line.strip(']\n')
                    content = content.replace('[', '')

                    if content == '':
                        continue

                    content = content.split(' ')
                    for i in range(len(content)):
                        if i == len(content) - 1 or i == 0:
                            content[i] = float(content[i])
                        else:
                            content[i] = float(content[i][:-1])
                    tmp.append(content)
        except:
            pass
        contents.append(tmp)
    return files.copy(), contents.copy()


def cut_img(path, pos, path_save):
    # path : picture path
    # pos : [x1, y1, x2, y2] normalized
    # path_save : picture save path
    try:
        img = Image.open(path)
    except:
        print("ERROR: ", path)
        return
    x1 = min([pos[0], pos[2]])
    y1 = min([pos[1], pos[3]])
    x2 = max([pos[0], pos[2]])
    y2 = max([pos[1], pos[3]])
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    cropped = img.crop((x1, y1, x2, y2))  # (left, upper, right, lower)
    try:
        cropped.save(path_save)
    except:
        print(111)


def crop(files, contents):
    for file, content in tqdm(zip(files, contents), total=len(files)):
        i = 0

        for box in content:
            conf = float(box[0])
            savePath = os.path.join(SavePath, '1' if conf >= THRESHOLD else '0')
            savePath = savePath + '/' + file[:-4] + "_" + str(i) + '.jpg'
            i += 1

            x1, y1, x2, y2 = box[1:]

            pos = [x1, y1, x2, y2]
            cut_img(os.path.join(ImgPath, file.replace(".txt", ".jpg")), pos, savePath)
        random_boxes = random_box(os.path.join(ImgPath, file.replace(".txt", ".jpg")), content)
        for b in random_boxes:
            i += 1
            savePath = os.path.join(SavePath, '0') + '/' + file[:-4] + "_" + str(i) + '.jpg'
            try:
                cut_img(os.path.join(ImgPath, file.replace(".txt", ".jpg")), b, savePath)
            except Exception as e:
                print(e)


def random_box(path, box):
    img = Image.open(path)
    iw = img.width
    ih = img.height
    outputs = []
    n = 0
    max_try = 50

    while len(outputs) < RandomBox:
        if n > max_try:
            break
        flag = 1
        while flag:
            if n > max_try:
                break
            n += 1
            flag = 0
            x = (random.random() - 0.5) * iw + iw / 2
            y = (random.random() - 0.5) * ih + ih / 2
            for b in box:
                b_x1 = min(b[1], b[3])
                b_x2 = max(b[1], b[3])
                b_y1 = min(b[2], b[4])
                b_y2 = max(b[2], b[4])
                if b_x1 < x < b_x2 or b_x1 < x < b_x2:
                    flag = 1
                    break
                if b_y1 < y < b_y2 or b_y1 < y < b_y2:
                    flag = 1
                    break
        flag = 1
        while flag:
            if n > max_try:
                break
            n += 1
            flag = 0
            w = (random.random() * 0.05 + 0.1) * iw
            h = (random.random() * 0.8 + 0.6) * w
            x1 = max(x - w, 0)
            x2 = min(x + w, iw)
            y1 = max(y - h, 0)
            y2 = min(y + h, ih)
            for b in box:
                b_x1 = min(b[1], b[3])
                b_x2 = max(b[1], b[3])
                b_y1 = min(b[2], b[4])
                b_y2 = max(b[2], b[4])
                if b_x1 < x1 < b_x2 or b_x1 < x2 < b_x2:
                    flag = 1
                    break
                if b_y1 < y1 < b_y2 or b_y1 < y2 < b_y2:
                    flag = 1
                    break
        if n < max_try:
            outputs.append([x1, y1, x2, y2])
    return outputs


if __name__ == '__main__':
    file_name, boxes = merge()
    crop(file_name, boxes)
