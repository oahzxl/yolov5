import argparse
import datetime
import logging
import logging.handlers
import os
import sys
import time
import cx_Oracle
import torch
import yaml
from tqdm import tqdm
import shutil

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.datasets import create_dataloader
from utils.general import check_img_size, non_max_suppression, set_logging, colorstr
from utils.torch_utils import select_device

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
REC_NUM = 0
SEL_NUM = 0
SEND_NUM = 0
FAIL_NUM = 0


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # 设置输出文件，时间间隔（s/S:秒 m/M分 h/H时 d/D天 minnight）,时间间隔数目，backupcount最大备份数目，默认为0不删除文件
    allLogHandleTime = logging.handlers.TimedRotatingFileHandler(filename='/home/pxjj/log/debug/debug.log', when='D',
                                                                 interval=1)
    allLogHandleTime.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 设置输出文件，时间间隔（s/S:秒 m/M分 h/H时 d/D天 minnight）,时间间隔数目，backupcount最大备份数目，默认为0不删除文件
    ErrorLogHandleTime = logging.handlers.TimedRotatingFileHandler(filename='/home/pxjj/log/error/error.log', when='D',
                                                                   interval=1)
    # 设置文件输出格式
    ErrorLogHandleTime.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    # 设置文件输出级别
    ErrorLogHandleTime.setLevel(logging.ERROR)
    allLogHandleTime.setLevel(logging.INFO)

    # 添加处理器
    logger.addHandler(allLogHandleTime)
    logger.addHandler(ErrorLogHandleTime)
    return logger


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/collectdata/20210603{jpg',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--data', type=str, default='data/pxjj.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--delete', action='store_true', help='delete origin file')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    return parser.parse_args()


def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    global SEL_NUM

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = mobilenetv3_large()
    #     model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
    #         "/home/pxjj/code/weights/MobileNetV3_Large_epoch_36_acc_0.8578125.tar")['model_state_dict'].items()})
    #     model = model.cuda().eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    outputs = []
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, vid_cap in tqdm(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        SEL_NUM += 1

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            # from pathlib import Path
            # p = Path(p)
            # if 0:
            #     for img in crop_img:
            #         img = transforms.Compose([
            #             transforms.Resize((224,224)),
            #             transforms.ToTensor(),
            #             transforms.Normalize([0.40565532,0.3918059,0.3886595],[0.22155257,0.21012186,0.2013864])
            #         ])(img)
            #         cls_res = modelc(img)
            #         cls_res = nn.functional.softmax(cls_res, dim=-1)[1]
            #         if cls_res > 0.5:
            #             outputs.append(path)
            #             break
            # else:
            if len(det):
                outputs.append(path)

    return outputs


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.1,
         iou_thres=0.6,  # for NMS
         augment=False,
         model=None,
         dataloader=None,
         save_hybrid=False,  # for hybrid auto-labelling
         half_precision=True,
         opt=None):
    global SEL_NUM
    outputs = []
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        # set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)
    # check_dataset(data)  # check
    data['val'] = arg.source
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    for batch_i, (img, targets, paths, shapes) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        SEL_NUM += len(paths)
        with torch.no_grad():
            # Run model
            out = model(img, augment=augment)[0]  # inference and training outputs

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False)

        # Statistics per image
        for si, pred in enumerate(out):
            if len(pred) > 0:
                outputs.append(paths[si])
    return outputs


def move_failed_upload(jpg_name):
    t = datetime.datetime.now()
    global FAIL_NUM
    FAIL_NUM = len(jpg_name)

    dir_name = '/' + str(t).split(':')[0].replace('-', '').replace(' ', '') + '{jpg'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in jpg_name:
        try:
            shutil.move(i, dir_name)
        except:
            continue

    dir_name = dir_name.replace('jpg', 'txt')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in jpg_name:
        i = i.replace('jpg', 'txt')
        try:
            shutil.move(i, dir_name)
        except:
            continue


def upload(result):
    global SEND_NUM
    failed_list = []
    uniview_list = ['KKBM', 'CPHM', 'CPLXBM', 'CPLXMC', 'CPYSMC',
                    'CLLXBM', 'TPZS', 'GCSJ']
    oracle_list = [
        'CJJG', 'CLFL', 'HPZL', 'HPHM', 'JDCSYR', 'SYXZ', 'FDJH', 'CLSBDH', 'CSYS',
        'CLPP', 'JTFS', 'FZJG', 'ZSXZQH', 'ZSXXDZ', 'DH', 'LXFS', 'TZSH', 'TZBJ', 'TZRQ',
        'CJFS', 'WFSJ', 'WFDD', 'WFDZ', 'WFXW', 'FKJE', 'SCZ', 'BZZ', 'ZQMJ', 'LRR', 'LRSJ',
        'JLLX', 'GXSJ', 'ZPSTR1', 'ZPSTR2', 'ZPSTR3', 'ZPSTR4', 'SCBJ', 'SCSJ', 'BZ', 'XZQH', 'SBBH']

    total_data = []
    total_path = []

    for i in range(len(result)):
        try:
            # print("upload中"+result[i].replace("jpg", "txt")+"开始进入utf-8解码")
            with open(result[i].replace("jpg", "txt"), "r", encoding='utf-8') as f:
                txt = f.readlines()[0]
                # print("utf-8解码结束")
        except:
            try:
                # print("upload中"+result[i].replace("jpg", "txt")+"utf-8失败，开始进入gbk解码")
                with open(result[i].replace("jpg", "txt"), "r", encoding='gbk') as f:
                    txt = f.readlines()[0]
                    # print("gbk解码结束")
            except:
                continue
        with open(result[i], 'rb') as f:
            imgdata = f.read()
        # print(txt)
        txt_data = txt.split('_')[:-1]
        if len(txt_data) == 9:
            txt_data[0] = txt_data[0] + txt_data[1]
            del txt_data[1]

        uniview_data = {}
        for j in range(len(uniview_list)):
            if txt_data[j] in ("null", "", "-"):
                uniview_data[uniview_list[j]] = None
            else:
                uniview_data[uniview_list[j]] = txt_data[j]
        # print('uniview data:\n', uniview_data, '\n')

        oracle_data = {}
        # for j in range(len(oracle_list)):
        #     oracle_data[oracle_list[j]] = None
        oracle_data['SBBH'] = uniview_data['KKBM']
        # oracle_data['ZQMJ'] = None
        oracle_data['CLFL'] = ''
        oracle_data['HPZL'] = uniview_data['CPLXBM'] if uniview_data['CPLXBM'] != 'null' else None
        oracle_data['HPHM'] = uniview_data['CPHM']
        oracle_data['ZPSTR1'] = imgdata
        oracle_data['WFSJ'] = datetime.datetime.strptime(uniview_data['GCSJ'], "%Y-%m-%d %H:%M:%S")
        oracle_data['WFXW'] = '1344:机动车违反禁令标志指示的'
        # oracle_data['WFXW'] = None

        # print('oracle data:\n', oracle_data, '\n')

        total_data.append(oracle_data.copy())
        total_path.append(result[i])
        # sql_data = {'SBBH' : oracle_data['SBBH'],
        #             'CLFL' : oracle_data['CLFL'],
        #             'HPZL' : oracle_data['HPZL'],
        #             'HPHM' : oracle_data['HPHM'],
        #             'ZPSTR1' : oracle_data['ZPSTR1'],
        #             'WFSJ' : oracle_data['WFSJ']
        # }
        # total_data.append(sql_data.copy)

    if len(total_data):
        # total_data[0]['CLFL'] = 'imgdata'
        keys = 'surveil_sequence.NEXTVAL,'
        # keys = ''
        for k in oracle_data.keys():
            keys = keys + ':' + k + ','
        SEND_NUM = len(total_data)

        while len(total_data) > 50:
            tmp_data = total_data[:50]
            total_data = total_data[50:]
            tmp_path = total_path[:50]
            total_path = total_path[50:]

            # 执行SQL,创建一个表
            # sql = "insert into VIO_SURVEIL values(" + keys[:-1] + ")"
            sql = "insert into VIO_SURVEIL(XH,SBBH,CLFL,HPZL,HPHM,ZPSTR1,WFSJ,WFXW) values(surveil_sequence.NEXTVAL,:SBBH,:CLFL,:HPZL,:HPHM,:ZPSTR1,:WFSJ,:WFXW)"
            manyflag = True
            manytimes = 1
            while (manyflag):
                try:
                    conn = cx_Oracle.connect("angyi_tpmspx", "angyi_tpmspx", "172.31.7.243:1521/pxtxzdb")
                    cursor = conn.cursor()
                    cursor.executemany(sql, tmp_data)

                    cursor.close()
                    conn.commit()
                    conn.close()
                    manyflag = False
                except Exception as e:
                    try:
                        cursor.close()
                        conn.rollback()
                        conn.close()
                    except:
                        pass
                    if manytimes > 5:
                        manyflag = False
                        logger.error(e)
                        logger.warning("EXCUTEMANY FAILED ! start insert one by one")
                        for idx, data in enumerate(tmp_data):
                            flag = True
                            count = 1
                            while flag:
                                try:
                                    singleconn = cx_Oracle.connect("angyi_tpmspx", "angyi_tpmspx",
                                                                   "172.31.7.243:1521/pxtxzdb")
                                    singlecursor = singleconn.cursor()
                                    singlecursor.execute(sql, data)

                                    singlecursor.close()
                                    singleconn.commit()
                                    singleconn.close()
                                    flag = False
                                except Exception as e1:
                                    try:
                                        singlecursor.close()
                                        singleconn.rollback()
                                        singleconn.close()
                                    except:
                                        pass
                                    if 'ORA-03135' in str(e1):
                                        logger.warning("insert one times:" + str(count))
                                        if count > 6:
                                            flag = False
                                            logger.error(str(e1))
                                        count = count + 1
                                    else:
                                        flag = False
                                        SEND_NUM = SEND_NUM - 1
                                        failed_list.append(tmp_path[idx])
                                        logger.error(str(e1) + "....skip this data")
                    else:
                        logger.warning("INSERTMANY FALIED! insert times:" + str(manytimes))
                        manytimes = manytimes + 1
                # finally:
                #     try:
                #         if cursor is not None:
                #             cursor.close()
                #     except Exception as e:
                #         logger.error("CLOSE CURSOR ERROR:"+str(e))
                #     try:
                #         if conn is not None:
                #             conn.close()
                #     except Exception as e:
                #         logger.error("CLOSE CONN ERROR:"+str(e))

        # 执行SQL,创建一个表
        # sql = "insert into VIO_SURVEIL values(" + keys[:-1] + ")"
        sql = "insert into VIO_SURVEIL(XH,SBBH,CLFL,HPZL,HPHM,ZPSTR1,WFSJ,WFXW) values(surveil_sequence.NEXTVAL,:SBBH,:CLFL,:HPZL,:HPHM,:ZPSTR1,:WFSJ,:WFXW)"
        manyflag2 = True
        manytimes2 = 1
        while (manyflag2):
            try:
                conn = cx_Oracle.connect("angyi_tpmspx", "angyi_tpmspx", "172.31.7.243:1521/pxtxzdb")
                cursor = conn.cursor()
                cursor.executemany(sql, total_data)

                cursor.close()
                conn.commit()
                conn.close()
                manyflag2 = False
            except Exception as e:
                try:
                    cursor.close()
                    conn.rollback()
                    conn.close()
                except:
                    pass
                if manytimes2 > 5:
                    manyflag2 = False
                    logger.error(e)
                    logger.warning("EXCUTEMANY FAILED ! start insert one by one")
                    for idx, data in enumerate(total_data):
                        flag = True
                        count = 1
                        while flag:
                            try:
                                singleconn = cx_Oracle.connect("angyi_tpmspx", "angyi_tpmspx",
                                                               "172.31.7.243:1521/pxtxzdb")
                                singlecursor = singleconn.cursor()
                                singlecursor.execute(sql, data)

                                singlecursor.close()
                                singleconn.commit()
                                singleconn.close()
                                flag = False
                            except Exception as e1:
                                try:
                                    singlecursor.close()
                                    singleconn.rollback()
                                    singleconn.close()
                                except:
                                    pass
                                if 'ORA-03135' in str(e1):
                                    singlecursor.close()
                                    singlecursor.rollback()
                                    logger.warning("insert one times:" + str(count))
                                    if count > 6:
                                        flag = False
                                        logger.error(str(e1))
                                    count = count + 1
                                else:
                                    flag = False
                                    SEND_NUM = SEND_NUM - 1
                                    failed_list.append(total_path[idx])
                                    logger.error(str(e1) + "....skip this data")
                else:
                    logger.warning("INSERTMANYLAST FALIED! insert times:" + str(manytimes2))
                    manytimes2 = manytimes2 + 1
            # finally:
            #     try:
            #         if cursor is not None:
            #             cursor.close()
            #     except Exception as e:
            #         logger.error("CLOSE CURSOR ERROR:"+str(e))
            #     try:
            #         if conn is not None:
            #             conn.close()
            #     except Exception as e:
            #         logger.error("CLOSE CONN ERROR:"+str(e))
    move_failed_upload(failed_list)


def delete_unnecessary_files(opt):
    source = opt.source.replace("jpg", "txt")
    # pdb.set_trace()
    global REC_NUM
    REC_NUM = len(os.listdir(source))
    if REC_NUM:
        for file in tqdm(sorted(os.listdir(source))):
            txt_file_name = os.path.join(source, file)
            try:
                pic_txt = file[:-4].split('_')
                if (pic_txt[-7] != "01" and pic_txt[-2] != "C") or not os.path.exists(
                        txt_file_name.replace("txt", "jpg")):
                    os.remove(txt_file_name)
                    os.remove(txt_file_name.replace("txt", "jpg"))
            except:
                print(txt_file_name)
                try:
                    os.remove(txt_file_name)
                    os.remove(txt_file_name.replace("txt", "jpg"))
                except:
                    pass
    else:
        logger.info('Skip - Processing Dir %s - Received Pictures 0' %
                    (arg.source[1:-4]))
        sys.exit(0)


if __name__ == '__main__':
    arg = init_args()
    logger = init_logger()
    t0 = time.time()
    try:
        delete_unnecessary_files(arg)
        delete_time = time.time() - t0
        t0 = time.time()
    except Exception as e:
        logger.error('Delete Error - Processing Dir %s - Received Pictures %d' %
                     (arg.source[1:-4], REC_NUM))
        logger.error(str(e))
        sys.exit(0)
    try:
        results = test(
            arg.data, arg.weights, arg.batch_size,
            arg.img_size, arg.conf_thres, arg.iou_thres, opt=arg)
        # results = detect(arg)
    except:
        try:
            results = test(
                arg.data, arg.weights, arg.batch_size,
                arg.img_size, arg.conf_thres, arg.iou_thres, opt=arg)
            # results = detect(arg)
        except Exception as e:
            logger.error('Python Error - Processing Dir %s - Received Pictures %d - '
                         'Selected Pictures %d - Delete Time %ds' %
                         (arg.source[1:-4], REC_NUM, SEL_NUM,
                          delete_time))
            logger.error(str(e))
            sys.exit(0)
    python_time = time.time() - t0
    t0 = time.time()
    try:
        upload(results)
        upload_time = time.time() - t0
        t0 = time.time()
        logger.info('Run Successfully - Processing Dir %s - Received Pictures %d - '
                    'Selected Pictures %d - Detected Trucks %d - Inserted Trucks %d - Failed Trucks %d - '
                    'Delete Time %ds - Python Time %ds - Upload Time %ds' %
                    (arg.source[1:-4], REC_NUM, SEL_NUM, len(results), SEND_NUM, FAIL_NUM,
                     delete_time, python_time, upload_time))
    except Exception as e:
        logger.error('Upload Error - Processing Dir %s - Received Pictures %d - '
                     'Selected Pictures %d - Detected Trucks %d - '
                     'Delete Time %ds - Python Time %ds' %
                     (arg.source[1:-4], REC_NUM, SEL_NUM, len(results),
                      delete_time, python_time))
        logger.error(str(e))
        sys.exit(0)
