
import math 
from cProfile import label
import os
import cv2
import glob
import shutil
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from natsort import natsorted
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def xyxy2xywh(width, height, xmin, xmax, ymin, ymax):
    # print(width, height, xmin, xmax, ymin, ymax)
    xcen = float((xmin + xmax)) / 2.0 / width
    ycen = float((ymin + ymax)) / 2.0 / height
    w = float(xmax - xmin)* 1.0  / width
    h = float(ymax - ymin)* 1.0  / height
    
    return str(xcen), str(ycen), str(w), str(h)

def xywh2xyxy(width, height, xcen, ycen, w, h):
    xmin = (xcen - w/2) * width
    xmax = (xcen + w/2) * width
    ymin = (ycen - h/2) * height
    ymax = (ycen + h/2) * height
    
    return int(xmin), int(xmax), int(ymin), int(ymax)

def xml2txt():
    
    datasets = ['20230620/img/back/52', '20230620/img/back/53']
    #'video_002','video_001','video_003','video_004','video_005','video_006','video_007','video_008','video_009'
    #'video_010','video_011','video_012','video_013'
    #'WPI','South Korea/korea384','South Korea/korea400','Lara'
    sequences = {'Chiayi_To_Tainan/day':2 } #'Sequence1':7,'Sequence2':11,'PublicDataset':15
    cont = 0
    for sequence in sequences:
        for dataset in range(cont, sequences[sequence]):
            xml_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/Annotations'
            out_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/labels_3cls_5pixel'
            # dif_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+dataset+'/labels_6cls_dif'

            all_classes = ['Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right'] # 'Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right'
            classes = ['Red', 'Yellow', 'Green'] #'Traffic Light' , 'Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right'

            # if os.path.isdir(out_path):
            #     shutil.rmtree(out_path)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            # if not os.path.isdir(dif_path):
            #     os.mkdir(dif_path)
            xmls = os.listdir(xml_path)
            xmls.sort()

            for xml in xmls:
                txt = os.path.splitext(xml)[0]+'.txt'
                out_file = open(out_path+os.sep+txt,"w")
                # dif = open(dif_path+os.sep+txt,"w")
                tree = ET.parse(xml_path+os.sep+xml)
                root = tree.getroot()
                width = root.find('size').find('width').text
                height = root.find('size').find('height').text
                # print(anno)
                for obj in root.findall('object'):

                    if obj.find('difficult') != None:
                        difficult = int(obj.find('difficult').text)
                    else:
                        difficult = 0

                    cls = obj.find('name').text
                    if cls not in all_classes :
                        continue
                    if cls in classes:
                            cls_id = classes.index(cls)
                        # if classes.index(cls) < 3:
                        #     cls_id = classes.index(cls)
                        # elif classes.index(cls) == 3:
                        #     continue
                        # elif classes.index(cls) == 4:
                        #     continue
                        # elif classes.index(cls) == 5:
                            # cls_id = 2
                    else:
                        cls_id = int(2)
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    
                    x, y, w, h = xyxy2xywh(int(width), int(height),
                        int(xmin), int(xmax), int(ymin), int(ymax))
                    # if difficult :
                        # dif.write(str(cls_id) + " " + x + " " + y + " " + w + " " + h + '\n')
                    # else:
                    if xmax-xmin > 5 or ymax-ymin > 5:
                        out_file.write(str(cls_id) + " " + x + " " + y + " " + w + " " + h + '\n')
                out_file.close()
                # dif.close()
            print(datasets[dataset]+" is finsh!")
            cont += 1

def data_rename():

    datasets = ['video_003','video_004','video_005','video_006','video_007','video_008','video_009',
    'video_010','video_011','video_012','video_013',
    'WPI','South Korea/korea384','South Korea/korea400','Lara']
    sequences = {'Sequence1':7, 'Sequence2':11, 'PublicDataset':15}

    cont = 0

    for sequence in sequences:
        for dataset in range(cont, sequences[sequence]):
            # img_type = 'full' #full
            txt_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/labels_3cls'
            save_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/augment1/labels_3cls'
            shutil.copytree(txt_path,save_path)
            
            txt_list = os.listdir(save_path)
            for txt in txt_list:
                os.rename(os.path.join(save_path, txt), os.path.join(save_path, '1_'+txt))
            print('finsh ',datasets[dataset])
            cont += 1
    
def txt_class_change():

    # 6class = 'Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right'
    txt_dir = '/home/rvl/Desktop/TrafficLight/newday/test/day_4/labels_sets/labels_6cls' # /test/video_004/labels
    txt_list = glob.glob(txt_dir + "/*.txt") 
    sort_txt = natsorted(txt_list)

    clsnums = {'labels_3cls':3} #, 'labels_4cls':4
    # save_path = '/home/rvl/Desktop/TrafficLight/newday/labels_set/labels_4class' #labels_3class, labels_4class
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    for clsnum in clsnums:
        save_path = '/home/rvl/Desktop/TrafficLight/newday/test/day_4/labels_sets/'+clsnum
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for txt in sort_txt:
            ori_f = open(txt, 'r')
            save_f = open(os.path.join(save_path,txt.split('/')[-1]),'w')
            if clsnums[clsnum] == 3:
                for obj in ori_f: # 6to3
                    save_line = obj
                    if int(obj[0])>2 :
                        save_line = str(2)+obj[1:]
                    save_f.write(save_line)
            elif clsnums[clsnum] == 4: # 6to4
                for obj in ori_f:
                    if int(obj[0])>1 :
                        save_line = str(int(obj[0])-2)+obj[1:]
                        save_f.write(save_line)
            save_f.close()
    # for idx, txt in enumerate(sort_txt):
    #     ori_f = open(txt, 'r')
    #     # save_f = open(os.path.join(save_path,txt[0]),'w')
    #     save_f = open(os.path.join(save_path,txt.split('/')[-1]),'w')

    #     #6to3
    #     for obj in ori_f:
    #         save_line = obj
    #         if int(obj[0])>2 :
    #             save_line = str(2)+obj[1:]
    #         save_f.write(save_line)
    #     save_f.close()  

    #     #6to4
    #     for obj in ori_f:
    #         if int(obj[0])>1 :
    #             save_line = str(int(obj[0])-2)+obj[1:]
    #             save_f.write(save_line)
    #     save_f.close()  
    print("finsh!")
    
def xml_class_change():

    name = 'South Korea/korea400'
    xml_path = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/'+name+'/Annotations'
    labels_path = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/'+name+'/labels_sets'
    out_path = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/'+name+'/Annotations_sets'
    out_path_3 = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/'+name+'/Annotations_sets/Annotations_3cls'
    out_path_ori = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/'+name+'/Annotations_sets/Annotations_ori'

    all_classes = ['Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right']
    classes = ['Left', 'Straight', 'Right'] 

    if not os.path.isdir(labels_path):
        os.mkdir(labels_path)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if not os.path.isdir(out_path_3):
        os.mkdir(out_path_3)
    if not os.path.isdir(out_path_ori):
        os.mkdir(out_path_ori)
    xml_list = os.listdir(xml_path)
    xml_list = natsorted(xml_list)
    pixel_th = 0

    for xml in xml_list:

        shutil.copyfile(os.path.join(xml_path,xml),os.path.join(out_path_3,xml))
        shutil.copyfile(os.path.join(xml_path,xml),os.path.join(out_path_ori,xml))
        
        tree = ET.parse(out_path_3+os.sep+xml)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            cls_root = obj.find('name')
            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            if ymax-ymin>pixel_th or xmax-xmin>pixel_th:
                if cls in classes :
                    cls_root.text = 'Green'
                elif cls in all_classes:
                    continue
                else:
                    print(cls)
                    root.remove(obj)
            else:
                print(ymax-ymin,xmax-xmin)
                root.remove(obj)
            
            
        tree.write(out_path_3+os.sep+xml)
    print("finsh!")

def write_trainval_txt():

    #train_val
    # datasets = ['20221115', '20230620/img/go/44']
    # sequences = {'Chiayi_To_Tainan/day':2 } 
    # save_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day' # day
    # train_txt = open(save_path+'/train.txt','w') # train_crop
    # val_txt = open(save_path+'/val.txt','w') # val_crop

    # cont, train, val = 0, 0, 0
    # for sequence in sequences:
    #     for dataset in range(cont, sequences[sequence]):
    #         xml_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/Annotations'

    #         xml_list = glob.glob(xml_path + "/*.xml") 

    #         img_train, img_val = train_test_split(xml_list, test_size=0.2)

    #         for xml_path in img_train:
    #             train_txt.write(xml_path.replace('Annotations', 'JEPGImages').replace('.xml', '.jpg')+'\n')
    #         for xml_path in img_val:
    #             val_txt.write(xml_path.replace('Annotations', 'JEPGImages').replace('.xml', '.jpg')+'\n')
    #         print(Path(datasets[dataset]).stem,len(img_train),len(img_val))
    #         train += len(img_train)
    #         val += len(img_val)
    #         cont += 1
    #     print('total:',train, val)
    # train_txt.close()
    # val_txt.close()
    
    #test
    img_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/20230620/img/back/53/JPEGImages'
    save_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day'

    img_list = glob.glob(img_path + "/*.png") 
    test_txt = open(save_path+'/test1.txt','w')

    for img_path in img_list:
        test_txt.write(img_path+'\n')
    test_txt.close()

    print("finsh!")

def cal_label_num():

    # txt_dir = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/'

    # datasets = ['train', 'val']#'train','val',
    # for dataset in datasets :
        
    #     dataset_txt = open(txt_dir+dataset+'.txt', 'r')
    #     cls = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0 }

    #     for txts in (dataset_txt):
    #         txt = open(txts.replace('JEPGImages','labels_3cls').replace('jpg', 'txt')[:-1], 'r')
    #         for label in (txt):
    #             cls[label[0]] += 1
    #     print(str(dataset),cls)


    ## train-data
    # datasets = ['video_002','video_003','video_005','video_006','video_007','video_008','video_009', # 7
    #             'video_010','video_011','video_012','video_013', # 4
    #             'WPI','South Korea/korea384','South Korea/korea400','Lara'] # 4
    # sequences = {'Sequence1':7, 'Sequence2':11, 'PublicDataset':15}
    # cont = 0
    # for sequence in sequences:
    #     for dataset in range(cont, sequences[sequence]):
    #         txt_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/labels'
    #         txt_lsit = os.listdir(txt_path)

    #         # cls = {'Red':0, 'Yellow':0, 'Green':0, 'Left':0, 'Straight':0, 'Right':0, 'Traffic Light':0}
    #         cls = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0 }
    #         for data in txt_lsit:
    #             txt = open(txt_path+'/'+data, 'r')
    #             for label in (txt):
    #                 cls[label[0]] += 1
    #             # tree = ET.parse(txt_path+'/'+data)
    #             # root = tree.getroot()
    #             # # print(data)
    #             # for obj in root.findall('object'):
    #             #     classes = obj.find('name').text
    #             #     cls[classes] += 1
    #         cont += 1
    #         print(str(dataset),cls)
    
    txt_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/20230620/img/back/53/Annotations'
    txt_lsit = os.listdir(txt_path)

    cls = {'Red':0, 'Yellow':0, 'Green':0, 'Left':0, 'Straight':0, 'Right':0, 'Traffic Light':0}
    # cls = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0 }
    for data in txt_lsit:
        # txt = open(txt_path+'/'+data, 'r')
        # for label in (txt):
        #     cls[label[0]] += 1
        tree = ET.parse(txt_path+'/'+data)
        root = tree.getroot()
        # print(data)
        for obj in root.findall('object'):
            classes = obj.find('name').text
            cls[classes] += 1
    print(cls)

def crop_img():

    txt_path = '/home/rvl/Desktop/yolov7/runs/test/3cls_day4_model123/FP/day_4'
    img_path = '/home/rvl/Desktop/TrafficLight/newday/test/day_4/images'

    outimg_path = '/home/rvl/Desktop/yolov7/runs/test/3cls_day4_model123/crop_images/images'
    outtxt_path = '/home/rvl/Desktop/yolov7/runs/test/3cls_day4_model123/crop_images/labels'

    if not os.path.exists(outimg_path):
        os.makedirs(outimg_path)
    if not os.path.exists(outtxt_path):
        os.makedirs(outtxt_path)

    txt_list = os.listdir(txt_path)
    txt_list = natsorted(txt_list)
    clses = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0, '6':0 }
    for i in clses:
        if not os.path.exists(outimg_path + '/../' + i):
            os.makedirs(outimg_path + '/../' + i)
    for txt in txt_list:
        objs = open(os.path.join(txt_path,txt),'r')
        
        img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
        H, W, _ = img.shape
        # print(size)
        num = 0
        for idx, obj in enumerate(objs):
            name = txt[:-4]+'_'+str(num)
            while os.path.exists(os.path.join(outimg_path,name+'.jpg')):
                num += 1
                name = txt[:-4]+'_'+str(num)
            # cls, x, y, w, h = obj.split(' ')
            cls, xmin, ymin, xmax, ymax, _ = obj.split(' ')
            # cls, xmin, ymin, xmax, ymax = obj.split(' ')
            cls, xmin, ymin, xmax, ymax = int(float(cls)), int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
            # xmin, xmax, ymin, ymax = xywh2xyxy(size[1], size[0], float(x), float(y), float(w), float(h))
            # print(xmin, xmax, ymin, ymax)
            if ymax-ymin < 5 or xmax-xmin < 5:
                continue
            else:
                size = 2 if ymax-ymin < 20 or xmax-xmin < 20 else 5
                left = xmin if xmin-size<0 else size
                right = W-1-xmax if xmax+size>W-1 else size
                top = ymin if ymin-size<0 else size
                bottom = H-1-ymax if ymax+size>H-1 else size
                crop_img = img[ymin-top:ymax+bottom,xmin-left:xmax+right]
                # cv2.imwrite(os.path.join(outimg_path,name+'.jpg'),crop_img)
                cv2.imwrite(os.path.join(outimg_path,'..',str(cls),name+'.jpg'),crop_img)

                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [255,0,0], 2)

                # cv2.imwrite(os.path.join(outimg_path,'..','draw_6',name+'.jpg'),img)

                
                save_txt = open(os.path.join(outtxt_path,name+'.txt'),'w')
                h, w, _ = crop_img.shape
                # print(w, h, float(1), float(1), float(w-1), float(h-1))
                
                xcen, ycen, w, h = xyxy2xywh(w, h, left, float(w-right), float(top), float(h-bottom))
                save_txt.write(str(cls) + " " + str(xcen) + " " + str(ycen) + " " + str(w) + " " + str(h) + '\n')
                save_txt.close()
                # print(xcen, ycen, w, h)
                # a = input()

        objs.close()
        
    
    print('finsh!')
            
def layoff_empty():
    ## train-data
    datasets = ['video_002','video_003','video_005','video_006','video_007','video_008','video_009', # 7
                'video_010','video_011','video_012','video_013', # 4
                'Lara','WPI','South Korea/korea384','South Korea/korea400'] # 4
                # 'video_002','video_003','video_005','video_006','video_007','video_008','video_009', # 7
                # 'video_010','video_011','video_012','video_013', # 4
                # 'WPI','South Korea/korea384','South Korea/korea400','Lara'

    sequences = {'Sequence1':7, 'Sequence2':11, 'PublicDataset':15}
                # 'Sequence1':7, 'Sequence2':11, 'PublicDataset':15
    ## test-data
    # datasets = ['video_001','video_004']
    # sequences = {'Sequence1':2}
    temp = 1
    cont = 0
    for sequence in sequences:
        for dataset in range(cont, sequences[sequence]):
            txt_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/labels'
            img_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/JPEGImages'
            out_path = '/home/rvl/Desktop/TrafficLight/allmix/images'
            
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            if not os.path.isdir(out_path.replace('images','labels')):
                os.mkdir(out_path.replace('images','labels'))

            txt_lsit = os.listdir(txt_path)
            img_list = os.listdir(img_path)

            for txt in txt_lsit:
                if os.stat(os.path.join(txt_path,txt)).st_size == 0: # an empty file
                    # print(txt)
                    continue
                else:
                    # shutil.copyfile(os.path.join(txt_path,txt),os.path.join(out_path.replace('images','labels'),str(temp)+'.txt'))
                    for img in img_list:
                        if txt[:-3] == img[:-3]:
                            shutil.copyfile(os.path.join(txt_path,txt),os.path.join(out_path.replace('images','labels'),str(temp)+'.txt'))
                            shutil.copyfile(os.path.join(img_path,img),os.path.join(out_path,str(temp)+'.jpg'))
                    temp += 1
            print(datasets[dataset]+" is finsh!")
            cont += 1

def draw_bboximg():


    txt_path = '/home/rvl/Desktop/yolov7/runs/test/3cls_video002/FN/video_002'
    # gt_path = '/home/rvl/Desktop/yolov7/runs/test/newday1_yolo_model123_notrace_6cls_video002_nored_nocls_th5_iou4_1280_nodif/crop_images/labels_resize'
    # img_path = '/home/rvl/Desktop/TrafficLight/dataset/PublicDataset/LISA/daySequence1/JPEGImages'
    img_path = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/images'
    out_path = '/home/rvl/Desktop/yolov7/runs/test/3cls_video002/draw_preds'

    # classes = ['Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right', 'Background']
    all = {'Red':0, 'Yellow':0, 'Green':0, 'Left':0, 'Straight':0, 'Right':0, 'Background':0}#
    # correct = {'Red':0, 'Yellow':0, 'Green':0, 'Left':0, 'Straight':0, 'Right':0, 'Background':0}
    # color = [(0,0,255), (67,211,255), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,0,0)]
    
    
    # txt_lsit = os.listdir(txt_path)
    # txt_lsit = natsorted(txt_lsit)

    img_list = os.listdir(img_path)
    img_list = natsorted(img_list)

    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    # if not os.path.isdir(out_path+'/JPEGImages'):
    #     os.mkdir(out_path+'/JPEGImages')
    # if not os.path.isdir(out_path+'/labels'):
    #     os.mkdir(out_path+'/labels')
    # for cls in classes:
    #     path = os.path.join(out_path,cls)
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # out = cv2.VideoWriter('/home/rvl/Desktop/yolov7/runs/test/newday1_yolo_augment157_notrace_3cls_lisa_nocls_th5_iou4_1280_nodif/video_002.mp4', fourcc, 5, (2048,1536))
    # shutil.copytree(img_path,out_path)
    num = 0


    # for txt in txt_lsit:
    #     # shutil.copyfile(os.path.join(img_path,txt.replace('.txt','.jpg')),os.path.join(out_path,txt.replace('.txt','.jpg')))
    #     # if os.path.exists(os.path.join(txt_path,img.replace('.jpg','.txt')))== 0 or os.stat(os.path.join(txt_path,img.replace('.jpg','.txt'))).st_size == 0: 
    #     #     continue
    #     # else:
    #     ori_img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
    #     # print(os.path.join(img_path,txt.replace('.txt','.jpg')))
    #     height, width, _  = ori_img.shape
    #     objs = open(os.path.join(txt_path,txt), 'r')
    #     gt = open(os.path.join(gt_path,txt), 'r')
    #     for gt_obj in (gt):
    #         gt_cls = gt_obj[0]
    #     for obj in (objs):
    #         # write_txt = open(os.path.join(out_path,'labels',str(num)+'.txt'),'w')
    #         # write_txt.write(obj)
    #         cls, x, y, w, h, conf= obj.split(" ")
    #         # cls, xmin, ymin, xmax, ymax, _ = obj.split(' ')
    #         # cls, xmin, ymin, xmax, ymax = int(float(cls)), int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
    #         if float(conf) > 0.0:
    #             xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
    #         # cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), color[int(cls)], 2)
    #         cv2.putText(ori_img, classes[int(cls)], (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color[int(cls)], 1)
    #         # xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
    #         # crop_img = ori_img[ymin:ymax,xmin:xmax]
    #         # cv2.imwrite(os.path.join(out_path,'JPEGImages',str(num)+'.jpg'),crop_img)
    #         all[classes[int(gt_cls)]] += 1
    #         if gt_cls == cls:
    #             correct[classes[int(gt_cls)]] += 1
    #         num += 1
    #     gt.close()
    #     objs.close()
    #     # cv2.imshow('img',ori_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     cv2.imwrite(os.path.join(out_path,classes[int(gt_cls)],txt.replace('.txt','.jpg')), ori_img)
    #     # out.write(ori_img)
    
    for img in img_list:
        ori_img = cv2.imread(os.path.join(img_path,img))
        height, width, _  = ori_img.shape
        
        if os.path.exists(os.path.join(txt_path,img.replace('.jpg','.txt'))):
            FNs = open(os.path.join(txt_path,img.replace('.jpg','.txt')), 'r')
            for FN in FNs:
                cls, xmin, ymin, xmax, ymax = FN.split(' ')
                cls, xmin, ymin, xmax, ymax = int(float(cls)), int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
                cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            FNs.close()
                
        if os.path.exists(os.path.join(txt_path.replace('FN','TP'),img.replace('.jpg','.txt'))):
            TPs = open(os.path.join(txt_path.replace('FN','TP'),img.replace('.jpg','.txt')), 'r')
            for TP in TPs:
                cls, xmin, ymin, xmax, ymax = TP.split(' ')
                cls, xmin, ymin, xmax, ymax = int(float(cls)), int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
                cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            TPs.close()
            
        cv2.imwrite(os.path.join(out_path,img), ori_img)

def comp_ori_label():

    img_path = '/home/rvl/Desktop/TrafficLight/newday/images'
    GT_path = '/home/rvl/Desktop/TrafficLight/newday/test/day_5/labels_sets/labels_3class'
    detc_path = '/home/rvl/Desktop/yolov7/runs/test/exp2_3class_300_day5_relabel/labels'
    out_path = '/home/rvl/Desktop/yolov7/runs/test/exp2_3class_300_day5_relabel'


    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    img_lsit = os.listdir(img_path)
    img_lsit = natsorted(img_lsit)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path+'/comp_day5.avi', fourcc, 20, (1080,1920)) #2048, 1536

    if os.path.isdir(out_path+'/compGT'):
        shutil.rmtree(out_path+'/compGT')
    shutil.copytree(img_path,out_path+'/compGT')

    for img in img_lsit:
        pic = cv2.imread(os.path.join(out_path+'/compGT',img))
        height, width, _  = pic.shape
        if os.path.exists(os.path.join(GT_path,img.replace('.jpg','.txt'))):
            GT_objs = open(os.path.join(GT_path,img.replace('.jpg','.txt')), 'r')
            for obj in (GT_objs):
                obj= obj.split(" ")
                cls, x, y, w, h  = obj[0], obj[1], obj[2], obj[3], obj[4]
                xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
                cv2.rectangle(pic, (xmin-3, ymin-3), (xmax+3, ymax+3), (0, 0, 255), 2)
            GT_objs.close()
        if os.path.exists(os.path.join(detc_path,img.replace('.jpg','.txt'))):
            detec_objs = open(os.path.join(detc_path,img.replace('.jpg','.txt')), 'r')
            for obj in (detec_objs):
                obj= obj.split(" ")
                cls, x, y, w, h  = obj[0], obj[1], obj[2], obj[3], obj[4]
                xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
                cv2.rectangle(pic, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            detec_objs.close()
        # cv2.imshow('img',pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(out_path+'/compGT',img), pic)
        out.write(pic)
    
    out.release()
    print('finsh!')

def txt_xywh2xyxy():

    txt_dir = '/home/rvl/Desktop/TrafficLight/day/test/video_001/labels'
    txt_list = glob.glob(txt_dir + "/*.txt")  
    sort_txt = natsorted(txt_list)
    width, height = 2048, 1536

    save_path = txt_dir.replace('labels','labels_xyxy')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, txt in enumerate(sort_txt):
        ori_f = open(txt, 'r')
        save_f = open(os.path.join(save_path,txt[-10:]),'w')
        
        for obj in ori_f:
            obj= obj.split(" ")
            cls, x, y, w, h  = obj[0], obj[1], obj[2], obj[3], obj[4]
            xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
            save_f.write(str(cls) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")

        save_f.close() 
        ori_f.close() 

def cal_det_light_size():

    txt_path = '/home/rvl/Desktop/TrafficLight/day/test/video_004/labels_sets/relabel/labels_3class_pixel5'

    # Red = {'0-5':0 , '6-10':0 , '11-15':0, '16-20':0, '21-25':0, '26-30':0, '31-35':0, '36-40':0 , '41up':0 } 
    Red = {'5':0 , '10':0 , '15':0, '20':0, '25':0, '30':0, '35':0, '40':0, '41':0} 
    Yellow = {'5':0 , '10':0 , '15':0, '20':0, '25':0, '30':0, '35':0, '40':0, '41':0} 
    Green = {'5':0 , '10':0 , '15':0, '20':0, '25':0, '30':0, '35':0, '40':0, '41':0} 


    txt_lsit = os.listdir(txt_path)
    txt_lsit = natsorted(txt_lsit)

    height, width = 1536, 2048
    maxsize = 0
    for txt in txt_lsit:
        objs = open(os.path.join(txt_path,txt), 'r')
        for obj in (objs):
            obj= obj.split(" ")
            cls, x, y, w, h  = obj[0], obj[1], obj[2], obj[3], obj[4]
            xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
            size = ((xmax-xmin-1)//5+1)*5
            if size>40:
                size = 41
            if cls == '0':
                Red[str(size)] += 1
            elif cls == '1':
                Yellow[str(size)] += 1
            elif cls == '2':
                Green[str(size)] += 1

        objs.close()
        # cv2.imwrite(os.path.join(out_path,txt.replace('.txt','.jpg')), img)
    print("Red:",Red)
    print("Yellow:",Yellow)
    print("Green:",Green)

def comp_TPFP():
    # img_path = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/images'#test/day_5/  
    dir = ['FN','TP']
    img_dirs = ['20230620/img/go/44', '20221115']
    clses = ['Red','Yellow','Green']#,'Left','Straight','Right','Background'
    FN_r,FN_g,FN_b,FN_h, FN_s, FN_v = [], [], [], [], [], []
    TP_r,TP_g,TP_b,TP_h, TP_s, TP_v = [], [], [], [], [], []
    for type in dir :
        # cls_r={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        # cls_g={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        # cls_b={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        # cls_h={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        # cls_s={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        # cls_v={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0,'Background':0}
        cls_r={'Red':0, 'Yellow':0,'Green':0}
        cls_g={'Red':0, 'Yellow':0,'Green':0}
        cls_b={'Red':0, 'Yellow':0,'Green':0}
        cls_h={'Red':0, 'Yellow':0,'Green':0}
        cls_s={'Red':0, 'Yellow':0,'Green':0}
        cls_v={'Red':0, 'Yellow':0,'Green':0}
        for img_dir in img_dirs :
            img_dirname = Path(img_dir).stem
            img_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/'+ img_dir +'/JPEGImages'
            txt_path = '/home/rvl/Desktop/yolov7/runs/test/Tainan_3cls_val/' + type + '/' + img_dirname
            out_path = '/home/rvl/Desktop/yolov7/runs/test/Tainan_3cls_val/' + type + '_img/'+ img_dirname
            if not os.path.isdir(Path(out_path).parent):
                os.mkdir(Path(out_path).parent)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            path = os.path.join(out_path,'fullimg')
            if not os.path.isdir(path):
                os.mkdir(path)
            # path = os.path.join(out_path,'center')
            # if not os.path.isdir(path):
            #     os.mkdir(path)   
            for cls in clses:
                path = os.path.join(out_path,cls)
                # center_path = os.path.join(out_path,'center',cls)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                os.mkdir(path)
                # if not os.path.isdir(center_path):
                #     os.mkdir(center_path)

            txt_lsit = os.listdir(txt_path)
            txt_lsit = natsorted(txt_lsit)
            if img_dirname == '44':
                replace_img = '.png'
            else:
                replace_img = '.jpg'

            for txt in txt_lsit:
                if os.stat(os.path.join(txt_path,txt)).st_size == 0 or not os.path.exists(os.path.join(txt_path,txt)): # an empty file
                    continue
                else:
                    objs = open(os.path.join(txt_path,txt), 'r')
                    ori_img = cv2.imread(os.path.join(img_path,txt.replace('.txt',replace_img)))
                    full_img = cv2.imread(os.path.join(img_path,txt.replace('.txt',replace_img)))
                    for i, obj in enumerate(objs):
                        obj = obj[:-1].split(" ")
                        cls, xmin, ymin, xmax, ymax = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
                        img_crop = ori_img[int(ymin):int(ymax),int(xmin):int(xmax)]
                        
                        if int(cls) < 7:
                            cv2.imwrite(os.path.join(out_path,clses[int(cls)],txt.replace('.txt','_'+str(i)+'.jpg')), img_crop)
                            w, h = (xmax-xmin)//3, (ymax-ymin)//3
                            center = ori_img[int(ymin+h):int(ymax-h), int(xmin+w):int(xmax-w)]
                            # cv2.imwrite(os.path.join(out_path,'center', clses[int(cls)],txts+'_'+txt.replace('.txt','_'+str(i)+'.jpg')), center)

                            b, g, r = np.mean(np.mean(center,1),0)
                            h, s, v = np.mean(np.mean(cv2.cvtColor(center,cv2.COLOR_BGR2HSV),1),0)
                            cls_r[clses[int(cls)]], cls_g[clses[int(cls)]], cls_b[clses[int(cls)]] = (cls_r[clses[int(cls)]]+r)//2, (cls_g[clses[int(cls)]]+g)//2, (cls_b[clses[int(cls)]]+b)//2
                            cls_h[clses[int(cls)]], cls_s[clses[int(cls)]], cls_v[clses[int(cls)]] = (cls_h[clses[int(cls)]]+h)//2, (cls_s[clses[int(cls)]]+s)//2, (cls_v[clses[int(cls)]]+v)//2

                        cv2.rectangle(full_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        cv2.putText(full_img, clses[int(cls)], (int(xmin-10), int(ymax)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1)
                
                    objs.close()

                cv2.imwrite(os.path.join(out_path,'fullimg',txt.replace('.txt','.jpg')), full_img)
        if type == 'FN':
            name = 'FN'
            FN_r,FN_g,FN_b,FN_h, FN_s, FN_v = list(cls_r.values()), list(cls_g.values()), list(cls_b.values()), list(cls_h.values()), list(cls_s.values()), list(cls_v.values())
        else:
            name = 'TP'
            TP_r,TP_g,TP_b,TP_h, TP_s, TP_v = list(cls_r.values()), list(cls_g.values()), list(cls_b.values()), list(cls_h.values()), list(cls_s.values()), list(cls_v.values())

        print(type+':BGR:\nRed={B},{G},{R}\n'.format(B=cls_b['Red'],G=cls_g['Red'],R=cls_r['Red']) + 
                    'Yellow={B},{G},{R}\n'.format(B=cls_b['Yellow'],G=cls_g['Yellow'],R=cls_r['Yellow']) + 
                    'Green={B},{G},{R}\n'.format(B=cls_b['Green'],G=cls_g['Green'],R=cls_r['Green']))
                    # 'Left={B},{G},{R}\n'.format(B=cls_b['Left'],G=cls_g['Left'],R=cls_r['Left'])+ 
                    # 'Straight={B},{G},{R}\n'.format(B=cls_b['Straight'],G=cls_g['Straight'],R=cls_r['Straight'])+ 
                    # 'Right={B},{G},{R}\n'.format(B=cls_b['Right'],G=cls_g['Right'],R=cls_r['Right']))
        print(type+':HSV:\nRed={H},{S},{V}\n'.format(H=cls_h['Red'],S=cls_s['Red'],V=cls_v['Red']) + 
                    'Yellow={H},{S},{V}\n'.format(H=cls_h['Yellow'],S=cls_s['Yellow'],V=cls_v['Yellow']) + 
                    'Green={H},{S},{V}\n'.format(H=cls_h['Green'],S=cls_s['Green'],V=cls_v['Green']))
                    # 'Left={H},{S},{V}\n'.format(H=cls_h['Left'],S=cls_s['Left'],V=cls_v['Left'])+ 
                    # 'Straight={H},{S},{V}\n'.format(H=cls_h['Straight'],S=cls_s['Straight'],V=cls_v['Straight'])+ 
                    # 'Right={H},{S},{V}\n'.format(H=cls_h['Right'],S=cls_s['Right'],V=cls_v['Right']))
        f = open(os.path.join(out_path,'../../BGR_HSV.txt'),'a')
        f.write(type+'\nBGR:\nRed={B},{G},{R}\n'.format(B=cls_b['Red'],G=cls_g['Red'],R=cls_r['Red']) + 
                    'Yellow={B},{G},{R}\n'.format(B=cls_b['Yellow'],G=cls_g['Yellow'],R=cls_r['Yellow']) + 
                    'Green={B},{G},{R}\n'.format(B=cls_b['Green'],G=cls_g['Green'],R=cls_r['Green'])+
                    # 'Left={B},{G},{R}\n'.format(B=cls_b['Left'],G=cls_g['Left'],R=cls_r['Left'])+ 
                    # 'Straight={B},{G},{R}\n'.format(B=cls_b['Straight'],G=cls_g['Straight'],R=cls_r['Straight'])+ 
                    # 'Right={B},{G},{R}\n'.format(B=cls_b['Right'],G=cls_g['Right'],R=cls_r['Right'])+
                    'HSV:\nRed={H},{S},{V}\n'.format(H=cls_h['Red'],S=cls_s['Red'],V=cls_v['Red']) + 
                    'Yellow={H},{S},{V}\n'.format(H=cls_h['Yellow'],S=cls_s['Yellow'],V=cls_v['Yellow']) + 
                    'Green={H},{S},{V}\n'.format(H=cls_h['Green'],S=cls_s['Green'],V=cls_v['Green']))
                    # 'Left={H},{S},{V}\n'.format(H=cls_h['Left'],S=cls_s['Left'],V=cls_v['Left'])+ 
                    # 'Straight={H},{S},{V}\n'.format(H=cls_h['Straight'],S=cls_s['Straight'],V=cls_v['Straight'])+ 
                    # 'Right={H},{S},{V}\n\n'.format(H=cls_h['Right'],S=cls_s['Right'],V=cls_v['Right']))
        f.close()
        #draw meanRGB for FN and TP
        color_img = np.zeros((30,180,3), np.uint8)
        cv2.circle(color_img, (15,15), 15, (cls_b['Red'], cls_g['Red'], cls_r['Red']), -1)
        cv2.circle(color_img, (45,15), 15, (cls_b['Yellow'], cls_g['Yellow'], cls_r['Yellow']), -1)
        cv2.circle(color_img, (75,15), 15, (cls_b['Green'], cls_g['Green'], cls_r['Green']), -1)
        # cv2.circle(color_img, (105,15), 15, (cls_b['Left'], cls_g['Left'], cls_r['Left']), -1)
        # cv2.circle(color_img, (135,15), 15, (cls_b['Straight'], cls_g['Straight'], cls_r['Straight']), -1)
        # cv2.circle(color_img, (165,15), 15, (cls_b['Right'], cls_g['Right'], cls_r['Right']), -1)
        cv2.imwrite(os.path.join(out_path,'..','meanRGB.png'), color_img)

        width = 0.3
        x = np.arange(len(clses))

        plt.bar(x, list(cls_h.values()), width, label=name+'_H')
        plt.bar(x + width, list(cls_s.values()), width, label=name+'_S')
        plt.bar(x + 2*width, list(cls_v.values()), width, label=name+'_V')
        
        plt.xticks(x + width, clses)
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.title(name+'_HSV')
        plt.savefig(os.path.join(out_path,'..',name+'_HSV.png'), bbox_inches='tight')
        # plt.show()
        plt.clf()

        plt.bar(x, list(cls_b.values()), width, label=name+'_B')
        plt.bar(x + width, list(cls_g.values()), width, label=name+'_G')
        plt.bar(x + 2*width, list(cls_r.values()), width, label=name+'_R')
        
        plt.xticks(x + width, clses)
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.title(name+'_BGR')
        plt.savefig(os.path.join(out_path,'..',name+'_BGR.png'), bbox_inches='tight')
        # plt.show()
        plt.clf()



    width = 0.3
    x = np.arange(len(clses))

    plt.bar(x, TP_h, width, label='TP_H')
    plt.bar(x + width, TP_s, width, label='TP_S')
    plt.bar(x + 2*width, TP_v, width, label='TP_V')
    plt.bar(x, FN_h, width, label='FN_H', color='blue', alpha=0.5)
    plt.bar(x + width, FN_s, width, label='FN_S', color='orange', alpha=0.5)
    plt.bar(x + 2*width, FN_v, width, label='FN_V', color='g', alpha=0.5)
    plt.xticks(x + width, clses)
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.title('HSV')
    plt.savefig(os.path.join(out_path,'..','HSV.png'), bbox_inches='tight')  
    # plt.show()
    plt.clf()


    plt.bar(x, TP_b, width, label='TP_B')
    plt.bar(x + width, TP_g, width, label='TP_G')
    plt.bar(x + 2*width, TP_r, width, label='TP_R')
    plt.bar(x, FN_b, width, label='FN_B', color='blue', alpha=0.5)
    plt.bar(x + width, FN_g, width, label='FN_G', color='orange', alpha=0.5)
    plt.bar(x + 2*width, FN_r, width, label='FN_R', color='g', alpha=0.5)
    plt.xticks(x + width, clses)
    plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.title('BGR')
    plt.savefig(os.path.join(out_path,'..','BGR.png'), bbox_inches='tight')  
    # plt.show()
    plt.clf()
    

    print('finsh!')

def expend_train():
    train_img = '/home/rvl/Desktop/TrafficLight/newday/train_val_txt/newday_1/train.txt'
    # save_path = '/home/rvl/Desktop/TrafficLight/newday/augment'
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    # os.makedirs(save_path)

    imgs = open(train_img,'r')
    imgs  = imgs.readlines()
    _, imgs = train_test_split(imgs, test_size=0.3)
    imgs_txt = open('/home/rvl/Desktop/TrafficLight/newday/train_val_txt/newday_1/train_03aug.txt','w')
    for img in imgs:
        imgs_txt.write(img)
                
    hsv = [[3.0, -115.0, -17.0], [4.0, -23.0, -33.0], [4.0, -109.0, -1.0]] #Red, Yellow, Green
    # hsv = [[4.0, -128.0, -49.0], [2.0, -4.0, 5.0], [1.0, -86.0, 7.0]] #Red, Yellow, Green
    # hsv = [[1.0, -111.0, -28.0], [0.0, 1.0, 7.0], [4.0, -116.0, -45.0]] #Red, Yellow, Green
    weight = np.array([1])#[0.25, 0.5, 0.75, 1]
    # for w in weight:
    #     path = os.path.join(save_path,str(w))
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    for img in imgs:
        # imgs_txt.write(img)
        save_path = os.path.join(Path(img).parents[1], 'augment1','JPEGImages')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ori_img = cv2.imread(img[:-1])
        ex_img = cv2.cvtColor(cv2.imread(img[:-1]),cv2.COLOR_BGR2HSV)
        ex_img = np.array([ex_img, ex_img, ex_img, ex_img]) #((copy img),height,width,3(hsv))
        # print(ex_img.shape)
        gt_path = img.replace('JPEGImages','labels_3cls').replace('.jpg','.txt')[:-1]
        if not os.path.exists(gt_path): # an empty file
                continue
        else:    
            objs = open(gt_path,'r')
            height, width, _  = ori_img.shape
            for obj in objs:
                cls, x, y, w, h = obj.split(" ")
                
                x, y, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
                
                # print(im)
                mask_size = [int(h*3), int(w*3)]
                mask_1d = cv2.getGaussianKernel(mask_size[0], 0)
                mask_1d1 = cv2.getGaussianKernel(mask_size[1], 0)
                mask_2d = mask_1d*mask_1d1.T 
                mask_2d = mask_2d/np.linalg.norm(mask_2d)
                mask_2d = mask_2d*(1/np.max(mask_2d))
                hsv_mask = mask_2d.reshape(mask_2d.shape[0],-1,1)*np.array(hsv[int(cls)]) #[ , ,3]
                mask = np.array(((hsv_mask*weight.reshape([1,1,1,-1]).T))) #[3, , ,3] [weight,masksize(,),hsv]
                mask = mask.astype('int8')

                #write mask img
                # im = ori_img[int(y-1.5*h):int(y+1.5*h),int(x-1.5*w):int(x+1.5*w)]
                # mask_2d_test = mask_2d*(1/np.max(mask_2d))*255
                # mask_2d_test = mask_2d_test.reshape(mask_2d_test.shape[0],-1,1)
                # im1 =(mask_2d_test*[1,1,1]).astype('uint8')
                # cv2.imwrite(Path(img).stem+'_obj.png',im)
                # cv2.imwrite(Path(img).stem+'_ori.png',ori_img)
                # cv2.imwrite(Path(img).stem+'_Gauss.png',mask_2d_test)
                # mix = cv2.addWeighted(im, 0.5, im1, 0.3, 50)
                # cv2.imwrite(Path(img).stem+'_mix.png',mix)

                
                # np.set_printoptions(threshold=np.inf, linewidth=np.nan)
                bbox = [int(x-1.5*w),int( y-1.5*h), int(x+1.5*w), int(y+1.5*h)]
                img_bbox = [bbox[0] if bbox[0]>=0 else 0, bbox[1] if bbox[1]>=0 else 0, bbox[2] if bbox[2]<width else width, bbox[3] if bbox[3]<height else height] #xmin, ymin, xmax, ymax
                res_bbox = [abs(a-b) for a,b in zip(img_bbox, bbox)] #residual [left, top, right, bottom]
                
                for j in range(0,mask_size[0]-res_bbox[1]-res_bbox[3]):
                    for i in range(0,mask_size[1]-res_bbox[0]-res_bbox[2]):
                        sub = ex_img[:,img_bbox[1]+j,img_bbox[0]+i,:]+ mask[:,res_bbox[1]+j,res_bbox[0]+i,:]
                        sub[sub>255],sub[sub<0] = 255, 0
                        ex_img[:,img_bbox[1]+j,img_bbox[0]+i,:] =  sub

                #write mask img
                # im2 = ex_img[0][int(y-1.5*h):int(y+1.5*h),int(x-1.5*w):int(x+1.5*w)]
                # cv2.imwrite(Path(img).stem+'_aug.png',cv2.cvtColor(im2,cv2.COLOR_HSV2BGR))
                # a=input()
            
            name = (img.split('/')[-1])[:-1]
            for i, names in enumerate(weight):
                # print(os.path.join(save_path,str(names),str(names)+'_'+name))
                cv2.imwrite(os.path.join(save_path,'1_'+name),cv2.cvtColor(ex_img[0],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'0.5',name),cv2.cvtColor(ex_img[1],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'0.75',name),cv2.cvtColor(ex_img[2],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'1.0',name),cv2.cvtColor(ex_img[3],cv2.COLOR_HSV2BGR))
            # print(ex_img.dtype)
            # i = (mask[mask> 255] or mask[mask< 0])
            # print(i)
            


            # cv2.imshow('ori',ori_img.astype(np.uint8))
            # cv2.imshow('0.25',cv2.cvtColor(ex_img[0],cv2.COLOR_HSV2BGR).astype(np.uint8))
            # cv2.imshow('0.5',cv2.cvtColor(ex_img[1],cv2.COLOR_HSV2BGR).astype(np.uint8))
            # cv2.imshow('0.75',cv2.cvtColor(ex_img[2],cv2.COLOR_HSV2BGR).astype(np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

                # print(res_bbox)


            # a=input()
    print("finsh")
            
def deal_WH():
    txt_dir = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/labels_sets/nodif/labels_6cls'
    img_path = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/images'
    txt_lsit = os.listdir(txt_dir)
    txt_lsit = natsorted(txt_lsit)

    save_path = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/labels_sets/nodif/labels_6cls_pixel5' #labels_3class, labels_4class

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for txt in txt_lsit:
        ori_f = open(os.path.join(txt_dir,txt), 'r')
        save_f = open(os.path.join(save_path,txt),'w')
        img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
        height, width, _ = img.shape
        # print(ori_f)
        # a = input()
        for obj in ori_f:
            save_line = obj
            obj= obj.strip('\n').split(" ")
            cls, x, y, w, h  = obj[0], obj[1], obj[2], obj[3], obj[4]
            xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
            # print(xmax-xmin)
            
            if xmax-xmin > 5 or ymax-ymin > 5:
                save_f.write(save_line)
        save_f.close()

def img2video():

    # datasets = ['video_001','video_002','video_004','video_005']
    # sequences = {'Sequence1':4}

    # cont = 0
    # for sequence in sequences:
    #     for dataset in range(cont, sequences[sequence]):
    #         img_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]+'/JPEGImages'
    #         out_path = '/home/rvl/Desktop/TrafficLight/dataset/'+sequence+'/'+datasets[dataset]
    #         fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    #         out = cv2.VideoWriter(os.path.join(out_path,datasets[dataset]+'.mp4'), fourcc, 10, (2048,1536))
    #         img_list = os.listdir(img_path)
    #         img_list = natsorted(img_list)
    #         for img in img_list:
    #             ori_img = cv2.imread(os.path.join(img_path,img))
    #             out.write(ori_img)
    #         out.release()
    #         cont += 1
    #         print(datasets[dataset]+' finsh!')
    
    # img_path = '/home/rvl/Desktop/yolov7/runs/test/newday1_augmentred_3cls_video002_nocls_th5_iou4_1280_nodif/labels'
    img_path = '/home/rvl/Desktop/yolov7/runs/detect/test4'
    out_path = '/home/rvl/Desktop/yolov7/runs/detect'
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter(os.path.join(out_path,'test4.mp4'), fourcc, 10, (2048,1536))
    img_list = os.listdir(img_path)
    img_list = natsorted(img_list)
    for img in img_list:
        ori_img = cv2.imread(os.path.join(img_path,img))
        out.write(ori_img)
    out.release()
    print('finsh!')

def video2img():
    dirs = ['go/51','back/52','back/53','back/54','back/55']#'go/44','go/48','go/50',
    video_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/20230620/video'
    out_path = '/home/rvl/Desktop/TrafficLight/dataset/Chiayi_To_Tainan/day/20230620/img'
        
    for dir in dirs:
        video = os.listdir(os.path.join(video_path,dir))
        video = sorted(video)
        out_path1 = os.path.join(out_path,dir,'JPEGImages')
        if not os.path.isdir(out_path1):
            os.makedirs(out_path1)
        index=0
        for vid in video:
            cap = cv2.VideoCapture(os.path.join(video_path,dir,vid))
            suc = cap.isOpened()
            while suc:
                suc, image = cap.read()
                if suc and index%30==0:
                    prefix=str(index//30).zfill(6)+".png"
                    cv2.imwrite(os.path.join(out_path1,prefix),image)
                index+=1
            cap.release()
            print(vid,':',index//30)
  
def U2test():

    classes = ['Red' , 'Yellow', 'Green']
    nums = {'Red':9892 , 'Yellow':1296, 'Green':6327 }
    test1 = 'newday1_augment2_3cls_video002_nocls_th5_iou4_1280_nodif'
    test2 = 'newday1_augment1_3cls_video002_nocls_th5_iou4_1280_nodif'
    all = 0
    for cls in classes:
        test1_path = '/home/rvl/Desktop/yolov7/runs/test/'+test1+'/FN_img/'+cls
        test2_path = '/home/rvl/Desktop/yolov7/runs/test/'+test2+'/FN_img/'+cls
        num = 0

        img1_list = os.listdir(test1_path)
        img2_list = os.listdir(test2_path)
        for img1 in img1_list:
            for img2 in img2_list:
                
                if img2 == img1:
                    num = num + 1
                    # print(img1,img2)
                    # s=input()
        all = all+num
        print(cls, num, num/nums[cls])
    print('all ',all/17515)

if __name__ == "__main__":
    
    # xml2txt() #xml to txt
    # data_rename() #rename img and txt
    # txt_class_change() #change txt classes number
    # xml_class_change()
    # write_trainval_txt() #將data分成train和val，並寫成txt檔
    # cal_label_num() #計算label（紅燈等）數量
    # crop_img()
    # layoff_empty()#刪除空txt後得到img
    # draw_bboximg() #在圖片上畫出bbox
    # comp_ori_label() #畫出原始與預測結果的影片
    # txt_xywh2xyxy().
    # cal_det_light_size()
    # comp_TPFP()
    # expend_train()
    # deal_WH()
    img2video()
    # video2img()
    # U2test()



    # txt_path = '/home/rvl/Desktop/TrafficLight/dataset/Sequence1/video_002/labels_6cls' #labels_sets/labels_6class
    # img_path = '/home/rvl/Desktop/TrafficLight/dataset/Sequence1/video_002/JPEGImages' 
    # out_path = '/home/rvl/Desktop/TrafficLight/dataset/Sequence1/video_002/draw5pixel'

    # if not os.path.isdir(out_path):
    #     os.mkdir(out_path)

    # txt_lsit = os.listdir(txt_path)

    # for txt in txt_lsit:
    #     if txt == '006765.txt' or txt =='006769.txt':
    #         objs = open(os.path.join(txt_path,txt),'r')
    #         # print(os.path.join(img_path,txt.replace('.txt','.jpg')))
    #         img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
    #         height, width, _ = img.shape
    #         draw = 0
    #         for obj in objs:
    #             cls, x, y, w, h = obj.split(" ")
    #             xmin, xmax, ymin, ymax = xywh2xyxy(width, height, float(x), float(y), float(w), float(h))
    #             if xmax-xmin>=5 or ymax-ymin>=5:
    #                 cv2.rectangle(img, (xmin-1, ymin-1), (xmax+1, ymax+1), (0,0,255), 2)
    #                 draw = 1
    #         if draw:
    #             cv2.imwrite(os.path.join(out_path,txt.replace('.txt','.jpg')), img)


        

    # img_path = '/home/rvl/Desktop/VGG16-PyTorch/vgg16/results.txt' #labels_sets/labels_6class
    # out_path = '/home/rvl/Desktop/VGG16-PyTorch/vgg16/error'
    # classes = ['Red', 'Yellow', 'Green', 'Left', 'Straight', 'Right', 'Background']
    # for i in range(7):
    #     path = os.path.join(out_path,str(i))
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
    #     for j in range(7):
    #         path_cls = os.path.join(path,classes[j])
    #         print(path)
    #         if not os.path.isdir(path_cls):
    #             os.mkdir(path_cls)
    # imgs = open(img_path,'r')
    # for img in imgs:
    #     path, pre = img.split(' ')
    #     _, _, cls = str(Path(path).parent).split('/')
    #     shutil.copyfile(os.path.join('/home/rvl/Desktop/VGG16-PyTorch',path),os.path.join(out_path,str(cls),classes[int(pre)],Path(path).name))
    


    # img_path = '/home/rvl/Desktop/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt' #labels_sets/labels_6class
    
    # imgs = open(img_path,'r')
    # imgs_1 = open(img_path.replace('val','val_1'),'w')

    # for img in imgs:
    #     img1 = img[:-1]+'.jpg'
    #     print(img1.replace('JPEGImages','Annotations').replace('.jpg','.xml'))
    #     if os.path.exists(img1) and os.path.exists(img1.replace('JPEGImages','Annotations').replace('.jpg','.xml')):
    #         imgs_1.write(img)
    # imgs.close()
    # imgs_1.close()

