
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


def crop_img():

    txt_path = '/home/rvl/Desktop/yolov7/TL/train_val_txt/newday_1/test.txt'
    outimg_path = '/home/rvl/Desktop/yolov7/TL/datasets/Crop_7cls/JPEGImages'
    outtxt_path = '/home/rvl/Desktop/yolov7/TL/datasets/Crop_7cls/labels'
    txt_list = open(txt_path,'r')

    if not os.path.exists(outimg_path):
        os.makedirs(outimg_path)
    if not os.path.exists(outtxt_path):
        os.makedirs(outtxt_path)
    
    for txt in txt_list:
        txt_path = txt[:-1].replace('.jpg','.txt').replace('JPEGImages','labels_6cls')
        img_path = txt[:-1]

        img = cv2.imread(img_path)
        H, W, _ = img.shape
        
        if not os.path.exists(txt_path) or os.stat(txt_path).st_size == 0: # an empty file
                continue
        else:
            objs = open(txt_path,'r')
            for idx, obj in enumerate(objs):
                cls, x, y, w, h = obj.split(' ')
                xmin, xmax, ymin, ymax = xywh2xyxy(W, H, float(x), float(y), float(w), float(h))

                # crop img 
                size = 2 if ymax-ymin < 20 or xmax-xmin < 20 else 5

                left = xmin if xmin-size<0 else size
                right = W-1-xmax if xmax+size>W-1 else size
                top = ymin if ymin-size<0 else size
                bottom = H-1-ymax if ymax+size>H-1 else size

                crop_img = img[ymin-top:ymax+bottom,xmin-left:xmax+right]

                cv2.imwrite(os.path.join(outimg_path, str(idx)+'.jpg'),crop_img)
                
                save_txt = open(os.path.join(outtxt_path, str(idx)+'.txt'),'w')

                h, w, _ = crop_img.shape
                xcen, ycen, w, h = xyxy2xywh(w, h, left, float(w-right), float(top), float(h-bottom))
                save_txt.write(cls + " " + str(xcen) + " " + str(ycen) + " " + str(w) + " " + str(h) + '\n')
                save_txt.close()
            objs.close()
            

def comp_TPFP():
    img_path = '/home/rvl/Desktop/TrafficLight/newday/test/video_002/images'
    dir = ['FN','TP']
    img_dirs = ['video_002']
    clses = ['Red','Yellow','Green']#,'Left','Straight','Right'
    FN_r,FN_g,FN_b,FN_h, FN_s, FN_v = [], [], [], [], [], []
    TP_r,TP_g,TP_b,TP_h, TP_s, TP_v = [], [], [], [], [], []
    for type in dir :
        # cls_r={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        # cls_g={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        # cls_b={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        # cls_h={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        # cls_s={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        # cls_v={'Red':0, 'Yellow':0,'Green':0,'Left':0,'Straight':0,'Right':0}
        cls_r={'Red':0, 'Yellow':0,'Green':0}
        cls_g={'Red':0, 'Yellow':0,'Green':0}
        cls_b={'Red':0, 'Yellow':0,'Green':0}
        cls_h={'Red':0, 'Yellow':0,'Green':0}
        cls_s={'Red':0, 'Yellow':0,'Green':0}
        cls_v={'Red':0, 'Yellow':0,'Green':0}
        for img_dir in img_dirs :
            img_dirname = Path(img_dir).stem
            txt_path = '/home/rvl/Desktop/yolov7/runs/test/Tainan_3cls_val/' + type + '/' + img_dirname
            out_path = '/home/rvl/Desktop/yolov7/runs/test/Tainan_3cls_val/' + type + '_img/'+ img_dirname
            if not os.path.isdir(Path(out_path).parent):
                os.mkdir(Path(out_path).parent)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            path = os.path.join(out_path,'fullimg')
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(out_path,'center')
            if not os.path.isdir(path):
                os.mkdir(path)   
            for cls in clses:
                path = os.path.join(out_path,cls)
                center_path = os.path.join(out_path,'center',cls)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                os.mkdir(path)
                if not os.path.isdir(center_path):
                    os.mkdir(center_path)

            txt_lsit = os.listdir(txt_path)
            txt_lsit = natsorted(txt_lsit)

            for txt in txt_lsit:
                if os.stat(os.path.join(txt_path,txt)).st_size == 0 or not os.path.exists(os.path.join(txt_path,txt)): # an empty file
                    continue
                else:
                    objs = open(os.path.join(txt_path,txt), 'r')
                    ori_img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
                    full_img = cv2.imread(os.path.join(img_path,txt.replace('.txt','.jpg')))
                    for i, obj in enumerate(objs):
                        obj = obj[:-1].split(" ")
                        cls, xmin, ymin, xmax, ymax = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
                        img_crop = ori_img[int(ymin):int(ymax),int(xmin):int(xmax)]
                        
                        if int(cls) < 7:
                            cv2.imwrite(os.path.join(out_path,clses[int(cls)],txt.replace('.txt','_'+str(i)+'.jpg')), img_crop)
                            w, h = (xmax-xmin)//3, (ymax-ymin)//3
                            center = ori_img[int(ymin+h):int(ymax-h), int(xmin+w):int(xmax-w)]
                            cv2.imwrite(os.path.join(out_path,'center', clses[int(cls)],txt+'_'+txt.replace('.txt','_'+str(i)+'.jpg')), center)

                            b, g, r = np.mean(np.mean(center,1),0)
                            h, s, v = np.mean(np.mean(cv2.cvtColor(center,cv2.COLOR_BGR2HSV),1),0)
                            cls_r[clses[int(cls)]], cls_g[clses[int(cls)]], cls_b[clses[int(cls)]] = (cls_r[clses[int(cls)]]+r)//2, (cls_g[clses[int(cls)]]+g)//2, (cls_b[clses[int(cls)]]+b)//2
                            cls_h[clses[int(cls)]], cls_s[clses[int(cls)]], cls_v[clses[int(cls)]] = (cls_h[clses[int(cls)]]+h)//2, (cls_s[clses[int(cls)]]+s)//2, (cls_v[clses[int(cls)]]+v)//2

                        cv2.rectangle(full_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                        # cv2.putText(full_img, clses[int(cls)], (int(xmin-10), int(ymax)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1)
                
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

def augment_train():
    train_img = '/home/rvl/Desktop/TrafficLight/newday/train_val_txt/newday_1/train.txt'

    imgs = open(train_img,'r')
    
    hsv = [[3.0, -115.0, -17.0], [4.0, -23.0, -33.0], [4.0, -109.0, -1.0]] #Red, Yellow, Green
    weight = np.array([1])#[0.25, 0.5, 0.75, 1]
    # for w in weight:
    #     path = os.path.join(save_path,str(w))
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    for img in imgs:
        file_name = Path(img).name[-5:]
        save_path = os.path.join(Path(img).parents[1], 'augment','JPEGImages')
        txt_save_path = save_path.replace('JPEGImages','labels_3cls')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(txt_save_path):
            os.makedirs(txt_save_path)
        ori_img = cv2.imread(img[:-1])

        ex_img = cv2.cvtColor(cv2.imread(img[:-1]),cv2.COLOR_BGR2HSV)
        ex_img = np.array([ex_img, ex_img, ex_img, ex_img])
        txt_path = img.replace('JPEGImages','labels_3cls').replace(file_name,'.txt')
        if not os.path.exists(txt_path): # an empty file
                continue
        else:    
            name = Path(img).stem
            shutil.copy(txt_path,os.path.join(txt_save_path,'1_'+name+'.txt'))
            objs = open(txt_path,'r')
            height, width, _  = ori_img.shape
            for obj in objs:
                cls, x, y, w, h = obj.split(" ")
                x, y, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
                
                mask_size = [int(h*3), int(w*3)]
                mask_1d = cv2.getGaussianKernel(mask_size[0], 0)
                mask_1d1 = cv2.getGaussianKernel(mask_size[1], 0)
                mask_2d = mask_1d*mask_1d1.T 
                mask_2d = mask_2d/np.linalg.norm(mask_2d)
                mask_2d = mask_2d*(1/np.max(mask_2d))
                hsv_mask = mask_2d.reshape(mask_2d.shape[0],-1,1)*np.array(hsv[int(cls)]) #[ , ,3]
                mask = np.array(((hsv_mask*weight.reshape([1,1,1,-1]).T))) #[3, , ,3] [weight,masksize(,),hsv]
                mask = mask.astype('int8')

            
                bbox = [int(x-1.5*w),int( y-1.5*h), int(x+1.5*w), int(y+1.5*h)]
                img_bbox = [bbox[0] if bbox[0]>=0 else 0, bbox[1] if bbox[1]>=0 else 0, bbox[2] if bbox[2]<width else width, bbox[3] if bbox[3]<height else height] #xmin, ymin, xmax, ymax
                res_bbox = [abs(a-b) for a,b in zip(img_bbox, bbox)] #residual [left, top, right, bottom]
                
                for j in range(0,mask_size[0]-res_bbox[1]-res_bbox[3]):
                    for i in range(0,mask_size[1]-res_bbox[0]-res_bbox[2]):
                        sub = ex_img[:,img_bbox[1]+j,img_bbox[0]+i,:]+ mask[:,res_bbox[1]+j,res_bbox[0]+i,:]
                        sub[sub>255],sub[sub<0] = 255, 0
                        ex_img[:,img_bbox[1]+j,img_bbox[0]+i,:] =  sub
            
            name = (img.split('/')[-1])[:-1]
            for i, names in enumerate(weight):
                cv2.imwrite(os.path.join(save_path,'1_'+name),cv2.cvtColor(ex_img[0],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'0.5',name),cv2.cvtColor(ex_img[1],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'0.75',name),cv2.cvtColor(ex_img[2],cv2.COLOR_HSV2BGR))
            # cv2.imwrite(os.path.join(save_path,'1.0',name),cv2.cvtColor(ex_img[3],cv2.COLOR_HSV2BGR))

if __name__ == "__main__":
    
    # crop_img()
    # comp_TPFP()
    # augment_train()

