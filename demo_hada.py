from scipy.special.orthogonal import jacobi
import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
# import ipm as ipm
import PIL
import math
from PIL import Image

import socket
import ctypes as ct
import sys
import struct
from numba import jit
import csv

import timing as timing
import numpy as np
from cam_util import UDP_CAM_Parser
import os,json
import socket
import cv2
import numpy as np
import time
from cam_util import UDP_CAM_Parser
import os,json


import socket
import threading
import struct

##################################################################################
RAD2DEG    = float(180)/np.pi
DEG2RAD    = np.pi/float(180)
SIM_TIME   = float(0.0) 
INI_TIME   = float(0.0)
CUR_TIME   = float(0.0)
TSK_TIME   = float(0.0)
DEL_TIME   = float(0.0)
START_TIME = float(0.0)

IDLE_TIME  = float(0.0)
SIM_COUNT  = int(0)  
SAMPLING_FREQ   = float(20.0)     #HZ
SAMPLING_PERIOD = float(0.1)

START_FOR = float(0.0)
END_FOR   = float(0.0)

def IdleTime(_ini_time, _cur_time, _sim_time, _Ts):
    while(True):
        _cur_time = timing.micros()*1e-6 # [ms]
        IDLE_TIME = _cur_time - _sim_time - _ini_time
        
        if(IDLE_TIME >= _Ts):
            break

@jit (nopython = True)
def makearray(min, max, intv):
    arr_ = np.arange(min, max ,intv)
    return arr_

@jit (nopython = True)
def color_image(i_ ,j_ ,color,img):
    for i in i_:
        for j in j_:
            img[i,j] = color
    return img


## 시작전 설정
CAM        = False  # True: ON / False : OFF 
UDP_ONOFF  = 1  # 1: ON / 0 : OFF
FILENAME   = "Lane_Test1"  # 저장할 파일 이름



if __name__ == "__main__":
    ######################################################################
    ######################## UDP Communication ###########################
    ######################################################################
    if(UDP_ONOFF):
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        Host = '192.168.1.12'
        Port = 1101

    src=[]
    dst_np = []

    #######################################################################
    #######################################################################
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'HADA':
        cls_num_per_lane = 56
    elif cfg.dataset == 'HADA_CU':
        cls_num_per_lane = 18
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    toggle = False # For CULane

    if cfg.dataset == 'HADA':
        splits = ['Hada_pilot.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor

    elif cfg.dataset == 'HADA_CU':
        splits = ['Hada_pilot.txt']                    
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        #img_w, img_h = 1280, 720
        row_anchor = culane_row_anchor
        toggle = True
    else:
        raise NotImplementedError


    ############################## UDP COMMUNICATION  ################################
    if(UDP_ONOFF):
        print("===================UDP Send Recv Check=====================")
        ###########Send Check#############
        testArray1 = np.asarray([float(0.0), float(0.0)])
        SendArray = (ct.c_double*len(testArray1)).from_buffer(bytearray(testArray1))
        client_sock.sendto((SendArray),(Host,Port))
        print("Send Success!\n")


    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print('HADA3'+'.avi')
    vout = cv2.VideoWriter('HADA'+'.avi', fourcc , 30.0, (img_w, img_h))

    prev_offset = float(0.0)
    prev_angle  = float(0.0)
    prev_cmd    = float(0.0)
    count_fail  = int(0)

    count = 0
    if(CAM):
        cap = cv2.VideoCapture(-1)       
        # cap = cv2.VideoCapture(cv2.CAP_DSHOW + 1) 
    else:
        cap = cv2.VideoCapture(cfg.src_video)

    buf_cmd_angle = []
    buf_flag_fail = []
    INI_TIME = timing.micros()*1e-6

    while(True):    
        if (1) :
            # if udp_cam.is_img==True :
                #img_cam = udp_cam.raw_img
                # cv2.imshow("cam", img_cam)
                # cv2.waitKey(1)
            
            ret, image = cap.read()
            image = cv2.resize(image,(1280,720))
            

            START_TIME = timing.micros()*1e-6
            image_ori  = image.copy()
            if toggle:
                image = cv2.resize(image,(1640,590))
            
            image_ori2 = image.copy()

            ori_width = image.shape[1]
            ori_height = image.shape[0]


            image=Image.fromarray(image)
            imgs = img_transforms(image)
            imgs = torch.reshape(imgs, [1,3,288,800])

            imgs = imgs.cuda()


            with torch.no_grad():
                out = net(imgs)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]

            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)

            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            

            pt_count  = [int(0),int(0),int(0),int(0),int(0)]
            cam_pt_left  = [] # (x,y)
            cam_pt_right = [] # (x,y)
            total_pt  = 0
            MIN_PT    = int(50) # 카메라 위치에 따라 달라짐.
            vis = image_ori2


            ##################################################################
            ########################### IPM CODE ##############################
            ##### Opencv Tool. getPerspectiveTransform / warpPerspective #####
            ##################################################################
            ipm_image = vis.copy()
            # src_np = np.array([[364,421],
            #         [952,412],
            #         [1188,521],
            #         [156,520],
            #         ],dtype = np.float32)

            # src_np = np.array([[477,309],
            #         [735,309],
            #         [914,389],
            #         [312,389]
            #         ],dtype = np.float32)
            src_np = np.array([[513,329],
                    [919,329],
                    [1148,449],
                    [223,449]
                    ],dtype = np.float32)




            ipm_width = max(np.linalg.norm(src_np[0]-src_np[1]), np.linalg.norm(src_np[2]-src_np[3]))
            ipm_height = max(np.linalg.norm(src_np[0]-src_np[3]), np.linalg.norm(src_np[1]-src_np[2]))

            IPM_width_left = int(640/2)-int(50)
            IPM_width_right = int(640/2)+int(50)
            IPM_height = int(480)

            dst_np = np.array([[IPM_width_left,0],
                    [IPM_width_right,0],
                    [IPM_width_right, IPM_height],
                    [IPM_width_left, IPM_height]
                    ],dtype =np.float32)

            IPM_Matrix = cv2.getPerspectiveTransform(src_np , dst_np)
            IPM_result = cv2.warpPerspective(image_ori, M= IPM_Matrix, dsize=(int(640), int(480)))
            

            ### DATA PRINTING ###

            PPAP = []  ##오른쪽 차선 x 좌표점들
            PAPP = []  ##왼쪽 차선 x 좌표점들
            PPAPy = []  ##오른쪽 차선 y 좌표점들
            PAPPy = []  ##왼쪽 차선 y 좌표점들

            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            # print("ppp:")
                            # print(ppp)
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
                            if((IPM_Matrix[2,0]*ppp[0]+IPM_Matrix[2,1]*ppp[1]+IPM_Matrix[2,2] !=0) and (IPM_Matrix[2,0]*ppp[0]+IPM_Matrix[2,1]*ppp[1]+IPM_Matrix[2,2]) !=0 ):
                                dstx = int((IPM_Matrix[0,0]*ppp[0]+IPM_Matrix[0,1]*ppp[1]+IPM_Matrix[0,2])/(IPM_Matrix[2,0]*ppp[0]+IPM_Matrix[2,1]*ppp[1]+IPM_Matrix[2,2]))
                                dsty = int((IPM_Matrix[1,0]*ppp[0]+IPM_Matrix[1,1]*ppp[1]+IPM_Matrix[1,2])/(IPM_Matrix[2,0]*ppp[0]+IPM_Matrix[2,1]*ppp[1]+IPM_Matrix[2,2]))
                            
                            ############ 640 480 IPM############
                            if(dstx < int(300)) and (dstx > int(240)) and (dsty > int(0)) and (dsty < int(481)):
                                PAPP.append(dstx) ##투심플 4차선
                                PAPPy.append(dsty)
                            
                            if(dstx > int(310)) and (dstx < int(410)) and (dsty > int(0)) and (dsty < int(481)):
                                PPAP.append(dstx)
                                PPAPy.append(dsty)

                            if (abs(dstx)<641) and (abs(dsty) <481) and (dsty>0):
                                cv2.circle(IPM_result,(dstx,dsty),3,(0,255,0),-1)
                                
                                #print((dstx,dsty))    #272 372 -> 100 3.4m => 3.4cm per pixel 0.034m
                            total_pt = total_pt + 1    
            
            # print(PAPP)
            # print(PAPPy)
            # print(PPAP)
            # print(PPAPy)
            
            
            ##################################################################
            #################### Pure Pursuit Guidance #######################
            ##################################################################
            car_velocity = 5
            ld = float((car_velocity*0.237)+5.29) *0.8

            width_pixels = 110
            one_pix_m = (3.4)/width_pixels ##픽셀당  미터
            car_rear_m = 1.12 / one_pix_m ## 픽셀수 차량 길이
            
            car_rear_x = 320
            car_rear_y = 480 + car_rear_m

            ld_pix = ld / one_pix_m
            G_y = car_rear_y - ld_pix
            
            # print(G_y) 
            alpha_angle = 0
            delta_cmd = 0



            if((len(PPAP)>2) and (len(PAPP)>2) and ((PPAPy[-3]-PPAPy[-1]) != 0) and ((PAPPy[-3]-PAPPy[-1]) != 0)):

                LINE_PA_X = (G_y-PPAPy[-1])*((PPAP[-3]-PPAP[-1])/(PPAPy[-3]-PPAPy[-1])) + PPAP[-1]
                LINE_AP_X = (G_y-PAPPy[-1])*((PAPP[-3]-PAPP[-1])/(PAPPy[-3]-PAPPy[-1])) + PAPP[-1]
                G_x = (LINE_PA_X + LINE_AP_X)/2
                print(G_x)
                if(G_x>320):
                    # alpha_angle = np.arctan((G_x-320)/(ld_pix)) * RAD2DEG
                    alpha_angle = np.arctan((G_x-320)*one_pix_m/(ld)) * RAD2DEG
                if(G_x<320):
                    # alpha_angle = -np.arctan((320-G_x)/(ld_pix)) * RAD2DEG
                    alpha_angle = -np.arctan((320-G_x)*one_pix_m/(ld)) * RAD2DEG
                print("alpha_angle:"+str(alpha_angle)+"[deg]\n")
                delta_cmd = -np.arctan((2*1.12*np.sin(alpha_angle*DEG2RAD))/ld)*100

                print("delta_cmd:"+str(delta_cmd)+"[deg]\n")
                print("ld:"+str(ld)+"[m]\n")
                cv2.circle(IPM_result,(int(G_x),int(G_y)),3,(0,0,255),-1)

            
            
            ## 우측 차선이 안잡히고 좌측 한차선만 잡힌경우 우측 가상차선 생성
            if(len(PPAP)>3 and len(PAPP)>3 and ((PAPPy[-3]-PAPPy[-1]) != 0) ): 
                for AP in range(len(PAPP)):
                    cv2.circle(IPM_result,(PAPP[AP]+width_pixels,PAPPy[AP]),3,(0,0,255),-1)
                    LINE_AP_X = (G_y-PAPPy[-1])*((PAPP[-3]-PAPP[-1])/(PAPPy[-3]-PAPPy[-1])) + PAPP[-1]
                    G_x = LINE_AP_X+(width_pixels)/2
                    cv2.circle(IPM_result,(int(G_x),int(G_y)),3,(0,255,255),-1)

                if(G_x>320):
                    # alpha_angle = np.arctan((G_x-320)/(ld_pix)) * RAD2DEG
                    alpha_angle = np.arctan((G_x-320)*one_pix_m/(ld)) * RAD2DEG
                if(G_x<320):
                    # alpha_angle = -np.arctan((320-G_x)/(ld_pix)) * RAD2DEG
                    alpha_angle = -np.arctan((320-G_x)*one_pix_m/(ld)) * RAD2DEG
                print("alpha_angle:"+str(alpha_angle)+"[deg]\n")
                delta_cmd = -np.arctan((2*1.12*np.sin(alpha_angle*DEG2RAD))/ld)*100

                print("delta_cmd:"+str(delta_cmd)+"[deg]\n")

            ## 좌측 차선이 안잡히고 우측 한차선만 잡힌경우 좌측 가상차선 생성
            if(len(PPAP)>3 and len(PAPP)>3 and ((PPAPy[-3]-PPAPy[-1]) != 0)): 
                for PA in range(len(PPAP)):
                    cv2.circle(IPM_result,(PPAP[PA]-width_pixels,PPAPy[PA]),3,(255,0,0),-1)
                    LINE_PA_X = (G_y-PPAPy[-1])*((PPAP[-3]-PPAP[-1])/(PPAPy[-3]-PPAPy[-1])) + PPAP[-1]
                    G_x = LINE_PA_X-(width_pixels)/2
                    cv2.circle(IPM_result,(int(G_x),int(G_y)),3,(0,255,255),-1)
                if(G_x>320):
                    # alpha_angle = np.arctan((G_x-320)/(ld_pix)) * RAD2DEG
                    alpha_angle = np.arctan((G_x-320)*one_pix_m/(ld)) * RAD2DEG
                if(G_x<320):
                    # alpha_angle = -np.arctan((320-G_x)/(ld_pix)) * RAD2DEG
                    alpha_angle = -np.arctan((320-G_x)*one_pix_m/(ld)) * RAD2DEG
                print("alpha_angle:"+str(alpha_angle)+"[deg]\n")
                delta_cmd = -np.arctan((2*1.12*np.sin(alpha_angle*DEG2RAD))/ld)*100

                print("delta_cmd:"+str(delta_cmd)+"[deg]\n")
            

            FLAG_fail = float(0.0) ## 임시 FLAG_fail. 이후에는 꼭 지울 것.  
            buf_cmd_angle.append(delta_cmd)
            buf_flag_fail.append(FLAG_fail)
            
            ####################################################################
            ########################### UDP COMMUNICATION  #####################
            ################### Pass cmd_angle & FLAG_LANE #####################
            ####################################################################  
            testArray1 = np.asarray([float(delta_cmd),float(FLAG_fail)])
            SendArray = (ct.c_double*len(testArray1)).from_buffer(bytearray(testArray1))
            client_sock.sendto(bytes(SendArray),(Host,Port))

        
            # print(PPAP)
            # print('\n')
            # print(PAPPy)
            ## PAPP : 왼쪽부터 2번째 차선
            ## PPAP : 왼쪽부터 3번째 차선
            # if (len(PPAP)>7) and (len(PPAP)>7): 

            #print("============================================================")
            count = count + 1
            CUR_TIME = timing.micros()*1e-6
            IdleTime(INI_TIME, CUR_TIME, SIM_TIME, SAMPLING_PERIOD)
            SIM_TIME = SAMPLING_PERIOD * SIM_COUNT
            SIM_COUNT = SIM_COUNT +1
            TSK_TIME  = CUR_TIME - START_TIME

            fps = float(1/(TSK_TIME))
            fps = round(fps,2)
            # middle_offset = round(middle_offset,4)
            # psi_angle = round(psi_angle,4)

            cv2.putText(vis, "FPS: "+str(fps),(800,100),cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),2)
            cv2.putText(vis, "SAMPLING FREQ: "+str(SAMPLING_FREQ)+"[HZ]",(800,150),cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),2)
            #cv2.putText(vis, "alpha_angle: "+str(alpha_angle)+ "[deg]",(800,250),cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),2)
            cv2.putText(vis, "delta_cmd: "+str(delta_cmd) +"[deg]",(800,200),cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),2)
            cv2.imshow('IPM', IPM_result)
            cv2.imshow('result',vis)

            k = cv2.waitKey(1)

            vout.write(vis)
            
            if(k%256 == 27):
                break
    ####################################################################
    ########################## Save Data ###############################
    ####################################################################
    with open(FILENAME + '.csv','w') as f :
        for i in range(len(buf_cmd_angle)):
            f.write("{}, {}\n".format(buf_cmd_angle[i], buf_flag_fail[i]))

    vout.release()