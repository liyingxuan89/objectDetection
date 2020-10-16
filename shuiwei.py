
# 初始化摄像头
import os
import shutil
import cv2
import time
import json
from skimage.measure import compare_ssim
import skimage
import argparse
import imutils
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def cv2ImgAddText(img, text, left, top, textColor, textSize):

    if isinstance(img, np.ndarray):
        img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    ttf = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

    font_style = ImageFont.truetype(ttf, textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=font_style)
    imgx = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return imgx

def save_notes(save_path, res_dict):
    with open(save_path, 'a') as f:
        f.write(time.strftime("%Y%m%d%H%M%S")+"|"+json.dumps(res_dict)+"\n")


def get_water_level(imageA, imageB):

    #imageA = cv2.imread("C:\\Users\\DEATH\\Desktop\\1.png")

    #imageB = cv2.imread("C:\\Users\\DEATH\\Desktop\\2.png")
    mask = cv2.imread("shuiweix.png")
    #cv2.imwrite("C:\\Users\\DEATH\\Desktop\\m.png",cv2.add(imageA,mask))
    #cv2.imshow("",cv2.add(imageA,mask))
    # convert the images to grayscale
    grayA = cv2.cvtColor(cv2.add(imageA,mask), cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(cv2.add(imageB,mask), cv2.COLOR_BGR2GRAY)
    #grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    #grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = skimage.metrics.structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))
    #print(len(diff))
    f=False
    i=0
    xiangsu=719
    dushu=0
    for h in range(230,719):
        if f:
            break
        for l in range(265,391):
            if diff[h][l]<128:
                xiangsu=h
                i=i+1
                #print(diff[h][l])
                #print(h)
                #print(l)
                f=True
                break

    if xiangsu<232:
        dushu=12
    elif xiangsu<=372:
        dushu=9.5+(372-xiangsu)/28*0.5
    elif xiangsu<=504:
        dushu=7+(504-xiangsu)/26.5*0.5
    elif xiangsu<=629:
        dushu=4.5+(629-xiangsu)/25*0.5
    elif xiangsu<=723:
        if xiangsu>718:
            dushu=0
        else:
            dushu=2.5+(723-xiangsu)/23.5*0.5
    #print(xiangsu)
    #print("dusu:"+str(dushu))
    return dushu


def water_detect(opt):
    parameters = opt.parameters
    levelUpper = parameters["levelUpper"]
    levelLower = parameters["levelLower"]
    print(opt)
    
    out = opt.output
    if opt.camera_id:
        out += "/{}".format(opt.camera_id)

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    #TRACKING_NUM = parameters["timelimit"]*24*60
    TRACKING_NUM = opt.record_length

    cap = cv2.VideoCapture(opt.source)#获取网络摄像机
    ret, frame = cap.read()
    # writeer
    fourcc = 'mp4v'  # output video codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sp = out + "/" + time.strftime("%Y%m%d%H%M%S") + "_" + opt.online_save_name
    vid_writer = cv2.VideoWriter(sp, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    i = -1
    frames={}
    frames[0]=frame
    #cv2.imwrite('new.png', frame)
    flag=True
    dushu=0
    IS_VIOLATION = False
    res = {"violation": False, "水位超标": False}
    while True:
        i += 1
        img_sp = out + "/" + opt.online_save_name + "_{}_.png".format(i)
        #time.sleep(0.5+random.random())
        ret, frame = cap.read()
        if flag:
            frames[1]=frame
        else :
            frames[0]=frame
        flag=bool(1-flag)
        # cv2.imshow("capture", frame)
        #print (flag)
        dushu=get_water_level(frames[0],frames[1])
        #dushu = 4.5
        if dushu <= float(levelUpper)/100.0:
            frame = cv2ImgAddText(frame, "当前水位值为{:.2f}.".format(dushu), 100, 500, (255, 0, 0), 50)
            #cv2.putText(frame, "The water level is: "+ "{:.2f}".format(dushu),(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        else:
            IS_VIOLATION = True
            res["水位超标"] = True
            res["violation"] = True
            # cv2.putText(frame, "WARNING: water level has reached {:.2f}.".format(dushu),(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            frame = cv2ImgAddText(frame, "警告：水位超过警戒线, 达{:.2f}.".format(dushu), 100, 500, (255, 0, 0), 50)
        #print(dushu)
        #把度数记录到指定得位置比如数据库或者某个文件中
        cv2.imshow("water level", frame)
        start_tracking = True if IS_VIOLATION else False
        if start_tracking :
            if TRACKING_NUM > 0:
                TRACKING_NUM -= 1
                #print(TRACKING_NUM)
                vid_writer.write(frame)
                save_notes(sp+".json", res)
            else:
                # TRACKING_NUM = parameters["timelimit"] * 24 * 60
                TRACKING_NUM = opt.record_length
                start_tracking = False
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                temp_cap = cv2.VideoCapture(opt.source)
                w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp_cap.release()
                fps = 24
                sp = out + "/" + time.strftime("%Y%m%d%H%M%S") + "_" + opt.online_save_name
                print(sp)
                vid_writer = cv2.VideoWriter(sp, fourcc, fps, (w, h))
        else:
            pass
        #print(img_sp)
        if i % 720 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_sp, frame)
        if cv2.waitKey(1) and 0xff == ord("q"):
            break
        #cv2.imwrite(str(i) + '.jpg', frame)# 存储为图像
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    cap.release()
    cv2.destroyAllWindows()
 

 
# 测试
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='http://183.221.111.158:10810/nvc/jmk/nvc/jmk/hls/stream_9/stream_9_live.m3u8', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/waterlevel', help='output folder')  # output folder
    parser.add_argument('--online_save_name', type=str, default="test.mp4", help='online video save path')
    parser.add_argument('--camera_id', type=str, default="offline", help='id of a camera')
    opt = parser.parse_args()

    water_detect(opt)

