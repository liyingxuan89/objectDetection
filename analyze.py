import argparse
import torch.backends.cudnn as cudnn
import numpy as np
from utils import google_utils
from utils.datasets import *
from utils.utils import *


def detect(opt):

    out, source, weights, imgsz = \
        opt.output, opt.source, opt.weights, opt.img_size
    parameters = opt.parameters
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # TRACKING_NUM = parameters["timelimit"] * 24 * 60
    TRACKING_NUM = opt.record_length
    save_img, view_img, save_txt = False, False, False

    # get scenario
    scenario = Path(opt.output).name
    if scenario == 'beam':
        TRACKING_NUM = 1440 * 20

    # Initialize
    device = torch_utils.select_device(opt.device)
    #print(device)
    if opt.camera_id:
        out += "/{}".format(opt.camera_id)

    if os.path.exists(out):
        shutil.rmtree(out, ignore_errors=True)  # delete output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float().eval()  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    # if scenario in {"damper"}:
    #     classify = True
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101_damper.pt', map_location=device))  # load weights
        modelc.to(device).eval()

    status_classify = True if scenario in ["damper"] else False

    if status_classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101_damper.pt', map_location=device))  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
        temp_cap = cv2.VideoCapture(source)
        w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
        fps = 24
        sp = out + "/" + time.strftime("%Y%m%d%H%M%S") + "_" + opt.online_save_name
        print(sp)
        stream_writer = cv2.VideoWriter(sp, fourcc, fps, (w, h))

    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    Tracking = TRACKING_NUM
    Start_tracking = False
    for idx, data in enumerate(dataset):  # path;fix size img;origin img;the video capture if there are video file
        path, img, im0s, vid_cap = data
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32, half precise computation to accelerate speed;
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) # pred: xyxy,prob,class
        t2 = torch_utils.time_synchronized()

        if scenario in ["rock"] :
            pred = [x[x[:, 4] > 0.6] for x in pred if x is not None]

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        if status_classify:
            pred = status_classifier(pred, modelc, img, im0s)

        # Process detections
        IS_VIOLATION = False
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                res, det, im0 = violation_detection(im0, scenario=scenario, detections=det, names=names, parameters=parameters)
                # print(res)
                IS_VIOLATION = res["violation"]
                if scenario == "beam":
                    if res["工人"] or res["试探员"]:
                        IS_VIOLATION = res["violation"]

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        #im0 = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

            if det is None or len(det) == 0:
                if scenario == "sensor":
                    IS_VIOLATION = True
                    res = {"violation": True, "传感器位置错误": False, "无传感器": False, "传感器数目不足": False,
                           "传感器离顶过近": False, "传感器离墙过近": False, "传感器悬挂过低": False,
                           }
                elif scenario == "Huge_rock":
                    IS_VIOLATION = False
                    res = {"violation": False, "大型煤块": False}

                elif scenario == "beam":
                    IS_VIOLATION = False
                    res = {"violation": False, "前探梁满足": False, "前探梁": False,
                           "工人": False, "试探员": False, "液压机": False}

                elif scenario == "damper":
                    IS_VIOLATION = True
                    res = {"violation": True, "没有风门": True, "风门打开": True}


            if IS_VIOLATION:
                Start_tracking = True

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                if Start_tracking:
                    if Tracking > 0:
                        Tracking -= 1
                        stream_writer.write(im0)
                        save_notes(sp+".json", res)
                    else:
                        Tracking = TRACKING_NUM
                        Start_tracking = False
                        if isinstance(stream_writer, cv2.VideoWriter):
                            stream_writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        temp_cap = cv2.VideoCapture(source)
                        w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        temp_cap.release()
                        fps = 24
                        sp = out + "/" + time.strftime("%Y%m%d%H%M%S") + "_" + opt.online_save_name
                        print(sp)
                        stream_writer = cv2.VideoWriter(sp, fourcc, fps, (w, h))
                else:
                    pass
                if idx % 720 == 0:
                    cv2.imwrite(save_path+"_{}_.png".format(idx), im0)
                # print("Tracking : ", Tracking)
                # print("idx : ", idx)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        #fourcc = 'mp4v'  # output video codec
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        fps = 24
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    if IS_VIOLATION:
                        vid_writer.write(im0)
                        save_notes(save_path+".json", res)
                        if idx % 720 == 0:
                            cv2.imwrite(save_path+"_{}_.png".format(idx), im0)
                        #IS_VIOLATION = False
                    #if evidence_vid_path != evidence_save_path:  # new evidence piece
                    # evidence_vid_path = evidence_save_path
                    # if isinstance(evidence_vid_writer, cv2.VideoWriter):
                    #     evidence_vid_writer.release()
                    # fourcc = 'mp4v'  # output video codec
                    # #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    # fps = 24
                    # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # evidence_vid_writer = cv2.VideoWriter(evidence_save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    #IS_VIOLATION:
                    # evidence_vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best_yolov5x_beam.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images/beam', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='http://183.221.111.158:10810/nvc/jmk/nvc/jmk/hls/stream_1/stream_1_live.m3u8', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/beam', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--record_length', type=int, default=1440*10, help='video recording length by frames')
    parser.add_argument('--online_save_name', type=str, default="test.mp4", help='online video save path')
    parser.add_argument('--camera_id', type=str, default="offline", help='id of a camera')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    opt.parameters = {"numLower": 4}
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect(opt)