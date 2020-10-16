import numpy as np
import pika
import cv2
import yaml
import json
import uuid
import shlex
import subprocess
from minio import Minio
from minio.error import ResponseError
from requests_toolbelt import MultipartEncoder
import requests
import os
import threading
import psutil
import time
from shutil import copyfile
from utils.utils import parse_violation_res
import argparse
from analyze import detect
from shuiwei import water_detect
import redis
from kazoo.client import KazooClient
import ctypes
import inspect


class Loadbalance:
    """
    负载
    """

    def __init__(self,
                 status_health_address=None,
                 message_ip='127.0.0.1',
                 message_port=6379,
                 message_password=None,
                 message_topic='nodeEvent'):
        self.__node_id = str(uuid.uuid4())
        self.__status_health_address = status_health_address
        self.__message_ip = message_ip
        self.__message_port = message_port
        self.__message_password = message_password
        self.__message_topic = message_topic

    def start(self, on_message):
        # 启动节点状态
        root_path = '/algorithm-server'
        zk = KazooClient(hosts=self.__status_health_address)
        zk.start()  # 与zookeeper连接
        zk.create(path=root_path + '/' + self.__node_id, value=b'', ephemeral=True, sequence=False)
        rc = redis.StrictRedis(self.__message_ip, self.__message_port, db=0, password=self.__message_password)
        ps = rc.pubsub()
        ps.subscribe(self.__message_topic)
        for item in ps.listen():
            if item['type'] == 'message':
                if on_message is not None:
                    node_message = json.loads(item['data'])
                    node_id = node_message['nodeId']
                    if node_id == self.__node_id:
                        on_message(node_message['action'], node_message['cameraInfo'])


def getConfig(path):

    if isinstance(path, str):
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
            return data_dict

    else:
        raise ValueError("Enter the right path and retry.")


def getCommand(scenario, source):
    print(scenario + " of source : " + source)
    
    if scenario == "waterlevel":
        cmd = "python3 shuiwei.py"
        cmd += " --output {}{} ".format("inference/output/", scenario)
        cmd += " --source {} ".format(source)
        return cmd
        
    cmd = "python3 detect.py "
    cfg = getConfig('data/conf.yaml')

    if "weights" in cfg:
        cmd += "--weights {} ".format(cfg["weights"][scenario])
    if "output" in cfg:
        cmd += "--output {} ".format(cfg["output"][scenario])
    if "img-size" in cfg:
        cmd += "--img-size {} ".format(cfg["img-size"])
    cmd += "--source {} ".format(source)
    return cmd


class MinioObjectUtils:
    '''
     对象存储工具类

     工具类依赖 minio的SDK：
        1：pip下载： pip install minio
        2：使用源码安装
            git clone https://github.com/minio/minio-py
            cd minio-py
            python setup.py install
     此类功能有：
        1： 下载对象到本地，方法名称：save_as_local.
        2： 读取对象数据：方法名称：read_object_as_datas.
        3： 读取对象片段数据，方法名称：read_partial_object.
        4： 删除对象：remove_object
    '''

    def __init__(self,
                 endpoint='183.221.111.158:9000',
                 access_key='admin',
                 secret_key='a23529ea43483a65d506b8fe5e40b1ca',
                 secure=False):
        '''
        此参数默认成都开发环境参数。

        :param endpoint: 对象存储服务endpoint。
        :param access_key: 对象存储的Access key。（如果是匿名访问则可以为空）。
        :param secret_key: 对象存储的Secret key。（如果是匿名访问则可以为空）。
        :param secure: 设为True代表启用HTTPS。 (默认是True)。
        '''
        self.__endpoint = endpoint
        self.__access_key = access_key
        self.__secret_key = secret_key
        self.__secure = secure
        self.__minio_Client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

    def read_object_as_datas(self, bucket_name, object_name, data_hook):
        '''
        从存储读取文件数据。
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :param data_hook: 数据回写函数
        :return:
        '''
        try:
            data = self.__minio_Client.get_object(bucket_name, object_name)
            for d in data.stream(32 * 1024):
                if data_hook is not None:
                    data_hook(d)
        except ResponseError as err:
            raise err

    def read_partial_object(self, bucket_name, object_name, offset=0, length=0, data_hook=None):
        '''
        下载一个对象的指定区间的字节数组。
        :param bucket_name: 存储桶名称
        :param object_name: 对象名称
        :param offset: 开始位置
        :param length: 长度
        :return:
        '''
        try:
            data = self.__minio_Client.get_partial_object(bucket_name, object_name, offset=offset, length=length)
            for d in data:
                if data_hook is not None:
                    data_hook(d)
        except ResponseError as err:
            raise err

    def save_as_local(self, bucket_name, object_name, file_path):
        '''
        下载并将文件保存到本地。
        :param bucket_name: 存储桶名称
        :param object_name: 文件名称
        :param file_path: 本地文件保存路径，如果路径不存在则会自动新建。
        :return:
        '''
        if file_path is None:
            raise Exception('文件路径不能为空')
        try:
            self.__minio_Client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=file_path)
        except ResponseError as err:
            raise err

    def remove_object(self, bucket_name, object_name):
        '''
        删除对象
        :param bucket_name:
        :param object_name:
        :return:
        '''
        try:
            self.__minio_Client.remove_object(bucket_name, object_name)
        except ResponseError as err:
            raise err


class StoreUtils:
    '''
        小对象文件存储
        依赖： requests_toolbelt,requests
              pip install requests_toolbelt
              pip install requests
    '''
    @staticmethod
    def upload(url='http://183.221.111.158:27000/repositories/storages', filename=None):
        '''
        文件上传
        :param url:
        :param filename:
        :return:
        '''
        if url is None:
            raise Exception('上传对象时，url不能为空')
        if filename is None:
            raise Exception('上传对象不能为空')
        if not os.path.isfile(filename):
            raise Exception('上传对象不是任何的文件')
        if not os.path.exists(filename):
            raise Exception('上传对象在本地不存在')
        data = MultipartEncoder(fields={'file': ('filename', open(filename, 'rb'), 'text/xml')})
        response = requests.post(url=url, data=data, headers={'Content-Type': data.content_type})
        status_code = response.status_code
        if status_code == 200:
            return response.text
        raise Exception('上传失败')


def kill_camera(url):
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if url in p.cmdline():
            p.terminate()


def wait_to_reconnect_camera(url):
    while True:
        if cv2.VideoCapture(url).read()[0]:
            print("re-connected to camera...")
            return True 
        else:
            print("Waiting to re-connect to camera {}".format(url))
            time.sleep(10)


class Opt:

    def __init__(self):
        self.name = "configures"
        self.opt = argparse.ArgumentParser()
        self.opt.add_argument('--record_length', type=int, default=14400, help='video recording length by frames')
        self.opt.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        self.opt.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
        self.opt.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
        self.opt.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.opt.add_argument('--view_img', action='store_true', help='display results')
        self.opt.add_argument('--save_txt', action='store_true', help='save results to *.txt')
        self.opt.add_argument('--classes', nargs='+', type=int, help='filter by class')
        self.opt.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.opt.add_argument('--augment', action='store_true', help='augmented inference')
        self.opt.add_argument('--update', action='store_true', help='update all models')
        self.cfg = self.opt.parse_args()

def judge(opt):

    live_url, scenario, camera_id, online_save_name, record_length, output, alarm_id =\
        opt.source, opt.scenario, opt.camera_id, opt.online_save_name, opt.record_length, opt.output, opt.alarm_id
    parameters = opt.parameters

    time.sleep(10)

    print("Computations on rail. ")
    # uuid_m = uuid.uuid1()
    uuid_queue = []
    while True:
        # camera offline break loop
        if not cv2.VideoCapture(live_url).read()[0]:
            break

        url = 'http://183.221.111.158:27000/repositories/storages'
        try:
            files_to_be_upload = os.listdir("inference/output/{}/{}".format(scenario, camera_id))
        except Exception as e:
            print(e)
            time.sleep(20)

        # if cv2.VideoCapture("")
        files_to_be_upload = os.listdir("inference/output/{}/{}".format(scenario, camera_id))
        store_ids = []
        picks = [x for x in files_to_be_upload if x.endswith(".png")]
        movs = [x for x in files_to_be_upload if x.endswith(".mp4")]
        jns = [x for x in files_to_be_upload if x.endswith(".json")]
        picks = sorted(picks)
        movs = sorted(movs)
        jns = sorted(jns)
        print(movs)
        s = ""
        if len(movs) <= 1:
            time.sleep(10)  # less than the length of a single video

        elif len(movs) > 1:

            # upload pictures
            if len(picks) >= 1 and len(jns) >=1:
                pic = picks[0]
                jn = jns[0]
                info = parse_violation_res("inference/output/{}/{}/{}".format(scenario, camera_id, jn), scenario=scenario, parameters=parameters)
                info = json.loads(info)
                # print(info)
                VIOLATION_FLAG = info["violation"]
                # info = {"violation": True}
                # VIOLATION_FLAG = "yes"
                uuid_m = uuid.uuid1()
                if VIOLATION_FLAG == "yes":
                    store_id = StoreUtils.upload(url=url,
                                                 filename="inference/output/{}/{}/{}".format(scenario, camera_id, pic))
                    store_ids.append(store_id)
                    Umessage = {
                        "cameraId": camera_id,
                        "imageId": store_id,
                        "response": [
                            {
                                "alarmId": alarm_id,
                                "videoCode": str(uuid_m),
                                "msg": "{}".format(info)
                            }
                        ]
                    }
                    channel.basic_publish(exchange="micro-alarm-algorithm", routing_key="micro-algorithm-response",
                                          body=json.dumps(Umessage))
                    uuid_queue.append(uuid_m)
                    print("published U_message of picks.")
                for each in picks[:-1]:
                    p = "inference/output/{}/{}/{}".format(scenario, camera_id, each)
                    os.remove(p)
                    print("remove picture {}".format(p))
                for each in jns[:-1]:
                    p = "inference/output/{}/{}/{}".format(scenario, camera_id, each)
                    os.remove(p)
                    print("remove json {}".format(p))

            # upload video
            for filename in movs[:-1]:
                cur_file = "inference/output/{}/{}/{}".format(scenario, camera_id, filename)
                VIOLATION_FLAG = "yes"
                if VIOLATION_FLAG == "yes":
                    cur_file_encoded = cur_file.replace(".mp4", "_encode.mp4")
                    try:
                        return_code = subprocess.check_output(
                            "ffmpeg -i {} -vcodec libx264 {}".format(cur_file, cur_file_encoded), shell=True)
                    except Exception as e:
                        print(e)
                        os.remove(cur_file)
                        continue

                    # print(return_code)
                    print("------------------------------------------------------------------")
                    print("get h264 video")
                    os.remove(cur_file)
                    print("removed old version video {}".format(cur_file))
                    store_id = StoreUtils.upload(url=url, filename=cur_file_encoded)
                    store_ids.append(store_id)
                    os.remove(cur_file_encoded)
                    print("removed h264 video {}".format(cur_file_encoded))
                    if len(uuid_queue) > 0:
                        for i in range(len(uuid_queue)):
                            Umessage = {"videoCode": str(uuid_queue.pop()), "videoUrl": store_id}  # upload message
                            print("published message of video.")
                            channel.basic_publish(exchange="micro-alarm-algorithm", routing_key="micro-algorithm-video",
                                                  body=json.dumps(Umessage))
                    # print(store_id)
                else:
                    os.remove(cur_file)
        elif len(movs) > 1 and len(jns) == 0:
            for filename in movs[:-1]:
                cur_file = "inference/output/{}/{}/{}".format(scenario, camera_id, filename)
                os.remove(cur_file)
        else:
            time.sleep(60)

    return


def mission(message, scenario):
    #scenario = message["alarms"][0]["typeNameEn"]
    alarm_id = message["alarmInfo"]["alarmId"]
    camera_id = message["cameraId"]
    live_url = message["liveUrl"]
    params = message["alarmInfo"]["params"]
    parameters = {x["paramsNameEn"]: x["paramsValue"] for x in params} if len(params) > 0 else {}
    print(parameters)
    print(live_url)
    print(scenario)
    #minio_utils = MinioObjectUtils()

    opt = Opt().cfg
    opt.source = live_url
    opt.scenario = scenario
    opt.output = "inference/output/{}".format(scenario)
    opt.online_save_name = "{}.mp4".format(camera_id)
    opt.camera_id = camera_id
    opt.alarm_id = alarm_id
    opt.parameters = parameters

    if scenario != "waterlevel":
        opt.weights = "weights/best_yolov5x_{}.pt".format(scenario)

    print("start detecting for {}.".format(scenario))
    if scenario == "waterlevel":
        main_mission = threading.Thread(target=water_detect, args=(opt,))
    else:
        main_mission = threading.Thread(target=detect, args=(opt,))
    judge_mission = threading.Thread(target=judge, args=(opt,))
    main_mission.start()
    judge_mission.start()
    # main_mission.join()
    # judge_mission.join()
    print("mission of {} at {} started.".format(scenario, camera_id))
    while True:
        if judge_mission.is_alive() is False:
            # main_mission.pause()
            # judge_mission.pause()
            print("camera {} offline.".format(camera_id))
            wait_to_reconnect_camera(live_url)
            time.sleep(30)
            if scenario == "waterlevel":
                main_mission = threading.Thread(target=water_detect, args=(opt,))
            else:
                main_mission = threading.Thread(target=detect, args=(opt,))
            judge_mission = threading.Thread(target=judge, args=(opt,))
            main_mission.start()
            judge_mission.start()
            #main_mission.join()
            #judge_mission.join()
        else:
            print("judge mission is alive.")
            time.sleep(30)


def callback(ch, method, properties, body):

    body = body.decode('utf8')
    print("into callback process ...")
    try:
        message = json.loads(body)
        print(' [x] received %r' % message)
        with open("data/camera_online_message.json", 'w') as f:
            f.write(json.dumps(message))
    except Exception as e:
        print(e)
        print("Please fill in message in Json fromat.")
        print("Loading message from local backup file of latest message.")
        try:
            with open("data/camera_online_message.json", "r") as f:
                message = json.loads(f.read())
        except:
            print("Fail to load message from local. Try it later.")
            return

    scenarios = [x["alarmInfo"]["typeNameEn"] for x in message]
    threads = []
    thread_dict = {}

    for i, s in enumerate(scenarios):
        t = threading.Thread(target=mission, args=(message[i], s))
        threads.append(t)
        thread_dict[t.__dict__["_name"]] = t.__dict__["_args"]
        t.start()

    # for i in range(len(threads)):
    #     threads[i].join()

    while True:
        for th in threads:
            if th.is_alive() is False:
                threads.remove(th)
                result = thread_dict.pop(th.name)
                new_th = threading.Thread(target=mission, args=result)
                threads.append(new_th)
                thread_dict[new_th.__dict__["_name"]] = new_th.__dict__["_args"]
                new_th.start()
        # connection.process_data_events()
        print("push 1 heart beat.")
        time.sleep(30)

    return



def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class MyThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(MyThread, self).__init__(*args, **kwargs)
        self.__stop = threading.Event()

    def stop(self):
        self.__stop.set()

    def stopped(self):
        return self.__stop.is_set()

    def run(self):
        print("begin to run the child thread.")
        while True:
            try:
                if self._target:
                    self._target(*self._args, **self._kwargs)
            finally:
                del self._target, self._args, self._kwargs

            if self.stopped():
                break




def on_message(action, camera_info):
    print(action, camera_info)
    print("into callback process...")
    scenario = camera_info['alarmInfo']['typeNameEn']
    camera_id = camera_info['cameraId']

    if action == "start":
        print("start mission for {}. ".format(camera_id))
        if camera_id not in threads_pool:
            t = threading.Thread(target=mission, args=(camera_info, scenario))
            threads_pool[camera_id] = t
            t.start()
            print(threads_pool)
        else:
            print(threads_pool)
            pass

    if action == "stop":
        if camera_id not in threads_pool:
            print("mission for {} doesn't exist.".format(camera_id))
            print(threads_pool)
            pass
        else:
            print("stops mission for {}.".format(camera_id))
            stop_thread(threads_pool[camera_id])
            threads_pool.pop(camera_id)
            print(threads_pool)

    if action == "update":
        print("update mission for {}".format(camera_id))
        if camera_id not in threads_pool:
            t = threading.Thread(target=mission, args=(camera_info, scenario))
            threads_pool[camera_id] = t
            t.start()
            print(threads_pool)
        else:
            stop_thread(threads_pool[camera_id])
            threads_pool.pop(camera_id)
            t = threading.Thread(target=mission, args=(camera_info, scenario))
            threads_pool[camera_id] = t
            t.start()
            print(threads_pool)


if __name__ == "__main__":


    threads_pool = {}

    loadbalance = Loadbalance(
        status_health_address='106.55.43.81:2181',
        message_ip='106.55.43.81',
        message_password='gshl@2019.redis')
    loadbalance.start(on_message=on_message)

    # username = "admin"
    # psword = "gshl@2019.rabbitmq"
    # credentials = pika.PlainCredentials(username, psword)
    #
    # connection = pika.BlockingConnection(
    #     pika.ConnectionParameters(host='106.55.43.81', port=5672, credentials=credentials))
    # channel = connection.channel()
    #
    # channel.queue_declare('micro-camera-list', durable=True)
    # channel.basic_consume(on_message_callback=callback, queue="micro-camera-list", auto_ack=True)
    #
    # print(' [*] Waiting for messages. To exit press CTRL+C')
    # while True:
    #     try:
    #         channel.start_consuming()
    #     except Exception as e:
    #         print(e)
    #         connection = pika.BlockingConnection(
    #             pika.ConnectionParameters(host='106.55.43.81', port=5672, credentials=credentials))
    #         channel = connection.channel()
    #         channel.queue_declare('micro-camera-list', durable=True)
    #         channel.basic_consume(on_message_callback=callback, queue="micro-camera-list", auto_ack=True)
    #         print("rebuilding channel...")
    #         channel.start_consuming()
    #         time.sleep(10)
    #         continue
    #     time.sleep(30)


