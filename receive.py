import pika
import yaml
import json
import shlex
import subprocess

username = "admin"
psword = "gshl@2019.rabbitmq"
credentials = pika.PlainCredentials(username, psword)

connection =pika.BlockingConnection(pika.ConnectionParameters(host='106.55.43.81', port=5672, credentials=credentials))
channel = connection.channel()

channel.queue_declare('algorithm_test', durable=True)


def getConfig(path):

    if isinstance(path, str):
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)
            return data_dict

    else:
        raise ValueError("Enter the right path and retry.")
    


def getCommand(scenario, source):
    print(scenario)
    print(source)
    
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


def consume(message):

    for each_m in message:
        print("current alarm:", each_m)
        scenario = each_m["alarms"][0]["typeNameEn"]
        #conf = each_m["conf"]
        source = each_m['liveUrl']
        cmd = getCommand(scenario, source)
        print(cmd)
        res = subprocess.check_output(shlex.split(cmd))
        print(res)

    return 


def callback(ch, method, properties, body):

    body = body.decode('utf8')
    #print(body)
    try:
        message = json.loads(body)
        print(' [x] received %r' % message)
    except Exception as e:
        print(e)
        print("Please fill in message in Json fromat.")
    res = consume(message)
    return 


channel.basic_consume(on_message_callback=callback, queue="algorithm_test", auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

