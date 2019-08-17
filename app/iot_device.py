import wiotp.sdk.device
import random
import time

def commandCallback(cmd):
    print('Command Received: %s' % cmd.data)

config = {
    'identity' : {
        'orgId' : '9z1a51',
        'typeId' : 'Maratona',
        'deviceId' : 'm-01'
    },
    'auth' : {
        'token' : 'qadr6C-47&2kt)r_u6'
    }
}

client = wiotp.sdk.device.DeviceClient(config=config)
client.connect()

while True:
    data = {'pressao' : random.randint(20, 50) }
    client.publishEvent(eventId='pressao', msgFormat='json', data=data, qos=0, onPublish=None)
    client.commandCallback = commandCallback
    time.sleep(2)
