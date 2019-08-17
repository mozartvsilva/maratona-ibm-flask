import wiotp.sdk.application
import wiotp.sdk.device
import json

def deviceEventCallback(event):
    str = '%s event %s received from device [%s]: %s'
    print(str % (event.format, event.eventId, event.device, json.dumps(event.data)))
    if event.data['pressao'] > 30:
        commandData = {'cmd' : 'just do it' }
        client.publishCommand('Maratona', 'm-01', 'reboot', 'json', commandData)

config = {
    'auth' : {
        'key' : 'a-9z1a51-kzuytqhs3c',
        'token' : 'N*r*3_onuyxP_d@T1y'
    }
}

client = wiotp.sdk.application.ApplicationClient(config=config)
client.connect()
client.subscribeToDeviceEvents()

while True:
    client.deviceEventCallback = deviceEventCallback