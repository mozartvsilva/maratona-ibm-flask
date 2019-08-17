import requests
import json

body = {
    # 'id' : 122451,
    # 'desafio' : 9,
    # 'cpf' : '03779301482'
}

print('body', body)
headers= {
}
print('headers', headers)

url = 'https://8d829621.us-south.apiconnect.appdomain.cloud/desafios/desafio9'
r = requests.post(url, data=body, headers=headers)
# ? json.dumps(body)

print('response status', r.status_code)
print('response', r.json())