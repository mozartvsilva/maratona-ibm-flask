from flask import Flask, request
import requests
import final_iot as iot
import final_wml as wml

app = Flask(__name__)


@app.route('/get', methods=['GET'])
def get():
    url = 'https://jsonplaceholder.typicode.com/posts'
    headers = {
        'content-type': 'text'
    }
    r = requests.get(url, headers)
    posts = {'posts': r.json()}
    print(posts)
    return posts


@app.route('/post', methods=['POST'])
def post():
    payload = request.get_json()
    print('payload', payload)

    url = 'https://jsonplaceholder.typicode.com/posts'
    r = requests.post(url, data=payload)
    print('response', r.json())
    return r.json()


@app.route('/put/<int:post_id>', methods=['PUT'])
def put(post_id):
    payload = request.get_json()
    print('payload', payload)

    url = 'https://jsonplaceholder.typicode.com/posts/%d' % post_id
    r = requests.put(url, data=payload)
    print('response', r.json())
    return r.json()
