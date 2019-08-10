from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def hello_world():
    headers = {
      'content-type': 'text'
    }
    r = requests.get('https://newsapi.org/v2/top-headlines?sources=google-news&apiKey=8049e496881049d3be91d4b7d5dfc753', headers)
    news = r.json()
    print(news)
    return news
