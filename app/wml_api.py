import urllib3, requests, json
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### loading dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=13)

original_test_images = test_images

### normalizing dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
val_images = val_images.astype('float32') / 255


iam_token = 'eyJraWQiOiIyMDE5MDUxMyIsImFsZyI6IlJTMjU2In0.eyJpYW1faWQiOiJJQk1pZC01NTAwMDNOM1A5IiwiaWQiOiJJQk1pZC01NTAwMDNOM1A5IiwicmVhbG1pZCI6IklCTWlkIiwiaWRlbnRpZmllciI6IjU1MDAwM04zUDkiLCJnaXZlbl9uYW1lIjoiTW96YXJ0IiwiZmFtaWx5X25hbWUiOiJTaWx2YSIsIm5hbWUiOiJNb3phcnQgU2lsdmEiLCJlbWFpbCI6Im1vemFydC52c2lsdmFAZ21haWwuY29tIiwic3ViIjoibW96YXJ0LnZzaWx2YUBnbWFpbC5jb20iLCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiJmZDg4ODYxZDlkNDg0ODI5YTY3ZDljMmQ0NGVjNTYxOCJ9LCJpYXQiOjE1NjYwNDI1NzgsImV4cCI6MTU2NjA0NjE3OCwiaXNzIjoiaHR0cHM6Ly9pYW0uY2xvdWQuaWJtLmNvbS9pZGVudGl0eSIsImdyYW50X3R5cGUiOiJ1cm46aWJtOnBhcmFtczpvYXV0aDpncmFudC10eXBlOmFwaWtleSIsInNjb3BlIjoiaWJtIG9wZW5pZCIsImNsaWVudF9pZCI6ImRlZmF1bHQiLCJhY3IiOjEsImFtciI6WyJwd2QiXX0.lvgNPsSJ60-NsrCuAWgAJO4EfMKqvUIj6fuuC3L2b-KdrfDghBM8Rz2EHqXyHY6FjA0ilm-icZrEfdG_Z7yI2ThEXegdVAfRE-tm7zxtfwJBLWmaIzHaThCpxRIEl4NnyyiwSrxheCFFHbel-snpkr4rAD8_8sBr8yDpiSlxog8d98tXEbM2g1Czl1eWsMkLLjSUULI-tJyQwJNGmBNfxdyUXXB1Vm1jiofs1LY5CwePFrZzQgXcS0Fq-MlOMCE6twTrkZQZttN5YfvytvXmVKgGew68bDYxWndadRgC3IWwlF0R5ddvhR9QFIZF5p8-HOCW5BytEMCl-zfkJOz9QQ'
ml_instance_id = 'f08120b8-fdb6-43c5-96a8-38f71bc1e643'

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + iam_token, 'ML-Instance-ID': ml_instance_id}

payload_scoring = {"values": [test_images[1].tolist()]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/v3/wml_instances/f08120b8-fdb6-43c5-96a8-38f71bc1e643/deployments/a8598589-de5a-43a3-b56c-279ffc1da781/online', json=payload_scoring, headers=header)
print("Scoring response")
print(json.loads(response_scoring.text))

object = json.loads(response_scoring.text)
prediction = object['values'][0][0]
classif = object['values'][0][1]
probability = object['values'][0][2]
print('prediction: ', np.max(prediction))
print('prediction_classes: ', classif)
print('probability: ', np.max(probability))


plt.figure()
plt.imshow(original_test_images[1])
plt.grid(False)
plt.show()
