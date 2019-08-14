import numpy as np

print('>> arange(6)')
a = np.arange(6)
print(a)

print('>> argmax')
print(np.argmax(a))

print('>> max')
print(np.max(a))

print('>> reshape')
b = a.reshape(2, 3) + 10
print(a, '->', b)

print('>> expand_dims')
c = np.expand_dims(a, 0)
print(a, '->', c)
