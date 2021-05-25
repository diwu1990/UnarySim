import numpy as np

np.random.seed(0)
index = np.array(range(0, 16))
np.random.shuffle(index)
print(index)


np.random.seed(0)
def func():
    index = np.array(range(0, 16))
    np.random.shuffle(index)
    print(index)

func()
