from torchvision import datasets
from torchvision.transforms import ToTensor
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

train_data = datasets.MNIST( root = 'data',
train = True, transform = ToTensor(), download = True,
)

print(train_data)

test_data = datasets.MNIST( root = 'data',
train = False,
transform = ToTensor() )

print(test_data)

print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(train_data.data.min(), train_data.data.max()))

print('The first training image shape: {}'.format(train_data.data[0].shape))
