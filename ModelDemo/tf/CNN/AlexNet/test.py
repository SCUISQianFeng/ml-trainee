
import sys
sys.path.append('..')

from alexnet import AlexNet

if __name__ == "__main__":
    model = AlexNet(input_shape=(224, 224, 3), classes=1000)
    model.summary()