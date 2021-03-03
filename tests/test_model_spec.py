from models.utils import count_parameters
from models.mobilenetv3 import MobileNetV3

def test_model_size():
    mbv3 = MobileNetV3(6, 256, )
    print(count_parameters(mbv3))
    # assert count_parameters(mbv3) < 1000000