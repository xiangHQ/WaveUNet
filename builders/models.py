from builders.model.ResUNet import ResUNet
from builders.model.VURNet import VURNet
from builders.model.DLPUNet import DLPUNet
from builders.model.WaveUNet import WaveUNet

def creat_model(model_name):
    if model_name == 'ResUNet':
        return ResUNet()
    elif  model_name == 'WaveUNet':
        return WaveUNet()
    elif  model_name == 'DLPUNet':
        return DLPUNet()
    elif  model_name == 'VURNet':
        return VURNet()