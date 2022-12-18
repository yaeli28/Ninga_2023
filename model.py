import torch
import os

model = torch.hub.load('/home/lab-maker/Desktop/yolov5', 'custom', path='last.pt', source = 'local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', source = 'local')
#model = torch.hub.load(os.getcwd(), 'custom', source = 'local', path = 'last.pt', force_reload = True)
# model = torch.hub.load('ultralytics/yolov5', 'deeplabv3_resnet101', pretrained = False)


model.conf = 0.7
