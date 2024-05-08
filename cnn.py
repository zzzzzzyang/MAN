import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from resnet import *
import pickle
import os


       
def resModel(args, device): #resnet18
    
    model = resnet18(end2end= False,  pretrained= False, num_class=args.num_class).to(device)
    
    if  args.pretrained_backbone_path:
       
        checkpoint = torch.load(args.pretrained_backbone_path, map_location=device)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()

        for key in pretrained_state_dict:
            if  ((key == 'fc.weight') | (key=='fc.bias') | (key=='feature.weight') | (key=='feature.bias') ) :
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]

        model.load_state_dict(model_state_dict, strict = False)
        print('Model loaded from Msceleb pretrained')
    else:
        print('No pretrained resent18 model built.')
    return model   


