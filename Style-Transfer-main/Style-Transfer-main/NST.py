import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.models as models 
from torchvision.utils import save_image

vgg19_pretrained = models.vgg19(pretrained=True).features

layer_list = ['0','5','10','19','28']

class VGG(nn.Module) : 
    def __init__(self,layer_list=['0','5','10','19','28']) : 
        super(VGG,self).__init__()

        self.chosen_feature_layes = layer_list
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self,x) : 
        features = [ ]

        for layer_id, layer in enumerate(self.model) : 
            x = layer(x)

            if str(layer_id) in self.chosen_feature_layes : 
                features.append(x)

        return features 

