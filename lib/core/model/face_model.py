import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

import timm

class Net(nn.Module):
    def __init__(self, model_name, pretrained, num_classes=1000):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        # self.model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, exportable=True)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self._fc = nn.Linear(1024 , num_classes, bias=True)

    def forward(self, inputs):

        #do preprocess


        bs = inputs.size(0)
        # Convolution layers
        x = self.model.forward_features(inputs)

        fm = x.view(bs, -1)
        x = self._fc(fm)

        return x



if __name__=='__main__':
    import torch
    import torchvision

    dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
    model = Net(model_name="tf_mobilenetv3_small_075", pretrained=True)

    ### load your weights
    model.eval()
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.


    torch.onnx.export(model, dummy_input, "classifier.onnx",opset_version=11 )
