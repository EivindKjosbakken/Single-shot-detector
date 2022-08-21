import torch
from typing import Tuple, List
from torch import nn

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        num_filters = 64 
        self.layer1 = nn.Sequential(
            #layer 1
            nn.Conv2d(
                in_channels = image_channels,
                out_channels=num_filters,
                kernel_size= 3,
                stride = 1,
                padding = 1, 
            ),
            nn.BatchNorm2d(num_filters), #applying batch norm for faster convergence
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels = num_filters,
                out_channels=num_filters*2,
                kernel_size= 3,
                stride = 1,
                padding = 1, 
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels = num_filters*2,
                out_channels=num_filters*2,
                kernel_size= 3,
                stride = 1,
                padding = 1, 
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(), 
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
            ),       
            nn.ReLU(), 
            )

        
        self.layer2 = nn.Sequential(
            #layer2:
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),)
        
        self.layer3 = nn.Sequential(
            #layer 3:
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[1],
                out_channels=num_filters*8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*8,
                out_channels=output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),)

        self.layer4 = nn.Sequential(
            #layer 4:
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[2],
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_filters*4), #endrer
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),)

        self.layer5 = nn.Sequential(
            #layer 5:
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[3],
                out_channels=num_filters*4,
                kernel_size=3, 
                stride=1,
                padding=1, 
            ),
            nn.BatchNorm2d(num_filters*4), #endrer
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=output_channels[4],
                kernel_size=3, 
                stride=2,
                padding=1, 
            ),
            nn.ReLU(),)

        self.layer6 = nn.Sequential(
            #layer 6:
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=num_filters*4,
                kernel_size=3, 
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_filters*4), #endrer
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        #iterating through each layer, taking output from previous layer, as input to next layer
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out_features.append(out1)
        out_features.append(out2)
        out_features.append(out3)
        out_features.append(out4)
        out_features.append(out5)
        out_features.append(out6)

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

