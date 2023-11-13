import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

class AutoEncoderHPyloris(nn.Module):
    def __init__(self):
        super(AutoEncoderHPyloris, self).__init__()
        
        """
        The proposed autoencoder has 3 convolutional blocks with one convolutional layer,
        batch normalization and leakyrelu activation each. The size of the convolutional
        kernel is 3 and the number of neurons and stride of each layer are, respectively,
        [32,64,64] and [1,2,2].
        """
        
        self.encoder =  nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Convolutional Layer 1
                        nn.BatchNorm2d(32),  # Batch Normalization 1
                        nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU Activation 1
                       
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Convolutional Layer 2
                        nn.BatchNorm2d(64),  # Batch Normalization 2
                        nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU Activation 2

                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Convolutional Layer 3
                        nn.BatchNorm2d(128),  # Batch Normalization 3
                        nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU Activation 3
                        
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Convolutional Layer 3
                        nn.BatchNorm2d(256),  # Batch Normalization 3
                        nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU Activation 3
                        
                        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Convolutional Layer 3
                        nn.BatchNorm2d(256),  # Batch Normalization 3
                        nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU Activation 3
        )
        
        self.decoder =  nn.Sequential(
                        nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
            
                        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        
                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        
                        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.2, inplace=True),
                        
                        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, image):
        encoded_img = self.encoder(image)
        decoded_img = self.decoder(encoded_img)
        return decoded_img