import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256,"M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGGnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGnet(in_channels=3, num_classes=1000).to(device)
    print(model)
    ## N = 3 (Mini batch size)
    # x = torch.randn(3, 3, 224, 224).to(device)
    # print(model(x).shape)

# ref https://github.com/aladdinpersson/Machine-Learning-Collection

    

train = ImageDataGenerator(rescale= 1/224)
validation = ImageDataGenerator(rescale= 1/224)

train_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir
val_dataset = train.flow_from_directory('googlenet/train/', 
                target_size=("224","224"), batch_size=3,class_mode= 'binary') #batch size'ı değiştir

model = VGGnet()

checkpoint = ModelCheckpoint('vggnet.h5', monitor='val_loss', mode='min',save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0,patience=3,verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)
callbacks = [earlystop,checkpoint,reduce_lr]


model.compile(loss="binary_crossentropy", optimizer = RMSprop(lr=0.001),metrics = ['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs = 10, batch_size=32,callbacks=callbacks,validation_data = val_dataset)
