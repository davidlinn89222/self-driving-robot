import torch.nn as nn

class CNN_Model1(nn.Module):
    def __init__(self, img_size, name):
        super(CNN_Model1, self).__init__()
        self.name = name
        self.max_pooling_num = 3
        self.img_size = img_size
        self.output_size = [img_size[0]/(2**self.max_pooling_num), img_size[1]/(2**self.max_pooling_num)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(64 * int(self.output_size[0]) * int(self.output_size[1]), 18)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # flatten the output of conv2 to (batch_size, ?)
        out = self.fc1(out)
        return out

class CNN_Model2(nn.Module):
    def __init__(self, img_size, name):
        super(CNN_Model2, self).__init__()
        self.name = name
        self.max_pooling_num = 1
        self.img_size = img_size
        self.output_size = [img_size[0]/(2**self.max_pooling_num), img_size[1]/(2**self.max_pooling_num)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(16 * int(self.output_size[0]) * int(self.output_size[1]), 18)

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)  # flatten the output of conv2 to (batch_size, ?)
        out = self.fc1(out)
        return out

class CNN_Model3(nn.Module):
    def __init__(self, img_size, name):
        super(CNN_Model3, self).__init__()
        self.name = name
        self.max_pooling_num = 5
        self.img_size = img_size
        self.output_size = [img_size[0]/(2**self.max_pooling_num), img_size[1]/(2**self.max_pooling_num)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(512 * int(self.output_size[0]) * int(self.output_size[1]), 4096),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU())

        self.fc3 = nn.Linear(2048, 18)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)  # flatten the output of conv2 to (batch_size, ?)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
