import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features1 = nn.Sequential(
            nn.Linear(272, 1089)
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, 11, padding=5, stride=1),  # 33 33
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=7, padding=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(1, 64, kernel_size=11, padding=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=7, padding=3, stride=1),
        )


    def forward(self, input_img):
        x_1 = self.features1(input_img)
        x_1 = x_1.view([-1, 1, 33, 33])
        x_2 = self.features2(x_1)
        return x_2
