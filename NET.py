import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features1 = nn.Sequential(
            nn.Linear(272, 1089)
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, 11, padding=5, stride=1),  # 33 33
            nn.ReLU(inplace=True),  # 参数inplace的意思是是否直接覆盖，在这里就是对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
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
        x_1 = x_1.view([-1, 1, 33, 33])  # 类似于 np.reshape，但是操作前后的tensor张量共享存储空间
        x_2 = self.features2(x_1)
        return x_2


class ResNet(nn.Module):


    def __init__(self):
        super(ResNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Linear(272, 1089)
        )
        """
            stride适当增大可以增大感受野
            padding设置为1可以填充像素，保证图像角落信息也被获取到
            kernel_size也关系到感受野，最常见的是3×3
        """
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=5),  # 该卷积层的参数为3*64*7(通道数)，步长为2，填充为3，偏移为0
            nn.BatchNorm2d(64),  # 数据规范化
            nn.ReLU(inplace=True),  # 参数inplace的意思是是否直接覆盖，在这里就是对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层，筛选出重要的参数并保留
        )
        # 添加5层残差层
        self.features2.add_module('layer1', Residual_block(64, 64, 2, first_block=True))
        self.features2.add_module('layer2', Residual_block(64, 128, 2))
        self.features2.add_module('layer3', Residual_block(128, 64, 2))
        self.features2.add_module('layer4', Residual_block(64, 32, 2))
        self.features2.add_module('layer5', Residual_block(32, 1, 2))
        # # 加入平均池化层
        # self.features2.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, input_img):
        x_1 = self.features1(input_img)
        x_1 = x_1.view([-1, 1, 33, 33])  # 类似于 np.reshape，但是操作前后的tensor张量   共享存储空间
        x_2 = self.features2(x_1)
        return x_2


# 残差单元
class Residual(nn.Module):
    """
    channnels的含义是每个卷积层中卷积核（过滤器）的数量，比如一般的RGB图片channels为3（三个通道）
    最初输入的图片样本的channels ，取决于图片类型，比如RGB；
    卷积操作完成后输出的out_channels，取决于卷积核的数量；不同的卷积核（过滤器）能检测不同特征，并决定了输出的数据有多少个通道，即决定了out_channels
    此时的out_channels也会作为下一次卷积时的卷积核的in_channels；
    卷积核中的in_channels ，刚刚2中已经说了，就是上一次卷积的out_channels ，如果是第一次做卷积，就是1中样本图片的channels
    ？？？downsample可能是stride非1的卷积或是池化。鉴于这是残差块，要保留梯度，所以应该是池化层？？？
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Residual, self).__init__()  # 继承Module类自己微调模型
        """
        in_channels：在Conv1d（文本应用）中，即为词向量的维度；在Conv2d中，即为图像的通道数
        out_channels：卷积产生的通道数，有多少个out_channels，就需要多少个一维卷积（也就是卷积核的数量）
        kernel_size：卷积核的尺寸；卷积核的第二个维度由in_channels决定，所以实际上卷积核的大小为kernel_size * in_channels
        stride：步长
        padding：填充。对输入的每一条边，补充0的层数，能够使卷积过的图像不会过于缩小，同时放大了角落或图像边缘的信息发挥的作用
        如果padding=(kernel_size-1)/2 那么经过卷积后，输入和输出的图像大小依旧相等，如果padding=0，那么必然会缩小（当然这里默认stride——步长为1）
        """
        # 第一层卷积神经网络
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 指定stride
        # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        # nn.BatchNorm2d()的作用是根据统计的mean和var来对数据进行标准化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 第二层卷积神经网络
        self.bn2 = nn.BatchNorm2d(out_channels)
        # ？？？下采样    1.使得图像符合显示区域的大小     2.生成对应图像的缩略图    ？？？
        self.downsamples = downsample
        if self.downsamples:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    """
    forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心。
    当执行model(x)的时候，底层自动调用forward方法计算结果。
    """

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))  # 调用激活函数ReLu，来激活第一层卷积运算得出的值
        out = self.bn2(self.conv2(out))  # 将激活后的值放入第二层卷积层中运算
        if self.downsamples:  # 如果可以下采样，就将x下采样，并在最后返回时采用下采样的值
            identity = self.downsample(x)

        """残差网络的计算方法就是对（初始输入+经过第一层卷积->激活->第二层卷积）进行激活，以达到跳远链接的目的"""
        return F.relu(identity + out)  # 返回初始输入和第一层运算、激活、第二层运算的总体激活值


# 残差块
def Residual_block(in_channels, out_channels, num_Residual, first_block=False):
    if first_block:  # 如果first_block为True，且输入的通道数和输出的通道数不一致，则抛出异常
        assert in_channels == out_channels
    BasicBlock = []  # 定义一个空的列表
    for i in range(num_Residual):
        if i == 0 and not first_block:  # 当进行第一次运算(i=0)且first_block=False时，添加第一个残差块，步长为2，下采样为True
            BasicBlock.append(
                Residual(in_channels, out_channels, downsample=True, stride=1))  # 执行Residual()会自动调用forward方法运算
        else:  # 不是第一次运算时，添加步长为1，下采样为False的残差块（具体见Residual的构造函数）
            BasicBlock.append(Residual(out_channels, out_channels))
    """返回一个由nn.Sequential搭建的模型，模型主要由残差单元构成"""
    return nn.Sequential(*BasicBlock)  # *BasicBlocK的’*‘代表把BasicBlock里的所有参数传进去给nn.Sequential去搭建模型