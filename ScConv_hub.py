import torch.nn.functional as F
from torch import relu
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transform
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# from thop import profile
import time
from PIL import Image


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 in_channels: int,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()
        self.group_num = group_num
        self.gn = nn.GroupNorm(num_channels=in_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # N, C, H, W = x.size()
        # gn_new = nn.GroupNorm(num_channels=C, num_groups=self.group_num)
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 in_channel: int,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * in_channel)
        self.low_channel = low_channel = in_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        # print(x.size())
        # print(self.up_channel)
        # print(self.low_channel)
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 in_channel: int,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(in_channel,
                       op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(in_channel,
                       op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.stride = stride
        self.planes = planes
        self.in_planes = in_planes
        #         self.conv1 = ScConv(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = ScConv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return relu(self.conv3(x) + self.shortcut(x))
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, num_blocks: list[int],
                 num_classes: int, nf: int) -> None:
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:

        out = relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        # out = nn.AvgPool2d(out, out.shape[2])  # -> 512, 1, 1
        out = self.avgpool(out)  # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)

        raise NotImplementedError("Unknown return type")


def resnet18(dataset: str, nf: int = 64) -> ResNet:
    if dataset == 'seq-cifar10' or dataset == 'rot-mnist':
        nclasses = 10
    elif dataset == 'seq-cifar100':
        nclasses = 100
    elif dataset == 'seq-tinyimg':
        nclasses = 200
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)


# 定义相关参数
batch_size = 64  # 加载数据时一次处理数量
total_train_step = 0  # 总训练步数
epoch = 30  # 计划训练轮数
lr = 0.009  # 学习率
weight_decay = 8e-5  # 权重
accuracy = []  # 准确率列表
loss_all = []  # 损失率列表


train_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 对测试集的处理
test_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


class PreDataset(CIFAR10):
    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]
        img = Image.fromarray(img)
        img_1 = self.transform(img)

        return img_1, target


# 导入训练集和测试集
train_dataset = PreDataset(root='./train_dataset', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./test_dataset', train=False, download=True, transform=test_transform)
train_data = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
test_data = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)


# 选择训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建神经网络
model = resnet18('seq-cifar10')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# #测试flops和参数量
# dummy_input = torch.randn(1, 3, 32, 32)
# dummy_input = dummy_input.to(device)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


# 记录训练时间
start_time = time.time()

# 正式训练
for i in range(epoch):
    print("*************第{}轮训练开始*************".format(i + 1))
    loss_total = 0
    model.train()
    start_time = time.time()
    for idx, data in enumerate(train_data):
        # 获取数据
        img1, target1 = data  # img1.shape == [4, 3, 224, 224]
        img1, target1 = img1.to(device), target1.to(device)

        # 训练
        output1 = model(img1)
        optimizer.zero_grad()
        loss = criterion(output1, target1)
        loss_total += loss
        loss.requires_grad_()
        loss.backward()
        optimizer.step()

        # total_train_step += 1

    end_time = time.time()
    print("训练花费时间为{}".format(end_time - start_time))
    loss_all.append(float(loss_total))
    print('第{}轮训练:'.format(i + 1), 'loss_total:', loss_total)

    #  输出单轮训练结果
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            total_accuracy += (output.argmax(1) == targets).sum()

    accuracy.append(total_accuracy)

    print("整体Accuracy:{}".format(total_accuracy / 10000))
#     # 模型的中间的结果的保存
#     torch.save(model, "alter_{}.pth".format(i + 1))  # 保存模型，第一个参数为模型名，第二个为保存的文件名称
