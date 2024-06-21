import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
import torch.nn as nn
import pandas as pd


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model,train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，常用于分类
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型的参数
    # 初始化
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()
    # 训练
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)
        # 初始化参数
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        train_num = 0.0  # 训练集样本数量
        val_num = 0.0   # 验证集样本数量
        # 分批次训练，每批次数据量batch_size=128
        for step, (b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()  # 模型开启训练模式
            # 前向传播过程，输入为一个batch，输出为一个batch，值为概率
            output = model(b_x)
            per_lab = torch.argmax(output, dim=1)  #查找每行中最大的值对应的标签
            loss = criterion(output, b_y)  # 该批次每个样本的平均loss值
            # 将梯度初始化为0,防止梯度累加
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息更新网络参数
            optimizer.step()
            # 对该轮次损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            train_corrects += torch.sum(per_lab == b_y.data)
            # 该轮次所有训练的样本数
            train_num += b_x.size(0)
        # 验证
        for step, (b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()  # 模型开启验证模式
            # 前向传播过程，输入为一个batch，输出为一个batch，值为概率
            output = model(b_x)
            per_lab = torch.argmax(output, dim=1)  # 查找每行中最大的值对应的标签
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            val_corrects += torch.sum(per_lab == b_y.data)
            # 所有训练的样本数
            val_num += b_x.size(0)

        # 计算并保存每次迭代的loss和准确率
        train_loss_all.append(train_loss / train_num)   # 该轮次的平均loss
        val_loss_all.append(val_loss / val_num)   # 该轮次的平均loss

        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} Train Loss: {:.4f} Train Acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} Val Loss: {:.4f} Val Acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度,并保存最高准确度对应的模型参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        # 计算耗时
        time_use = time.time()-since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 保存最优模型的参数
    torch.save(best_model_wts, "E:/code/AlexNet/best_model.path")
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #  将模型实例化
    print("-" * 10)
    LeNet = AlexNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
