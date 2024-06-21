import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet

###############
def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 初始化
    test_corrects = 0.0
    test_num = 0.0  # 测试集样本数量
    # 只进行前向传播，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:  # 每一个批次只有一个数据
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 模型开启测试模式
            # 前向传播过程，输入为一个数据，输出为多个概率值
            output = model(test_data_x)
            per_lab = torch.argmax(output, dim=1)  # 查找每行中最大概率值对应的行标
            # 如果预测正确，则准确度加1
            test_corrects += torch.sum(per_lab == test_data_y.data)
            # 该轮次所有测试的样本数
            test_num += test_data_x.size(0)
        # 计算并保存测试准确率
        test_acc = test_corrects.double().item() / test_num
        print(" Test Acc:{:.4f}".format(test_acc))


if __name__ == "__main__":
    model = AlexNet()
    model.load_state_dict(torch.load('best_model.path'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()  # 模型开启验证模式
            # 前向传播过程，输入为一个数据，输出为多个概率值
            output = model(b_x)
            per_lab = torch.argmax(output, dim=1)  # 查找每行中最大概率值对应的行标
            result = per_lab.item()
            label = b_y.item()
            print(" 预测值:", classes[result],"---------", "真实值:", classes[label])
