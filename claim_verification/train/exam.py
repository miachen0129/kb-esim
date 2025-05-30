import torch
import torch.nn as nn
import torch.optim as optim


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2*input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # 拼接两个输入张量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 模型参数
input_size = 3  # 输入特征维度
hidden_size = 64  # 隐藏层大小
num_classes = 3  # 输出类别数量

# 创建模型
model = MLPClassifier(input_size, hidden_size, num_classes)

# 随机生成训练数据
n = 1000  # 样本数量
x1_train = torch.randn(n, input_size)  # 第一个输入张量，大小为 (n, input_size)
x2_train = torch.randn(n, input_size)  # 第二个输入张量，大小为 (n, input_size)
labels_train = torch.randint(0, num_classes, (n,))  # 随机生成标签，范围为 [0, num_classes)，大小为 (n,)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()

    # 前向传播
    outputs = model(x1_train, x2_train)

    # 计算损失
    loss = criterion(outputs, labels_train)

    # 清零梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')
