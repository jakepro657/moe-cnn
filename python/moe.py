import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm  # tqdm 라이브러리 임포트

# 폴더 구조 생성
base_dir = 'MoE_CNN_Training'
directories = [
    'data',
    'data/CIFAR-10',
    'code',
    'models',
    'results'
]

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for directory in directories:
    os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

print(f"폴더 구조가 {base_dir}에 생성되었습니다.")

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 데이터셋 다운로드 및 로드
trainset = torchvision.datasets.CIFAR10(root='./MoE_CNN_Training/data/CIFAR-10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./MoE_CNN_Training/data/CIFAR-10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("CIFAR-10 데이터셋이 준비되었습니다.")

# Attention 모듈 정의
class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return out

# Expert 모듈 정의
class Expert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = Attention(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attention(x)
        x = self.pool(x)
        return x

# MoE 모듈 정의
class MoE(nn.Module):
    def __init__(self, num_experts, in_channels, out_channels, num_classes):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(in_channels, out_channels) for _ in range(num_experts)])
        self.gating_network = nn.Linear(in_channels * 32 * 32, num_experts)  # CIFAR-10 이미지 크기 (32x32)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        gate_inputs = x.view(batch_size, -1)
        gate_outputs = self.gating_network(gate_inputs)
        gate_outputs = F.softmax(gate_outputs, dim=1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 각 배치에 대해 가중합 계산
        gate_outputs = gate_outputs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weighted_sum = torch.sum(gate_outputs * expert_outputs, dim=1)
        out = F.adaptive_avg_pool2d(weighted_sum, (1, 1)).view(batch_size, -1)
        out = self.fc(out)
        return out

# 모델 초기화
num_experts = 4
in_channels = 3
out_channels = 64
num_classes = 10

model = MoE(num_experts, in_channels, out_channels, num_classes)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)

# 정확도 계산 함수
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total

# 학습 함수 정의
def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch + 1}")
        for i, data in progress_bar:
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / (i + 1)))

        # 에포크가 끝날 때마다 학습 및 테스트 정확도 출력
        train_accuracy = calculate_accuracy(trainloader, model)
        test_accuracy = calculate_accuracy(testloader, model)
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    print("학습이 완료되었습니다.")

# 모델 학습
train_model(model, trainloader, testloader, criterion, optimizer)

# 모델 저장
torch.save(model.state_dict(), './moe_cnn.pth')