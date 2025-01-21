# 패키지 임포트
from PIL import Image
import torch 
import torch.nn as nn
from torchvision.trasforms import Compose, ToTensor, Normalize
from torch.nn.functional import softmax
device='cuda' if torch.cuda.is_available() else 'cpu' 

# 추론에 사용할 데이터 준비(전처리)
example_image_path = 'test_image.jpeg'
image = Image.open(example_image_path)
image = image.resize((32, 32))

# cifar10 평균과 표준편차
CIFAR_MEAN = [0.491, 0.482, 0.447]
CIFAR_STD = [0.247, 0.244, 0.262]

transform = Compose([
    ToTensor(),
    Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
])

image = transform(image)
image = image.unsqueeze(0).to(device)

# 훈련 완료한 모델 준비
# 모델 껍데기 만들기: 설계도 + 하이퍼파라미터 
class myVGG(nn.Module):
    # 초기화
    def __init__(self, num_classes=10):
        super().__init__()
        # cifar10 활용하므로 크기 변형
        # 레이어 5개, 출력
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.out = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
    # 피드포워드: 레이어 5개, 출력
    def forward(self, x): 
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

num_classes = 10
model = myVGG(num_classes=num_classes)

# 빈 모델에 학습된 웨이트를 덮어씌우기
weight = torch.load('best_model.ckpt', map_location=device)
model.load_state_dict(weight)
model = model.to(device)

# 준비한 데이터를 모델에 입력
output = model(image)

# 추론 결과 해석 
# 결과를 사람이 이해할 수 있는 형태로 변환
probability = softmax(output, dim=1)
values, indices = torch.max(probability, dim=1)
prob = values.item()*100
predict = indices.item()

print(f"모델은 해당 사진에 {prob:.2f}%의 확률로 {predict}이라고 추론했습니다.")
# 모델의 추론 결과를 객관적으로 평가