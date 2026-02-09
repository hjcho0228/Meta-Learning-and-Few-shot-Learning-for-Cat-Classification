import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
import random
from PIL import Image
from tqdm.auto import trange, tqdm
import glob

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 현재 사용 중인 디바이스 출력
print(f"현재 사용 중인 디바이스: {'GPU' if use_cuda else 'CPU'}")

# ResNet-50 Feature Extractor 설정
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()  # 마지막 Fully Connected Layer 제거
resnet = resnet.to(device)

# ----------------------------
# 모델 저장 함수
# ----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# ----------------------------
# 모델 로드 함수
# ----------------------------
# ----------------------------
# 모델 로드 함수
# ----------------------------
def load_model(model, path):

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # 모델을 평가 모드로 전환
    print(f"Model loaded from {path}")
    return model

# ----------------------------
# 데이터셋 정의
# ----------------------------
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 각 클래스 폴더가 포함된 루트 디렉토리
        transform: 이미지 전처리 (예: ToTensor, Normalize 등)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_indices = {}
        self.data = []
        self.labels = []

        # 각 클래스 폴더와 이미지 파일 읽기
        for label, class_folder in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):
                image_files = sorted(os.listdir(class_path))
                indices = list(range(len(self.data), len(self.data) + len(image_files)))
                self.class_indices[label] = indices                                             # 레이블(클래스)별 인덱스 저장
                for image_file in image_files:
                    self.data.append(os.path.join(class_path, image_file))                      # 이미지 경로를 저장
                    self.labels.append(label)                                                   # 이미지에 해당하는 레이블을 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")                       # RGB로 변환
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# 에피소드 생성 함수
# ----------------------------
def generate_episodes(class_indices, episodes, ways, support_size, query_size):
    
    all_episodes = []

    for _ in range(episodes):
        # 1) 에피소드에 포함될 클래스 선택
        selected_classes = random.sample(list(class_indices.keys()), ways) # 예시 : [1,2]

        support_set = []
        query_set = []
        support_labels = []
        query_labels = []

        # 2) 각 클래스에 대해 Support와 Query 나누기
        for label in selected_classes:
            # 현재 클래스의 모든 이미지 인덱스
            indices = class_indices[label]
            # 셔플 후 Support와 Query로 분리
            random.shuffle(indices)
            support = indices[:support_size]
            query = indices[support_size:support_size + query_size]

            # Support/Query와 라벨 저장
            support_set.extend(support)
            query_set.extend(query)
            support_labels.extend([label] * support_size)
            query_labels.extend([label] * query_size)

        # 3) 에피소드 저장
        all_episodes.append({
            'support_set': support_set,             # support set에 해당하는 인덱스
            'query_set': query_set,                 # query set에 해당하는 인덱스
            'support_labels': support_labels,       # support set의 라벨
            'query_labels': query_labels,           # query set의 라벨
        })

    return all_episodes 

# ----------------------------
# 에피소드 기반 DataLoader 정의
# ----------------------------
class EpisodeDataset(Dataset):
    def __init__(self, episodes_data, dataset, transform):
        self.episodes_data = episodes_data
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.episodes_data)

    def __getitem__(self, idx):
        episode = self.episodes_data[idx]
        support_set = torch.stack([self.transform(Image.open(self.dataset.data[i]).convert("RGB")) for i in episode['support_set']])
        query_set = torch.stack([self.transform(Image.open(self.dataset.data[i]).convert("RGB")) for i in episode['query_set']])
        support_labels = torch.tensor(episode['support_labels'])
        query_labels = torch.tensor(episode['query_labels'])
        return support_set, query_set, support_labels, query_labels
    
# ----------------------------
# Prototypical Loss 정의
# ----------------------------
def prototypical_loss(support_embeddings, query_embeddings, support_labels, query_labels, distance_fn='euclidean'):

    unique_labels = torch.unique(support_labels)
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}  # 예시 : {20: 0, 585: 1}
    query_labels = torch.tensor([label_map[label.item()] for label in query_labels])
    prototypes = []
    
    # 각 클래스에 대해 프로토타입 계산
    for label in unique_labels:
        class_embeddings = support_embeddings[support_labels == label]
        prototypes.append(class_embeddings.mean(dim=0))
    prototypes = torch.stack(prototypes)  # torch.Size([2,2048])
    # Query와 Prototypes 간의 거리 계산
    if distance_fn == 'euclidean':
        distances = torch.cdist(query_embeddings, prototypes)  # torch.Size([14,2]) : 각 샘플과의 거리, 총 14개 샘플이고 각 샘플마다 거리가 두 개씩
    elif distance_fn == 'cosine':
        distances = 1 - torch.nn.functional.cosine_similarity(
            query_embeddings.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
        )
    else:
        raise ValueError(f"Unsupported distance function: {distance_fn}")

    scores = 1 / (1 + distances)  # 역수로 변환하여 점수 계산 (작은 거리가 큰 점수가 되도록)
    scores = scores / scores.sum(dim=1, keepdim=True)  # 각 행을 합하여 1로 정규화

    # Query의 예측 라벨
    predictions = torch.argmax(scores, dim=1)  # 점수가 가장 큰 클래스 인덱스를 반환
    loss = nn.CrossEntropyLoss()(scores, query_labels.to(scores.device)) # distance는 (14,2), query_labels는 (14)의 사이즈를 가짐
    accuracy = (predictions == query_labels.to(predictions.device)).float().mean().item()


    return loss, accuracy

# ----------------------------
# 학습 루프 정의
# ----------------------------
def train_prototypical_networks(resnet, episode_loader, epochs, distance_fn='euclidean', save_path=None):
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)  # ResNet 학습용 Adam 옵티마이저
    epoch_losses = []  # 에폭별 손실 기록
    epoch_accuracies = []  # 에폭별 정확도 기록

    # tqdm으로 에폭 진행 표시
    for epoch in trange(epochs, desc="Epochs"):
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        resnet.train()

        # tqdm으로 에피소드 진행 표시
        for support_set, query_set, support_labels, query_labels in tqdm(
            episode_loader, desc=f"Training Epoch {epoch + 1}", leave=False
        ):
            # DataLoader에서 반환된 텐서의 차원 조정 및 GPU로 이동
            support_set, query_set = support_set.squeeze(0).to(device), query_set.squeeze(0).to(device)
            support_labels, query_labels = support_labels.squeeze(0).to(device), query_labels.squeeze(0).to(device)

            # Feature 추출
            support_embeddings = resnet(support_set)  # 크기: (n_support, feature_dim)
            query_embeddings = resnet(query_set)      # 크기: (n_query, feature_dim)
            
            # Prototypical Loss 계산
            loss, acc = prototypical_loss(
                support_embeddings=support_embeddings,
                query_embeddings=query_embeddings,
                support_labels=support_labels,
                query_labels=query_labels,
                distance_fn=distance_fn
            )

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()  # 기울기 초기화
            loss.backward()        # 역전파로 기울기 계산
            optimizer.step()       # 파라미터 업데이트

            # 현재 에폭의 손실 및 정확도 누적
            epoch_loss += loss.item()
            epoch_acc += acc

        # 에폭별 평균 손실 및 정확도 계산
        avg_loss = epoch_loss / len(episode_loader)
        avg_acc = epoch_acc / len(episode_loader)

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_acc)

        # 에폭 결과 출력
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

        # 매 5번째 에폭마다 모델 저장
        if save_path and (epoch + 1) % 10 == 0:
            formatted_path = save_path.format(epoch=epoch + 1)
            torch.save(resnet.state_dict(), formatted_path)
            tqdm.write(f"Model saved at {formatted_path}")

    print("Training complete!")
    return epoch_losses, epoch_accuracies

# ----------------------------
# 메타 테스트 함수
# ----------------------------
'''
def meta_test(resnet, cat1_dir, cat2_dir, transform):
    def compute_prototypes(support_set_dir):
        image_paths = glob.glob(os.path.join(support_set_dir, "*.jpg"))
        embeddings = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(image)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        prototype = embeddings.mean(dim=0)
        return prototype

    # Prototypes 계산
    cat1_prototype = compute_prototypes(os.path.join(cat1_dir, "support_set"))
    cat2_prototype = compute_prototypes(os.path.join(cat2_dir, "support_set"))
    prototypes = torch.stack([cat1_prototype, cat2_prototype])

    # Query 이미지와 라벨 가져오기
    query_images = glob.glob(os.path.join(cat1_dir, "query_set", "*.jpg")) + \
                   glob.glob(os.path.join(cat2_dir, "query_set", "*.jpg"))
    query_labels = [0] * len(glob.glob(os.path.join(cat1_dir, "query_set", "*.jpg"))) + \
                   [1] * len(glob.glob(os.path.join(cat2_dir, "query_set", "*.jpg")))

    # Query 평가
    correct, total = 0, 0
    for img_path, true_label in zip(query_images, query_labels):
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(image)
            distances = torch.cdist(embedding, prototypes.unsqueeze(0)).squeeze(0)  # (n_classes,)
            predicted_class = torch.argmin(distances).item()  # 최소값의 인덱스(예측 클래스)

            # 성능 기록
            correct += int(predicted_class == true_label)
            total += 1
        print("Distances:", distances)
        print("Predicted Class:", predicted_class, "True Label:", true_label)
        print("\n")

    print(f"Overall Accuracy: {correct / total:.4f}")
'''

def meta_test(resnet, cat1_dir, cat2_dir, query_dir, transform, device):

    def compute_prototypes(support_set_dir):
        import glob
        image_paths = glob.glob(os.path.join(support_set_dir, "*.jpg"))
        embeddings = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(image)  # (1, embed_dim)
            embeddings.append(embedding)
        if len(embeddings) == 0:
            # 만약 support_set 폴더가 비어있다면, 임의로 0벡터 반환(실전에서는 에러 처리)
            return torch.zeros(1, device=device)

        embeddings = torch.cat(embeddings, dim=0)  # (N, embed_dim)
        prototype = embeddings.mean(dim=0)         # (embed_dim,)
        return prototype

    # 1) 카테고리별 프로토타입 계산
    cat1_prototype = compute_prototypes(os.path.join(cat1_dir, "support_set"))
    cat2_prototype = compute_prototypes(os.path.join(cat2_dir, "support_set"))
    prototypes = torch.stack([cat1_prototype, cat2_prototype], dim=0)  # (2, embed_dim)
    print("Prototypes:", prototypes.sum(dim=1))

    # 2) Query 이미지 수집 (라벨은 사용 안 함)
    query_images = glob.glob(os.path.join(query_dir, "*.jpg"))

    # (선택) 예측 클래스 이름을 보기 좋게 표시하기 위한 매핑 (인덱스→이름)
    class_names = ["Cat1", "Cat2"]

    # 성능 기록용 변수
    correct = 0
    total = 0

    # 3) Query마다 예측
    for img_path in query_images:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(image)  # (1, embed_dim)
            # prototypes.shape = (2, embed_dim) → cdist 결과 (1,2)
            distances = torch.cdist(embedding, prototypes.unsqueeze(0)).squeeze(0)
            # distances: (1, 2) → squeeze(0) → (2,)

            predicted_class_idx = torch.argmin(distances).item()
            predicted_class_name = class_names[predicted_class_idx]

        print(f"[{img_path}]")
        print("Embedding:", embedding.sum)
        print("Distances:", distances.cpu().numpy())
        print(f"Predicted Class: {predicted_class_idx} ({predicted_class_name})\n")


# ----------------------------
# 메타 테스트 함수 (Threshold 추가)
# ----------------------------
def meta_test_with_threshold(resnet, cat1_dir, cat2_dir, cat3_dir, threshold, transform):
    def compute_prototypes(support_set_dir):
        image_paths = glob.glob(os.path.join(support_set_dir, "*.jpg"))
        embeddings = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(image)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        prototype = embeddings.mean(dim=0)
        return prototype

    # Prototypes 계산
    cat1_prototype = compute_prototypes(os.path.join(cat1_dir, "support_set"))
    cat2_prototype = compute_prototypes(os.path.join(cat2_dir, "support_set"))
    prototypes = torch.stack([cat1_prototype, cat2_prototype])  # (2, feature_dim)

    # Query 이미지와 라벨 가져오기 (cat1, cat2)
    query_images = glob.glob(os.path.join(cat1_dir, "query_set", "*.jpg")) + \
                   glob.glob(os.path.join(cat2_dir, "query_set", "*.jpg"))
    query_labels = [0] * len(glob.glob(os.path.join(cat1_dir, "query_set", "*.jpg"))) + \
                   [1] * len(glob.glob(os.path.join(cat2_dir, "query_set", "*.jpg")))

    # Query 이미지 추가 (cat3)
    cat3_images = glob.glob(os.path.join(cat3_dir, "*.jpg"))
    query_images += cat3_images
    query_labels += [-1] * len(cat3_images)  # cat3은 unknown class로 라벨 -1

    # Query 평가
    correct, total, unknown_count = 0, 0, 0
    cat1_unknown, cat2_unknown, cat3_correct = 0, 0, 0  # Unknown 카운트 및 cat3 정확도
    for img_path, true_label in zip(query_images, query_labels):
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(image)
            distances = torch.cdist(embedding, prototypes.unsqueeze(0)).squeeze(0)  # (n_classes,)
            min_distance, predicted_class = distances.min(dim=1)  # 최소 거리 및 클래스 인덱스
            min_distance = min_distance.item()

            # Threshold를 이용한 unknown class 판별
            if min_distance > threshold:
                predicted_class = -1  # Unknown class
                unknown_count += 1
                if true_label == 0:
                    cat1_unknown += 1
                elif true_label == 1:
                    cat2_unknown += 1
                elif true_label == -1:
                    cat3_correct += 1
            else:
                predicted_class = predicted_class.item()

            # 성능 기록
            if predicted_class == true_label:
                correct += 1
            total += 1
            
            print("\nDistances:", distances, "Min Distance:", min_distance)
            print("Predicted Class:", predicted_class, "True Lable:", true_label)
            

    # 결과 출력
    print(f"Overall Accuracy: {correct / total:.4f}")
    print(f"Total Unknown: {unknown_count}/{total} ({unknown_count / total:.4f})")
    print(f"cat1 Unknown: {cat1_unknown}")
    print(f"cat2 Unknown: {cat2_unknown}")
    print(f"cat3 Correctly Classified as Unknown: {cat3_correct}/{len(cat3_images)}")




# ----------------------------
# 메인 실행
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 1. 데이터셋 로드
    # ----------------------------
    root_dir = "./imagenet10k_by_class"         # 데이터셋 폴더 경로
    # ResNet-50에 맞는 transform 적용
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomImageDataset(root_dir, transform=transform)

    # 클래스별 인덱스 가져오기
    datas = dataset.data                        # [..., './imagenet10k_by_class/n13054560/n13054560_1744.JPEG' ,...]
    labels = dataset.labels                     # [1 10개, 2 10개, 3 10개, ..., 1000 10개]
    class_indices = dataset.class_indices       # {1: [...], 2: [...], 3: [...]}
    
    # ----------------------------
    # 2. 에피소드 생성
    # ----------------------------
    total_classes = len(class_indices)          # 총 클래스 수
    images_per_class = len(class_indices[1])    # 클래스당 이미지 수 (10개)
    episodes = 500                              # 총 에피소드 개수
    support_size = 3                            # Support set에서 클래스당 샘플 수
    query_size = 7                              # Query set에서 클래스당 샘플 수
    ways = 2                                    # 한 에피소드에서 선택할 클래스 수

    # 에피소드 생성
    episodes_data = generate_episodes(class_indices, episodes, ways, support_size, query_size)

    # ----------------------------
    # 3. DataLoader 구성 (옵션)
    # ----------------------------
    # 에피소드 기반의 DataLoader를 정의할 수도 있습니다.
    # 예: 각 에피소드 데이터를 DataLoader로 래핑하여 학습에 활용
    # 에피소드 데이터셋 및 DataLoader 생성
    episode_dataset = EpisodeDataset(episodes_data, dataset, transform)
    episode_loader = DataLoader(episode_dataset, batch_size=1, shuffle=True) # 한 번에 하나의 에피소드 단위로 학습함
    
    # ----------------------------
    # 4. 학습 시작
    # ----------------------------
    '''
    # 학습 설정
    epochs = 200  # 총 학습 에폭 수
    distance_fn = 'euclidean'  # 거리 함수 ('euclidean' 또는 'cosine')
    save_path = "./prototypical_network_resnet50_epoch_{epoch}.pth"
    '''
    '''
    # Prototypical Networks 학습
    train_prototypical_networks(
        resnet=resnet, 
        episode_loader=episode_loader, 
        epochs=epochs, 
        distance_fn=distance_fn, 
        save_path=save_path
    )
    print("Training complete!")
    '''

    # ----------------------------
    # 5. Meta-Test 설정
    # ----------------------------
    
   # ResNet-50 모델 초기화 (저장된 가중치를 로드하기 위해)
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()  # 마지막 Fully Connected Layer 제거
    resnet = resnet.to(device)
    
    save_path = "./prototypical_network_resnet50_epoch_10.pth"
    resnet = load_model(resnet, save_path)
    
    # cat1, cat2 디렉토리 경로
    cat1_dir = "./cat1_images"
    cat2_dir = "./cat2_images"
    cat3_dir = "./cat3_images"
    query_dir = "./query_images"

    # Meta-Test 수행

    print("Starting Meta-Test...")
    meta_test(resnet, cat1_dir, cat2_dir, query_dir, transform, device)
    '''
    threshold = 5.5
    print("\n Starting Meta-Test-With-Threshold")
    meta_test_with_threshold(resnet,cat1_dir,cat2_dir,cat3_dir,threshold,transform)
    
    '''