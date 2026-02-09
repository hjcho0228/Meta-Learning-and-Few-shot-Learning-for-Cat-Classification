import numpy as np
import os
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# CIFAR-100 train 파일 로드
cifar100_path = './cifar-100-python/train'
data = unpickle(cifar100_path)
# meta 파일 로드
meta_path = './cifar-100-python/meta'
meta_data = unpickle(meta_path)

# 데이터 디코딩
images = data[b'data']  # 이미지 데이터: (50000, 3072) 형태
fine_labels = data[b'fine_labels']  # 세부 클래스 라벨: 0~99 범위
coarse_labels = data[b'coarse_labels']  # 상위 클래스 라벨: 0~19 범위
filenames = data.get(b'filenames', None)  # 파일 이름 정보 (필요한 경우)

# 클래스 이름 로드 및 디코딩
fine_label_names = [label.decode('utf-8') for label in meta_data[b'fine_label_names']]
coarse_label_names = [label.decode('utf-8') for label in meta_data[b'coarse_label_names']]

# 이미지 데이터 형상 변환 (32x32x3 형태로 재구성)
images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # [N, H, W, C]

# 저장 경로 설정
output_dir = './cifar100_by_class'
os.makedirs(output_dir, exist_ok=True)

# 클래스별로 디렉토리 생성 및 이미지 저장
for i, img in enumerate(images):
    fine_label = fine_labels[i]
    fine_label_name = fine_label_names[fine_label]
    
    # 클래스 디렉토리 생성
    class_dir = os.path.join(output_dir, fine_label_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 이미지 저장 (PNG 형식)
    img = Image.fromarray(img)
    img.save(os.path.join(class_dir, f"img_{i}.png"))

print(f"이미지들이 클래스별로 {output_dir}에 저장되었습니다!")




