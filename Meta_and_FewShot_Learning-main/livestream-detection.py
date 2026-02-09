import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import os
from datetime import datetime

# 디렉토리 생성 (query_images 폴더 생성)import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import os
import glob
from PIL import Image

##############################
# 1) 메타 러닝 ResNet 로드
##############################
def load_meta_resnet(model_path=None, device='cpu'):
    """
    - 메타 러닝(프로토타입 네트워크)으로 학습한 ResNet50을 로드
    - 마지막 fc를 Identity로 교체했으며, 저장된 state_dict를 불러옴
    """
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()  # 최종 분류 레이어 제거 (임베딩 추출용)
    resnet = resnet.to(device)

    if model_path is not None and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        resnet.load_state_dict(state)
        print(f"[INFO] Meta-ResNet loaded from {model_path}")
    else:
        print(f"[WARNING] No valid model path. Using default ResNet50 as feature extractor.")
    resnet.eval()
    return resnet

##############################
# 2) cat1, cat2 프로토타입 계산
##############################
def compute_prototypes(resnet, cat_dir, device='cpu', transform=None):
    """
    cat_dir/support_set/*.jpg 에 있는 샘플들의 평균 임베딩(=프로토타입) 계산
    """
    support_path = os.path.join(cat_dir, "support_set")
    image_files = glob.glob(os.path.join(support_path, "*.jpg"))
    if len(image_files) == 0:
        print(f"[WARNING] No support images in {support_path}. Returning zero vector.")
        return torch.zeros(1, device=device)

    embeddings = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        if transform is not None:
            img_tensor = transform(img).unsqueeze(0).to(device)
        else:
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = resnet(img_tensor)  # (1, embed_dim)
        embeddings.append(emb)

    embeddings = torch.cat(embeddings, dim=0)  # (N, embed_dim)
    prototype = embeddings.mean(dim=0)         # (embed_dim,)
    return prototype

##############################
# 3) 추론(거리 계산) 함수
##############################
def classify_cat(embedding, prototypes):
    """
    embedding: (1, embed_dim) - 분류할 고양이 임베딩
    prototypes: (2, embed_dim) - cat1, cat2의 프로토타입 스택
    return: (predicted_class_idx, distance_list)
    """
    # 거리 계산 (유클리디안 예시)
    # embedding.shape   = (1, D)
    # prototypes.shape = (2, D)  → cdist → (1,2)
    dist = torch.cdist(embedding, prototypes.unsqueeze(0), p=2).squeeze(0)  # (2,)
    pred_idx = torch.argmin(dist).item()
    return pred_idx, dist  # pred_idx in [0,1], dist: Tensor(2,)

##############################
# 4) 실제 라이브 스트림 코드
##############################
def detect_cats_and_classify(
    yolo_model,
    meta_resnet,
    cat1_dir, cat2_dir,
    device='cpu'
):
    """
    - yolo_model: COCO로 학습된 YOLOv5 (cat=15)로 고양이 검출
    - meta_resnet: 메타 러닝(프로토타입)으로 학습된 임베딩 추출용 ResNet
    - cat1_dir, cat2_dir: 각 클래스(고양이1, 고양이2) support_set 폴더가 있는 디렉토리
    """

    # 1) 프로토타입 계산
    transform_proto = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    cat1_proto = compute_prototypes(meta_resnet, cat1_dir, device, transform_proto)
    cat2_proto = compute_prototypes(meta_resnet, cat2_dir, device, transform_proto)
    # prototypes.shape = (2, embed_dim)
    prototypes = torch.stack([cat1_proto, cat2_proto], dim=0)

    # 분류용 transform (Crop된 이미지)
    transform_infer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    # 2) 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    # 클래스 이름 매핑
    class_names = ["Cat1", "Cat2"]

    print("라이브 스트리밍: 고양이 감지(YOLO) + 메타러닝 분류(cat1 vs cat2). ESC 또는 q로 종료.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임을 가져오지 못했습니다.")
            break

        # 3) YOLOv5로 감지 수행
        results = yolo_model(frame)

        # 고양이만 필터링 (COCO: cat=15)
        cat_detections = results.xyxy[0][results.xyxy[0][:, -1] == 15]

        # 4) 감지된 고양이 처리
        for i, (x_min, y_min, x_max, y_max, conf, cls_id) in enumerate(cat_detections):
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            # 바운딩 박스 시각화
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            # YOLO confidence
            cat_label = f"Cat:{conf:.2f}"
            cv2.putText(frame, cat_label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # 5) 고양이 Crop → 메타 분류
            cropped = frame[y_min:y_max, x_min:x_max]
            # BGR → RGB
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            # transform
            input_tensor = transform_infer(Image.fromarray(cropped_rgb)).unsqueeze(0).to(device)

            # 6) 임베딩 추출 + 프로토타입 거리 계산
            with torch.no_grad():
                emb = meta_resnet(input_tensor)  # (1, embed_dim)
                pred_idx, dist_list = classify_cat(emb, prototypes)

            pred_class = class_names[pred_idx]
            distance_info = min(dist_list)  # [dist_cat1, dist_cat2]

            # 7) 분류 결과를 바운딩 박스 아래에 표시
            classify_text = f"{pred_class} Dist:{distance_info}"
            cv2.putText(frame, classify_text, (x_min, y_max + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 8) 결과 화면 표시
        cv2.imshow("Cat Detection + Classification", frame)

        # 9) 종료 조건 (ESC/q)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            print("종료 키 입력, 스트리밍 종료.")
            break

    cap.release()
    cv2.destroyAllWindows()

##############################
# 실행 메인 코드
##############################
if __name__ == "__main__":
    # 1) 디바이스 설정
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Device: {device}")

    # 2) YOLOv5 로드 (COCO 사전학습)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  
    yolo_model.eval()
    # GPU 사용 가능하면 yolo_model.to(device)도 가능 (추론 속도 개선)

    # 3) 메타 러닝 ResNet 로드
    meta_model_path = "./prototypical_network_resnet50_epoch_10.pth"  # 사용자 환경에 맞게 수정
    meta_resnet = load_meta_resnet(model_path=meta_model_path, device=device)

    # 4) cat1, cat2 디렉토리 설정 (support_set 포함)
    cat1_dir = "./cat1_images"
    cat2_dir = "./cat2_images"

    # 5) 라이브 스트림에서 고양이 감지 + cat1/cat2 분류
    detect_cats_and_classify(
        yolo_model=yolo_model,
        meta_resnet=meta_resnet,
        cat1_dir=cat1_dir,
        cat2_dir=cat2_dir,
        device=device
    )

output_dir = "./query_images"
os.makedirs(output_dir, exist_ok=True)

# Faster R-CNN 모델 로드
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()  # 평가 모드로 설정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# COCO 데이터셋의 클래스 인덱스
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# 고양이 클래스 ID
CAT_CLASS_ID = 17

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # BGR -> RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 이미지를 텐서로 변환
    img_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(device)

    # 모델 추론
    with torch.no_grad():
        predictions = model(img_tensor)

    # 예측 결과
    boxes = predictions[0]['boxes'].cpu().numpy()  # 바운딩 박스 좌표
    labels = predictions[0]['labels'].cpu().numpy()  # 클래스 라벨
    scores = predictions[0]['scores'].cpu().numpy()  # 신뢰도 점수

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label == CAT_CLASS_ID and score > 0.5:  # 고양이이며 신뢰도 50% 이상인 경우
            x1, y1, x2, y2 = box.astype(int)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 글자 더 굵게, 검정색으로 표시
            cv2.putText(frame, f"Cat: {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # 색상: 검정색, 굵기: 3

            # 바운딩 박스 영역 저장
            cropped_img = frame[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(output_dir, f"cat_{timestamp}.jpg")
            cv2.imwrite(output_path, cropped_img)
            print(f"Saved: {output_path}")

    # 프레임 출력
    cv2.imshow("Livestream Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()