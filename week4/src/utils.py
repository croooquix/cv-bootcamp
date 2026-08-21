import os
import yaml
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_config(config_path="week4/config/search_config.yaml"):
    """YAML 설정 파일을 로드합니다."""
    if not os.path.exists(config_path):
        # 상위 디렉토리 기준 경로 처리
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(os.path.join(current_dir, "../config/search_config.yaml"))
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def crop_with_padding(image, bbox, padding_ratio=0.10):
    """
    [멘토 피드백 #1 반영] 
    YOLO의 타이트한 바운딩 박스로 인한 실루엣/깃 잘림을 방지하기 위해 
    상하좌우에 padding_ratio(예: 10%)만큼의 여백을 안전하게 추가하여 Crop합니다.
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_w = int(box_w * padding_ratio)
    pad_h = int(box_h * padding_ratio)
    
    # 이미지 경계(0 ~ w, 0 ~ h) 내에서 패딩 좌표 계산
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y2 + pad_h)
    
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop_img

def normalize_vector(vector):
    """
    [멘토 피드백 #2 반영] 
    Faiss의 IndexFlatIP(Inner Product)에서 정확한 코사인 유사도를 구하기 위해 
    벡터의 L2 Norm 길이를 1로 정규화합니다.
    """
    vector = np.array(vector, dtype=np.float32)
    if vector.ndim == 1:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    else:
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        vector = vector / norm
    return vector

def load_clip_model(model_name="openai/clip-vit-base-patch32", device=None):
    """CLIP Processor와 Model을 로드합니다."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"[*] CLIP 모델 로딩 중 ({model_name}) on device: {device}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device

def extract_image_embedding(image, processor, model, device):
    """
    OpenCV BGR 이미지 또는 PIL 이미지를 받아 CLIP Image Encoder로 
    512차원 L2 정규화 특징 벡터를 추출합니다.
    """
    if isinstance(image, np.ndarray):
        # OpenCV BGR -> RGB -> PIL Image
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
    else:
        pil_img = image

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    # Tensor -> Numpy -> L2 Normalization
    features_np = image_features.cpu().numpy().astype(np.float32)
    return normalize_vector(features_np)

def extract_text_embedding(text_query, processor, model, device):
    """
    텍스트 쿼리(예: '검은색 롱 가죽 코트')를 받아 CLIP Text Encoder로 
    512차원 L2 정규화 특징 벡터를 추출합니다.
    """
    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    features_np = text_features.cpu().numpy().astype(np.float32)
    return normalize_vector(features_np)
