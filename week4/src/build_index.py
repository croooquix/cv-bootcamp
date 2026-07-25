import os
import glob
import json
import cv2
import numpy as np
import faiss
from ultralytics import YOLO
from utils import (
    load_config, 
    crop_with_padding, 
    load_clip_model, 
    extract_image_embedding, 
    normalize_vector
)

def main():
    print("=" * 60)
    print("🛍️ [Week 4] Faiss Vector DB 및 CLIP 인덱스 생성 (Offline Indexing)")
    print("=" * 60)

    # 1. 설정 로드
    config = load_config()
    yolo_path = config["paths"]["yolo_model_path"]
    if not os.path.exists(yolo_path):
        print(f"[!] {yolo_path} 모델이 없어 기본 사전학습 모델({config['paths']['yolo_fallback_path']})을 사용합니다.")
        yolo_path = config["paths"]["yolo_fallback_path"]

    dataset_dir = config["paths"]["dataset_images_dir"]
    padding_ratio = config["crop"]["padding_ratio"]
    embedding_dim = config["clip"]["embedding_dim"]

    # 출력 저장 디렉토리 생성
    index_save_path = os.path.abspath(config["paths"]["index_save_path"])
    meta_save_path = os.path.abspath(config["paths"]["metadata_save_path"])
    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)

    # 2. 모델 로드
    print(f"[*] YOLOv8 모델 로드 중: {yolo_path}")
    yolo_model = YOLO(yolo_path)

    processor, clip_model, device = load_clip_model(config["clip"]["model_name"])

    # 3. 이미지 파일 수집
    image_paths = glob.glob(os.path.join(dataset_dir, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_dir, "*.png")) + \
                  glob.glob(os.path.join(dataset_dir, "*.jpeg"))
    
    print(f"[*] 총 {len(image_paths)}개 이미지를 대상으로 인덱싱을 시작합니다.")

    embeddings_list = []
    metadata_list = []
    item_count = 0

    # 4. 이미지 순회 및 백그라운드 인덱싱
    for idx, img_path in enumerate(image_paths):
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = yolo_model(image, verbose=False)
        boxes_found = 0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.25: # 신뢰도 25% 미만 필터링
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_idx = int(box.cls[0])
                class_name = result.names[class_idx]

                # 멘토 피드백 #1 반영: 10% Margin 추가 Crop
                cropped_img = crop_with_padding(image, (x1, y1, x2, y2), padding_ratio=padding_ratio)

                if cropped_img.size == 0:
                    continue

                # CLIP 임베딩 추출 & L2 정규화 (멘토 피드백 #2)
                vec = extract_image_embedding(cropped_img, processor, clip_model, device)
                embeddings_list.append(vec.flatten())

                # 메타데이터 기록
                metadata_list.append({
                    "id": item_count,
                    "image_path": img_path,
                    "filename": os.path.basename(img_path),
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })
                
                item_count += 1
                boxes_found += 1

        # 검출된 객체가 없을 경우 전체 이미지 fallback 인덱싱
        if boxes_found == 0:
            vec = extract_image_embedding(image, processor, clip_model, device)
            embeddings_list.append(vec.flatten())
            metadata_list.append({
                "id": item_count,
                "image_path": img_path,
                "filename": os.path.basename(img_path),
                "class_name": "full_image",
                "confidence": 1.0,
                "bbox": [0, 0, image.shape[1], image.shape[0]]
            })
            item_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"   - Progress: [{idx + 1}/{len(image_paths)}] 이미지 완료 (누적 {item_count}개 의류 벡터)")

    if not embeddings_list:
        print("[!] 인덱싱할 의류 벡터를 추출하지 못했습니다.")
        return

    # 5. Faiss IndexFlatIP (Inner Product = Cosine Similarity with L2 Normalization)
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    
    print(f"[*] Faiss IndexFlatIP 구축 중 (차원: {embedding_dim}, 총 벡터: {len(embeddings_matrix)}개)...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_matrix)

    # 6. 저장
    faiss.write_index(index, index_save_path)
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ 인덱싱 완료!")
    print(f"   - Faiss Index: {index_save_path}")
    print(f"   - Metadata: {meta_save_path}")
    print(f"   - 총 인덱싱된 의류 객체 수: {item_count}개")
    print("=" * 60)

if __name__ == "__main__":
    main()
