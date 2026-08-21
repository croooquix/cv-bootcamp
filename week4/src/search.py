import os
import argparse
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
    extract_text_embedding
)

class FashionSearchEngine:
    def __init__(self, config_path="week4/config/search_config.yaml"):
        self.config = load_config(config_path)
        self.index_path = os.path.abspath(self.config["paths"]["index_save_path"])
        self.meta_path = os.path.abspath(self.config["paths"]["metadata_save_path"])
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"[!] 인덱스 파일이 존재하지 않습니다. 먼저 python week4/src/build_index.py를 실행하여 인덱스를 구축해 주세요.\n"
                f"    - Index Path: {self.index_path}\n    - Meta Path: {self.meta_path}"
            )
            
        print("[*] Faiss Index 및 메타데이터 로드 중...")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 모델 로드
        yolo_path = self.config["paths"]["yolo_model_path"]
        if not os.path.exists(yolo_path):
            yolo_path = self.config["paths"]["yolo_fallback_path"]
        # 텍스트 검색에는 YOLO가 필요하지 않으므로 이미지 검색 시점에 로드합니다.
        self.yolo_model = None
        self.processor, self.clip_model, self.device = load_clip_model(self.config["clip"]["model_name"])
        # self.yolo_model = YOLO(yolo_path)
        # self.processor, self.clip_model, self.device = load_clip_model(self.config["clip"]["model_name"])
        self.padding_ratio = self.config["crop"]["padding_ratio"]


    def _get_yolo_model(self):
        """이미지 검색이 처음 호출될 때 YOLO 모델을 한 번만 로드합니다."""
        if self.yolo_model is None:
            yolo_path = os.environ.get(
                "YOLO_MODEL_PATH",
                self.config["paths"]["yolo_model_path"],
            )

            if not os.path.exists(yolo_path):
                yolo_path = self.config["paths"]["yolo_fallback_path"]

            print(f"[*] YOLO 모델 로딩 중: {yolo_path}")
            self.yolo_model = YOLO(yolo_path)

        return self.yolo_model

    def search_by_image(self, image_input, top_k=5):
        """
        이미지를 입력받아 유사한 의류 상품 Top-K를 검색합니다.
        image_input: 파일 경로(str) 또는 OpenCV BGR 이미지(np.ndarray)
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {image_input}")
        else:
            image = image_input

        # 1. YOLO로 쿼리 이미지 내 의류 검출 및 패딩 크롭 (멘토 피드백 #1)
        yolo_model = self._get_yolo_model()
        results = yolo_model(image, verbose=False)
        cropped_img = None
        detected_class = "Unknown"

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cropped_img = crop_with_padding(image, (x1, y1, x2, y2), padding_ratio=self.padding_ratio)
                    detected_class = result.names[int(box.cls[0])]
                    break
            if cropped_img is not None:
                break

        if cropped_img is None or cropped_img.size == 0:
            cropped_img = image # Fallback: 전체 이미지 사용

        # 2. CLIP Image Embedding & L2 Normalization (멘토 피드백 #2)
        query_vec = extract_image_embedding(cropped_img, self.processor, self.clip_model, self.device)

        # 3. Faiss Cosine Similarity 검색
        scores, indices = self.index.search(query_vec, top_k)

        search_results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                item_meta = self.metadata[idx].copy()
                item_meta["similarity_score"] = float(score)
                search_results.append(item_meta)

        return search_results, detected_class, cropped_img

    def search_by_text(self, text_query, top_k=5):
        """
        텍스트 쿼리(예: '검은색 롱 가죽 코트')를 입력받아 유사한 의류 상품 Top-K를 검색합니다.
        """
        # 1. CLIP Text Embedding & L2 Normalization
        query_vec = extract_text_embedding(text_query, self.processor, self.clip_model, self.device)

        # 2. Faiss Cosine Similarity 검색
        scores, indices = self.index.search(query_vec, top_k)

        search_results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                item_meta = self.metadata[idx].copy()
                item_meta["similarity_score"] = float(score)
                search_results.append(item_meta)

        return search_results

def main():
    parser = argparse.ArgumentParser(description="Week 4 Multimodal Fashion Search Engine")
    parser.add_argument("--image", type=str, help="검색할 쿼리 이미지 파일 경로")
    parser.add_argument("--text", type=str, help="검색할 텍스트 쿼리 (예: '검은색 롱 가죽 코트')")
    parser.add_argument("--top_k", type=int, default=5, help="반환할 유사 상품 개수")
    args = parser.parse_args()

    engine = FashionSearchEngine()

    if args.image:
        print(f"\n🖼️  이미지 기반 검색 실행: {args.image}")
        results, detected_class, _ = engine.search_by_image(args.image, top_k=args.top_k)
        print(f"[*] 검출된 객체 라벨: {detected_class}")
        print("-" * 60)
        for r in results:
            print(f" - [{r['class_name']}] 코사인 유사도: {r['similarity_score']*100:.2f}% | 파일: {r['filename']}")

    elif args.text:
        print(f"\n🔤 텍스트 기반 검색 실행: '{args.text}'")
        results = engine.search_by_text(args.text, top_k=args.top_k)
        print("-" * 60)
        for r in results:
            print(f" - [{r['class_name']}] 코사인 유사도: {r['similarity_score']*100:.2f}% | 파일: {r['filename']}")
    else:
        print("[!] --image 혹은 --text 옵션을 지정하여 실행해 주세요.")
        print("    예시 1: python week4/src/search.py --image week3/data/test/images/some_img.jpg")
        print("    예시 2: python week4/src/search.py --text '검은색 롱 가죽 코트'")

if __name__ == "__main__":
    main()
