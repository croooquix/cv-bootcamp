import ssl
ssl.SSLContext._load_windows_store_certs = lambda self, storename, purpose: None

import os
import sys
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# week4/src 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from search import FashionSearchEngine

# 페이지 기본 설정
st.set_page_config(
    page_title="이커머스 멀티모달 패션 검색엔진",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# [멘토 피드백 #3 반영] @st.cache_resource 데코레이터를 사용하여
# YOLO, CLIP, Faiss 모델 로딩을 최초 1회만 수행하고 메모리에 캐싱합니다.
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="🚀 AI 모델 및 벡터 데이터베이스를 불러오는 중입니다...")
def get_search_engine():
    config_path = os.path.join(os.path.dirname(__file__), "config/search_config.yaml")
    return FashionSearchEngine(config_path=config_path)

def main():
    # 헤더
    st.title("🛍️ 멀티모달 패션 AI 추천 & 비주얼 검색엔진")
    st.caption("YOLOv8 의류 검출 + CLIP 멀티모달 임베딩 + Faiss 벡터 검색 기반의 실시간 추천 서비스")
    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 검색 설정")
        top_k = st.slider("추천 아이템 개수 (Top-K)", min_value=1, max_value=10, value=5, step=1)

    # 엔진 로드
    try:
        engine = get_search_engine()
    except Exception as e:
        st.error(f"❌ 검색 엔진 로드 실패: {e}")
        st.warning("먼저 `python week4/src/build_index.py`를 실행하여 인덱스를 구축해 주세요.")
        return

    # 탭 구성: 이미지 검색 vs 텍스트 검색
    tab1, tab2 = st.tabs(["🖼️ 이미지로 검색 (Visual Search)", "🔤 텍스트로 검색 (Semantic Text Search)"])

    # ===================================================================
    # TAB 1: 이미지 기반 검색 (Visual Search)
    # ===================================================================
    with tab1:
        st.subheader("카메라 샷 또는 패션 이미지 업로드")
        uploaded_file = st.file_uploader("검색할 이미지를 업로드하세요 (.jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 2])

            # 이미지 읽기
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            with col1:
                st.image(image_rgb, caption="업로드 원본 이미지", use_container_width=True)

            # 검색 수행
            with st.spinner("AI가 옷 영역을 포착하고 유사 아이템을 탐색 중입니다..."):
                results, detected_class, cropped_img = engine.search_by_image(image_bgr, top_k=top_k)

            with col1:
                if cropped_img is not None and cropped_img.size > 0:
                    crop_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"YOLO 검출 영역 (10% 여백 적용): {detected_class}", width=200)

            st.markdown("### 🎯 추천 유사 상품 (Top Match)")
            
            # 결과 카드 컬럼 구성
            cols = st.columns(top_k)
            for idx, (col, r) in enumerate(zip(cols, results)):
                with col:
                    sim_pct = r['similarity_score'] * 100
                    st.metric(label=f"Rank #{idx+1}", value=f"{sim_pct:.1f}% Match")
                    
                    if os.path.exists(r['image_path']):
                        img = Image.open(r['image_path'])
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("이미지 파일 없음")
                        
                    st.caption(f"**{r['class_name'].upper()}**")
                    st.caption(f"📄 `{r['filename']}`")

    # ===================================================================
    # TAB 2: 텍스트 기반 검색 (Semantic Text Search)
    # ===================================================================
    with tab2:
        st.subheader("원하는 스타일이나 의류 특징을 텍스트로 입력하세요")
        st.info("💡 **CLIP 이론 팁**: 텍스트-이미지 간 코사인 유사도는 양 종단(Text-Vision) 인코더 특성상 원시 수치가 **25~35% 대가 최고 상위 매칭 수치**입니다. (30% 이상 시 높은 연관성)")
        
        # 예시 검색어 버튼
        st.caption("💡 추천 예시 키워드:")
        example_cols = st.columns(4)
        selected_example = None
        if example_cols[0].button("🧥 롱 코트 (Long Coat)"):
            selected_example = "long coat outerwear"
        if example_cols[1].button("👗 패션 드레스 (Fashion Dress)"):
            selected_example = "elegant dress"
        if example_cols[2].button("👖 데님 팬츠 (Denim Pants)"):
            selected_example = "denim pants"
        if example_cols[3].button("👕 티셔츠 (T-shirt)"):
            selected_example = "casual t-shirt"

        text_input = st.text_input(
            "검색 쿼리 입력 (영어 또는 한국어)", 
            value=selected_example if selected_example else "",
            placeholder="예: black leather jacket, denim pants, red dress..."
        )

        if text_input:
            with st.spinner(f"'{text_input}' 스타일의 상품을 벡터 공간에서 매칭 중입니다..."):
                results = engine.search_by_text(text_input, top_k=top_k)

            st.markdown(f"### 🎯 '{text_input}' 검색 결과 (Top Match)")
            cols = st.columns(top_k)
            for idx, (col, r) in enumerate(zip(cols, results)):
                with col:
                    # 텍스트-이미지 모달리티 갭 반영 보정 시각화
                    raw_score = r['similarity_score']
                    # CLIP Text-Image 스케일 보정 (0.25 -> 80%, 0.35 -> 100%)
                    relative_match = min(100.0, max(0.0, (raw_score - 0.15) / 0.20 * 100))
                    
                    st.metric(label=f"Rank #{idx+1}", value=f"{relative_match:.1f}% Match", delta=f"Raw: {raw_score*100:.1f}%")
                    
                    if os.path.exists(r['image_path']):
                        img = Image.open(r['image_path'])
                        st.image(img, use_container_width=True)
                    else:
                        st.warning("이미지 파일 없음")
                        
                    st.caption(f"**{r['class_name'].upper()}**")
                    st.caption(f"📄 `{r['filename']}`")

if __name__ == "__main__":
    main()
