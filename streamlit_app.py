import streamlit as st
import numpy as np
from PIL import Image
import pickle
import requests
import os

# =========================
# 1. 구글 드라이브에서 모델 가져오기
# =========================

# 🔸 구글 드라이브에서 공유한 model.pkl의 파일 ID 넣기
# 예: https://drive.google.com/file/d/여기123abc아이디/view?usp=sharing
FILE_ID = "1QPRXxwHljOWE7mOLbwZZtpvBZBpJq4ei"  # 꼭 바꿔줘!!
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

MODEL_PATH = "model.pkl"


def download_model_if_needed():
    """로컬에 model.pkl 없으면 구글 드라이브에서 한 번 다운로드"""
    if os.path.exists(MODEL_PATH):
        return
    r = requests.get(GDRIVE_URL)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)


try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache


@cache_resource
def load_model():
    download_model_if_needed()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# =========================
# 2. 왕 정보 (이미지 2장 + 설명)
# =========================

KING_INFO = {
    "나폴레옹": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/5/50/"
            "Jacques-Louis_David_-_The_Emperor_Napoleon_in_His_Study_at_the_Tuileries_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/1/18/"
            "Napoleon_Bonaparte_by_Antoine-Jean_Gros%2C_1802.png",
        ],
        "desc": "프랑스의 황제로서 유럽 전역에 큰 영향을 끼친 전략가.",
    },
    "유스티니아누스 1세": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/d/d6/Justinian_I_mosaic.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/0/0e/Justinianus_I.jpg",
        ],
        "desc": "동로마 제국의 황제로, 로마법 대전 편찬과 제국 재통일을 추진한 인물.",
    },
    "칭기즈칸": {
        "images": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/"
            "YuanEmperorAlbumGenghisPortrait.jpg/440px-YuanEmperorAlbumGenghisPortrait.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/94/"
            "Genghis_Khan%2C_National_Museum_of_Mongolia.jpg",
        ],
        "desc": "몽골 제국의 창건자로, 세계 역사상 가장 큰 제국 중 하나를 세운 정복자.",
    },
}

KING_NAMES = list(KING_INFO.keys())
IMG_SIZE = (64, 64)  # 네가 학습할 때 쓴 이미지 크기랑 맞추기


# =========================
# 3. 전처리 & 예측 함수
# =========================

def preprocess_image(img: Image.Image) -> np.ndarray:
    """훈련 때랑 똑같이 전처리해야 함"""
    img = img.convert("L")        # 흑백
    img = img.resize(IMG_SIZE)    # (64, 64)로 리사이즈

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.flatten()           # 1차원 벡터로
    return arr


def predict_proba(model, img: Image.Image) -> np.ndarray:
    x = preprocess_image(img)
    x = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
    else:
        # predict_proba 없으면 decision_function을 소프트맥스로 변환
        scores = model.decision_function(x)[0]
        scores = np.array(scores, dtype="float32")
        exp = np.exp(scores - scores.max())
        proba = exp / exp.sum()

    proba = np.array(proba, dtype="float32")
    proba = proba / proba.sum()
    return proba


# =========================
# 4. Streamlit 메인 앱
# =========================

def main():
    st.set_page_config(page_title="역사 인물 닮은꼴 테스트", page_icon="👑")
    st.title("👑 역사 인물 닮은꼴 테스트")
    st.write(
        "사진을 찍거나 업로드하면 **나폴레옹 / 유스티니아누스 1세 / 칭기즈칸** 중 "
        "누구랑 가장 닮았는지 보여줄게!"
    )

    # 모델 로딩 (드라이브에서 필요시 자동 다운로드)
    model = load_model()

    tab_cam, tab_upload = st.tabs(["📸 사진 찍기", "📁 사진 업로드"])
    img = None

    with tab_cam:
        cam = st.camera_input("카메라로 촬영하기")
        if cam is not None:
            img = Image.open(cam)

    with tab_upload:
        up = st.file_uploader(
            "사진 업로드 (jpg, jpeg, png)",
            type=["jpg", "jpeg", "png"],
            key="img_uploader",
        )
        if up is not None:
            img = Image.open(up)

    if img is not None:
        st.subheader("입력한 사진")
        st.image(img, use_column_width=True)

        if st.button("🔍 닮은 인물 분석하기"):
            proba = predict_proba(model, img)

            if len(proba) != len(KING_NAMES):
                st.error(
                    "⚠️ 모델이 가진 클래스 수와 KING_INFO 인물 수가 달라.\n"
                    "모델 학습할 때 클래스 순서/개수가 지금 이름 리스트랑 맞는지 확인해야 해!"
                )
                return

            # 가장 확률 높은 인물
            best_idx = int(np.argmax(proba))
            best_name = KING_NAMES[best_idx]
            best_percent = float(proba[best_idx] * 100)

            st.success(f"가장 닮은 인물은 **{best_name} ({best_percent:.1f}%)** 입니다!")

            # 선택된 인물 이미지 2장 출력
            st.subheader(f"📸 {best_name} 이미지")
            for url in KING_INFO[best_name]["images"]:
                st.image(url, use_column_width=True)

            # 설명
            st.subheader(f"📝 {best_name} 설명")
            st.write(KING_INFO[best_name]["desc"])

            # 전체 확률 막대
            st.subheader("📊 전체 확률")
            for name, p in zip(KING_NAMES, proba):
                st.write(f"**{name}**: {float(p) * 100:.1f}%")
                st.progress(float(p))


if __name__ == "__main__":
    main()
