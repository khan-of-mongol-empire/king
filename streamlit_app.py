import streamlit as st
import numpy as np
from PIL import Image
import pickle

# 🔸 모델 파일 이름 (같은 폴더에 넣기)
MODEL_PATH = "왕분류.pkl"

# 🔸 왕(인물) 정보 — 이미지 2장씩 링크로 넣기
KING_INFO = {
    "나폴레옹": {
        "images": [
            "https://i.namu.wiki/i/Tjeg41KBODqBzsuHR4UFdRiXQOpc8ZzxAdszhgmmZS73vOqyoQG-BOIzBVw9x7MzlT-q4stS86gOiLvWkN6ECQ.webp",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6UF5G4IscpS-nqLZizjeGg0TgBR_wydMoO8Zhc9dHsQ&s=10"
        ],
        "desc": "한때 유럽 대부분을 통치했던 프랑스 제1제국의 황제."
    },
    "유스티니아누스 1세": {
        "images": [
            "https://i.namu.wiki/i/9zwcbXO46dFqk9DvRMSbegSDSSdoFVTJpD2cRS_yyA0BCq2b4nqlH1oKL-S7Q-pgruke5jeAsl5i163hN2D2fA2h-8ua-adJW49AROgNvSN1RBeFqhVlPa0NC6rSjXcuD8UnZ6w7bbUJFEa3TjJ1MQ.webp",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Justinian555AD.png/960px-Justinian555AD.png"
        ],
        "desc": "동로마의 전성기와 최대 영토를 달성한 황제."
    },
    "칭기즈칸": {
        "images": [
            "https://i.namu.wiki/i/gUku3ZIkztgmgwVVbMVmGZQ5QTFKwN1VXaS5Yxi1VUmyCi4K6yalUmqqto9kmd9mNqxtrp1kHF0KvgMZjJhZU5vhJlQpl42j3yRHNBAgQhEToie6F9owckbA5A6v-7qqCA6851jmc6N8os0GCTkLpQ.webp",
            "https://i.namu.wiki/i/S6aKb1LoOArJ35wzMnKDdsXWPs_Q563jAVkwj9nzF7IEdxlHZHslS8tcrsvSi_lv4enY2DZ3CIm13W2n5qBVrqDtUvB7YVBi-a3C4o4UBky71h3BpYutbJjJi3nc-GWzPFctpRV_BKPzu2yAUbDBsw.svg"
        ],
        "desc": "몽골 제국의 창건자로 세계 역사상 가장 큰 제국 중 하나를 세운 정복자."
    }
}

# 모델 클래스 이름 리스트
KING_NAMES = list(KING_INFO.keys())


# -------------------------------
# 모델 불러오기
# -------------------------------
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache

@cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# -------------------------------
# 이미지 전처리 (훈련 방식에 맞추기)
# -------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("L")  # 흑백 변환
    img = img.resize((64, 64))  # 모델이 학습한 크기와 일치해야 함

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.flatten()
    return arr


# -------------------------------
# 예측 확률 계산
# -------------------------------
def predict_proba(model, img: Image.Image) -> np.ndarray:
    x = preprocess_image(img)
    x = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
    else:
        scores = model.decision_function(x)[0]
        exp = np.exp(scores - scores.max())
        proba = exp / exp.sum()

    proba = np.array(proba, dtype="float32")
    proba = proba / proba.sum()
    return proba


# -------------------------------
# MAIN STREAMLIT APP
# -------------------------------
def main():
    st.set_page_config(page_title="역사 인물 닮은꼴 테스트", page_icon="👑")
    st.title("👑 역사 인물 닮은꼴 테스트")
    st.write("사진을 업로드하거나 촬영하면 **나폴레옹, 유스티니아누스 1세, 칭기즈칸 중 가장 닮은 인물**이 누구인지 알려줍니다!")

    model = load_model()

    # 입력 방식 탭
    tab_cam, tab_upload = st.tabs(["📸 사진 찍기", "📁 사진 업로드"])

    img = None

    # 카메라 입력
    with tab_cam:
        cam = st.camera_input("카메라로 촬영하기")
        if cam:
            img = Image.open(cam)

    # 파일 업로드
    with tab_upload:
        up = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up)

    # 분석
    if img is not None:
        st.subheader("입력된 사진")
        st.image(img, use_column_width=True)

        if st.button("🔍 닮은 인물 분석하기"):
            proba = predict_proba(model, img)

            if len(proba) != len(KING_NAMES):
                st.error("⚠️ 모델 클래스 수와 KING_INFO 인물 수가 다릅니다!")
                return

            # 가장 닮은 인물 선택
            best_idx = np.argmax(proba)
            best_name = KING_NAMES[best_idx]
            best_percent = proba[best_idx] * 100

            st.success(f"가장 닮은 인물은 **{best_name} ({best_percent:.1f}%)** 입니다!")

            # 인물 사진 2장 출력
            st.subheader(f"📸 {best_name} 이미지")
            for img_url in KING_INFO[best_name]["images"]:
                st.image(img_url, use_column_width=True)

            # 인물 설명
            st.subheader(f"📝 {best_name} 설명")
            st.write(KING_INFO[best_name]["desc"])

            # 확률 상세
            st.subheader("📊 전체 확률")
            for name, p in zip(KING_NAMES, proba):
                st.write(f"**{name}**: {p * 100:.1f}%")
                st.progress(float(p))


if __name__ == "__main__":
    main()
