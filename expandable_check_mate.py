import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets"
CLOSE_BTN_ICON = ASSETS_DIR / "ic_cancel.png"
CLOSE_BTN_SIZE = 48
CLOSE_BTN_MARGIN = 20

# 1. 페이지 설정
st.set_page_config(
    page_title="Check Mate : 익스팬더블 배너",
    page_icon="✅",
    layout="wide",
)

# ── 화질 분석 함수 ──────────────────────────────────────────────
def evaluate_quality(pil_image):
    img_array = np.array(pil_image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    p_raw = np.mean(20 * np.log(np.abs(fshift) + 1))
    purity_score = max(0, min(100, 100 - (p_raw - 175.0) * 30))

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = max(0, min(100, lap_var / 8))

    is_blurry = clarity_score < 12
    is_pixelated = purity_score < 35
    quality_score = (purity_score * 0.7) + (clarity_score * 0.3)

    return is_blurry, is_pixelated, quality_score


# ── 배너 규격 정의 ──────────────────────────────────────────────
BANNER_SPECS = {
    "대표이미지": {
        "size": (686, 200),
        "max_kb": 150,
        "crop_right": 108,          # 우측 크롭 영역 (px)
        "crop_left": 0,
        "close_btn": False,         # 닫기버튼 영역 없음
        "description": "686 × 200 px  |  150 KB 이하  |  우측 크롭 108 px",
    },
    "확장이미지": {
        "size": (686, 380),
        "max_kb": 150,
        "crop_right": 0,
        "crop_left": 0,
        "close_btn": True,          # 우상단 닫기버튼 영역 표시
        "close_btn_size": CLOSE_BTN_MARGIN + CLOSE_BTN_SIZE,  # 닫기버튼 영역 (px)
        "description": "686 × 380 px  |  150 KB 이하  |  우상단 닫기버튼 영역 주의",
    },
}


# ── 가이드 오버레이 함수 ────────────────────────────────────────
def apply_guide_overlay(pil_image, banner_type):
    cfg = BANNER_SPECS[banner_type]
    width, height = pil_image.size
    canvas = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    red     = (255,  50,  50,  90)   # 닫기버튼 / 크롭 위험 영역
    emerald = ( 50, 255, 170,  76)   # 안전여백 힌트 (미사용 시 생략 가능)

    if banner_type == "대표이미지":
        crop_r = cfg["crop_right"]
        # 우측 크롭 영역 (빨간색)
        draw.rectangle([(width - crop_r, 0), (width, height)], fill=red)
        # 크롭 경계선
        draw.line([(width - crop_r, 0), (width - crop_r, height)],
                  fill=(255, 80, 80, 200), width=2)

    elif banner_type == "확장이미지" and cfg.get("close_btn"):
        btn = cfg["close_btn_size"]
        # 우상단 닫기버튼 영역 (빨간색)
        draw.rectangle([(width - btn, 0), (width, btn)], fill=red)
        close_icon = Image.open(CLOSE_BTN_ICON).convert("RGBA").resize(
            (CLOSE_BTN_SIZE, CLOSE_BTN_SIZE), Image.Resampling.LANCZOS
        )
        x = width - CLOSE_BTN_MARGIN - CLOSE_BTN_SIZE
        y = CLOSE_BTN_MARGIN
        overlay.paste(close_icon, (x, y), close_icon)

    return Image.alpha_composite(canvas, overlay).convert("RGB")


# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }

.stApp { background-color: #111111; }
h1, h2, h3, h4 { color: #FFFFFF !important; }

[data-testid="stCaptionContainer"],
.stCaption, .stCaption div, .stCaption p {
    color: #FFFFFF !important; opacity: 1 !important;
}
[data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important; font-weight: 500 !important;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-border"] {
    background-color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #FFFFFF !important;
}
button[data-baseweb="tab"] {
    color: #AAAAAA !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-bottom-color: #FFFFFF !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid #FFFFFF !important;
}

/* 검수 결과 */
.check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
.check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
.status-text { font-size: 0.9rem; color: #AAAAAA; }

/* 가이드 컨테이너 */
.guide-container {
    background-color: #1E1E1E;
    padding: 15px 25px;
    border-radius: 12px;
    border: 1px solid #333333;
    margin-bottom: 25px;
}
.guide-row {
    display: flex; flex-wrap: wrap;
    align-items: center; gap: 20px;
    margin-bottom: 8px;
}
.guide-item {
    display: flex; align-items: center;
    font-size: 0.85rem; color: #DDDDDD;
}
.color-box {
    width: 16px; height: 16px;
    border-radius: 4px; margin-right: 8px; flex-shrink: 0;
}
.warning-text {
    font-size: 0.85rem; font-weight: bold;
    display: flex; align-items: center; gap: 5px;
}

/* 규격 뱃지 */
.spec-badge {
    display: inline-block;
    background: #2A2A2A;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #BBBBBB;
    margin-bottom: 16px;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #333333;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] .stRadio label { color: #DDDDDD !important; }
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; margin-bottom: 20px !important; }
[data-testid="stSidebar"] li { color: #DDDDDD !important; margin-bottom: 8px !important; }
[data-testid="stSidebar"] .stMarkdown { margin-bottom: 0px !important; }

#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
.stImage { display: flex; justify-content: center; }
</style>
""", unsafe_allow_html=True)


# ── 메인 UI ────────────────────────────────────────────────────
st.title("Check Mate : 익스팬더블 배너 검수")
st.caption("익스팬더블 광고 배너 소재 품질 및 규격 사전 검증 프로그램")

# ── 탭 구성 ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📌 대표이미지 검수", "📐 확장이미지 검수"])


def render_tab(banner_type):
    cfg = BANNER_SPECS[banner_type]
    exp_w, exp_h = cfg["size"]

    # 가이드 범례
    if banner_type == "대표이미지":
        guide_html = """
        <div class="guide-container">
          <div class="guide-row">
            <div class="guide-item">
              <div class="color-box" style="background:rgba(255,50,50,0.8);"></div>우측 크롭 영역 (108px)
            </div>
            <div class="warning-text" style="color:#DDDDDD;">
              ⚠️ 문구·메인 상품·비주얼이 우측 크롭 영역에 걸리지 않도록 확인하세요!
            </div>
          </div>
          <div class="guide-row">
            <div class="warning-text" style="color:#FF5252;">
              ⚠️ 광고 플래그는 시스템 자동 부착 — 이미지에 광고 문구가 포함되어 있으면 안됩니다!
            </div>
          </div>
        </div>
        """
    else:
        guide_html = """
        <div class="guide-container">
          <div class="guide-row">
            <div class="guide-item">
              <div class="color-box" style="background:rgba(255,50,50,0.8);"></div>우상단 닫기버튼 영역
            </div>
            <div class="warning-text" style="color:#DDDDDD;">
              ⚠️ 비주얼 복잡도로 닫기버튼 가독성을 해치지 않는지 직접 확인하세요!
            </div>
          </div>
          <div class="guide-row">
            <div class="warning-text" style="color:#FF5252;">
              ⚠️ 광고 플래그는 시스템 자동 부착 — 이미지에 광고 문구가 포함되어 있으면 안됩니다!
            </div>
          </div>
        </div>
        """

    st.markdown(guide_html, unsafe_allow_html=True)

    # 파일 업로더
    uploaded = st.file_uploader(
        f"{banner_type} 시안 이미지를 업로드하세요",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{banner_type}",
    )

    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    actual_w, actual_h = image.size
    file_size_kb = uploaded.size / 1024

    is_dim_valid = (actual_w, actual_h) == (exp_w, exp_h)
    is_size_valid = file_size_kb <= cfg["max_kb"]

    with st.spinner("이미지 품질을 분석 중입니다..."):
        is_blurry, is_pixelated, quality_score = evaluate_quality(image)

    # ── 검수 결과 3열 ──
    col1, col2, col3 = st.columns(3)

    with col1:
        css = "check-pass" if is_dim_valid else "check-fail"
        label = "✅ 규격 통과" if is_dim_valid else "❌ 규격 오류"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">업로드: {actual_w}×{actual_h}px'
            f'  |  권장: {exp_w}×{exp_h}px</div>',
            unsafe_allow_html=True,
        )

    with col2:
        css = "check-pass" if is_size_valid else "check-fail"
        label = "✅ 용량 적합" if is_size_valid else "❌ 용량 초과"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">{file_size_kb:.1f} KB  |  제한: {cfg["max_kb"]} KB</div>',
            unsafe_allow_html=True,
        )

    with col3:
        if not is_blurry and not is_pixelated and quality_score >= 60:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-text">품질 점수: {quality_score:.0f}점</div>',
                unsafe_allow_html=True,
            )
            st.warning(
                "화질 점수가 기준(60점)에 미달되었습니다.\n\n"
                "고화질 원본 이미지로 교체해 주시고,\n"
                "동일한 경고가 뜬다면 UX디자인팀에 검수 요청을 해주세요."
            )

    st.divider()

    # ── 오버레이 프리뷰 ──
    preview = apply_guide_overlay(image, banner_type)

    if banner_type == "대표이미지":
        caption = "대표이미지 가이드 프리뷰 — 우측 크롭 영역(빨간색) 침범 여부를 직접 확인하세요"
    else:
        caption = "확장이미지 가이드 프리뷰 — 우상단 닫기버튼 영역(빨간색) 가독성을 직접 확인하세요"

    st.image(preview, caption=caption, width=actual_w)


with tab1:
    render_tab("대표이미지")

with tab2:
    render_tab("확장이미지")
