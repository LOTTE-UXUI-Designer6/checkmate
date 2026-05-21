import io
import re
import unicodedata
import streamlit as st
import tempfile
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
        "crop_right": 108,
        "crop_left": 0,
        "close_btn": False,
        "description": "686 × 200 px  |  150 KB 이하  |  우측 크롭 108 px",
    },
    "확장이미지": {
        "size": (686, 380),
        "max_kb": 150,
        "crop_right": 0,
        "crop_left": 0,
        "close_btn": True,
        "close_btn_size": CLOSE_BTN_MARGIN + CLOSE_BTN_SIZE,
        "description": "686 × 380 px  |  150 KB 이하  |  우상단 닫기버튼 영역 주의",
    },
}

VIDEO_SPECS = {
    "min_size": (686, 380),
    "aspect_ratio": 16 / 9,
    "max_mb": 30,
    "max_duration_sec": 60,
    "formats": ["mp4", "avi"],
}

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXT = set(VIDEO_SPECS["formats"])
QUALITY_THRESHOLD = 60


def _norm_filename(filename):
    return unicodedata.normalize("NFC", Path(filename).name)


def serialize_uploaded_file(uploaded):
    return {
        "name": uploaded.name,
        "type": getattr(uploaded, "type", ""),
        "data": uploaded.getvalue(),
    }


def deserialize_uploaded_file(data):
    restored = io.BytesIO(data["data"])
    restored.name = data["name"]
    restored.type = data.get("type", "")
    return restored


def slot_from_filename(filename):
    name = _norm_filename(filename)
    if "확장형" in name or "확장후" in name:
        return "확장이미지"
    if "대표이미지" in name or "확장전" in name:
        return "대표이미지"
    if re.search(r"686\s*[x×]\s*380", name, re.IGNORECASE):
        return "확장이미지"
    if re.search(r"686\s*[x×]\s*200", name, re.IGNORECASE):
        return "대표이미지"
    return None


def match_image_slot(filename, width, height):
    by_name = slot_from_filename(filename)
    if by_name:
        return by_name
    rep_size = BANNER_SPECS["대표이미지"]["size"]
    exp_size = BANNER_SPECS["확장이미지"]["size"]
    if (width, height) == exp_size:
        return "확장이미지"
    if (width, height) == rep_size:
        return "대표이미지"
    return None


def analyze_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        Path(tmp_path).unlink(missing_ok=True)
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    Path(tmp_path).unlink(missing_ok=True)

    return {"width": width, "height": height, "duration": duration, "fps": fps}


def classify_batch_files(files):
    result = {
        "대표이미지": None,
        "확장이미지": None,
        "video": None,
        "warnings": [],
        "unmatched": [],
    }
    video_candidates = []
    for f in files:
        ext = Path(f.name).suffix.lstrip(".").lower()
        if ext in ALLOWED_VIDEO_EXT:
            video_candidates.append(f)
        elif ext in ALLOWED_IMAGE_EXT:
            try:
                f.seek(0)
                with Image.open(f) as im:
                    w, h = im.size
                f.seek(0)
            except OSError:
                result["warnings"].append(f"`{f.name}`: 이미지로 열 수 없습니다.")
                continue

            slot = match_image_slot(f.name, w, h)
            if slot is None:
                result["unmatched"].append((f.name, w, h))
                continue

            if result[slot] is not None:
                result["warnings"].append(
                    f"{slot} 파일이 여러 개입니다. `{result[slot].name}` → `{f.name}`(으)로 교체됩니다."
                )
            result[slot] = f
        else:
            result["warnings"].append(
                f"`{f.name}`: 이미지(PNG/JPG) 또는 영상(MP4/AVI)이 아닙니다."
            )

    if video_candidates:
        if len(video_candidates) > 1:
            for extra in video_candidates[:-1]:
                result["warnings"].append(
                    f"영상 파일이 여러 개입니다. `{extra.name}` → `{video_candidates[-1].name}`(으)로 교체됩니다."
                )
        result["video"] = video_candidates[-1]

    return result


def resolve_batch_upload(files):
    if not files:
        st.session_state["batch_prev_fps"] = []
        return []
    file_list = list(files)
    current_fps = [(f.name, f.size) for f in file_list]
    prev_fps = st.session_state.get("batch_prev_fps", [])

    if prev_fps and current_fps != prev_fps:
        st.session_state["batch_prev_fps"] = current_fps

    st.session_state["batch_prev_fps"] = current_fps
    return file_list


def _uploader_run() -> int:
    """초기화 버튼을 누를 때마다 증가하는 카운터 — uploader key suffix로 사용."""
    return st.session_state.get("uploader_run", 0)


def resolve_single_upload(uploaded, state_key):
    fp_key = f"{state_key}_fp"
    if not uploaded:
        st.session_state.pop(fp_key, None)
        return None
    fp = (uploaded.name, uploaded.size)
    st.session_state[fp_key] = fp
    return uploaded


# ── 가이드 오버레이 함수 ────────────────────────────────────────
def apply_guide_overlay(pil_image, banner_type):
    cfg = BANNER_SPECS[banner_type]
    width, height = pil_image.size
    canvas = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    purple = (160, 80, 255, 90)

    if banner_type == "대표이미지":
        crop_r = cfg["crop_right"]
        draw.rectangle([(width - crop_r, 0), (width, height)], fill=purple)
        draw.line([(width - crop_r, 0), (width - crop_r, height)],
                  fill=(180, 100, 255, 200), width=2)

    elif banner_type == "확장이미지" and cfg.get("close_btn"):
        btn = cfg["close_btn_size"]
        draw.rectangle([(width - btn, 0), (width, btn)], fill=purple)
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
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');

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
.stTabs [data-baseweb="tab-border"]    { background-color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #FFFFFF !important; }
button[data-baseweb="tab"] {
    color: #AAAAAA !important; font-size: 1rem !important;
    font-weight: 600 !important; border-bottom-color: #FFFFFF !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important; border-bottom: 2px solid #FFFFFF !important;
}

/* 검수 결과 */
.check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
.check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
.status-text { font-size: 0.9rem; color: #AAAAAA; }
.stButton > button {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    padding: 6px 10px !important;
    width: 72px !important;
    min-width: 56px !important;
    flex: none !important;
}
.guide-caption { color: #B26BFF; font-size: 1.05rem; margin-top: 12px; font-weight: 600; padding:0 20px 20px; text-align:center;}

/* 가이드 컨테이너 */
.guide-container {
    background-color: #1E1E1E; padding: 15px 25px;
    border-radius: 12px; border: 1px solid #333333; margin-bottom: 25px;
}
.guide-row {
    display: flex; flex-wrap: wrap;
    align-items: center; gap: 20px; margin-bottom: 8px;
}
.guide-item   { display: flex; align-items: center; font-size: 0.85rem; color: #DDDDDD; }
.color-box    { width: 16px; height: 16px; border-radius: 4px; margin-right: 8px; flex-shrink: 0; }
.warning-text { font-size: 0.85rem; font-weight: bold; display: flex; align-items: center; gap: 5px; }

/* 규격 뱃지 */
.spec-badge {
    display: inline-block; background: #2A2A2A; border: 1px solid #444;
    border-radius: 8px; padding: 6px 14px; font-size: 0.82rem;
    color: #BBBBBB; margin-bottom: 16px;
}

/* ── 사전 체크 공지 영역 ───────────────────────────────────── */
.precheck-box {
    background: #0E1A2B;
    border: 1px solid #1E3A5A;
    border-left: 4px solid #00B0FF;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 4px 0 24px 0;
}
.precheck-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.precheck-header-icon { font-size: 15px; }
.precheck-header-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #00B0FF;
    letter-spacing: -0.02em;
    text-transform: uppercase;
}
.precheck-list {
    list-style: none;
    padding: 0; margin: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.precheck-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.precheck-num {
    min-width: 22px; height: 22px;
    background: #1A3A5A;
    border: 1px solid #2A5A8A;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    color: #00B0FF;
    flex-shrink: 0;
    margin-top: 1px;
}
.precheck-text {
    font-size: 0.9rem;
    color: #ddeeff;
    line-height: 1.6;
}
.precheck-text .kw {
    display: inline-block;
    background: #1A3050;
    border: 1px solid #2A5080;
    border-radius: 4px;
    padding: 0px 7px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 2px;
}
.precheck-text .ext {
    display: inline-block;
    background: #1A2A40;
    border: 1px solid #2A4060;
    border-radius: 4px;
    padding: 0px 7px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #80CCFF;
    margin: 0 2px;
    font-family: monospace;
}

/* ── 사이드바 전체 ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0D0D0D !important;
    border-right: 1px solid #2A2A2A !important;
    min-width: 300px !important;
    max-width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { font-family: 'Noto Sans KR', sans-serif !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span { color: #f2f2f2 !important; }

/* LNB 커스텀 컴포넌트 */
.lnb-wrap { padding: 28px 20px 40px 20px; font-family: 'Noto Sans KR', sans-serif; }

/* 로고 영역 */
.lnb-logo {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 32px; padding-bottom: 20px;
    border-bottom: 1px solid #222222;
}
.lnb-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #A050FF 0%, #00B0FF 100%);
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; flex-shrink: 0;
}
.lnb-logo-text  { font-size: 0.95rem; font-weight: 700; color: #FFFFFF !important; letter-spacing: -0.3px; line-height: 1.2; }
.lnb-logo-sub   { font-size: 0.8rem; color: #eeeeee !important; font-weight: 400; margin-top: 2px; }

/* 섹션 블록 */
.lnb-section { margin-bottom: 24px; }
.lnb-section-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.lnb-section-icon   { font-size: 14px; opacity: 0.9; }
.lnb-section-title  { font-size: 0.85rem !important; font-weight: 700 !important; color: #888888 !important; letter-spacing: -0.02em; text-transform: uppercase; }

/* 규격 카드 */
.lnb-card {
    background: #161616; border: 1px solid #252525;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}
.lnb-card-title {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: #EEEEEE !important; margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.lnb-card-badge {
    display: inline-block; background: #1E1E1E; border: 1px solid #333;
    border-radius: 4px; padding: 1px 7px;
    font-size: 0.68rem !important; color: #888888 !important;
    font-weight: 500 !important; vertical-align: middle;
}

/* 규격 행 */
.lnb-spec-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 6px 0; border-bottom: 1px solid #1E1E1E;
}
.lnb-spec-row:last-child { border-bottom: none; padding-bottom: 0; }
.lnb-spec-label {
    font-size: 0.73rem !important; color: #777777 !important;
    font-weight: 500 !important; min-width: 52px; flex-shrink: 0; padding-top: 1px;
}
.lnb-spec-value { font-size: 0.73rem !important; color: #CCCCCC !important; font-weight: 400 !important; line-height: 1.5; }
.lnb-spec-value strong { color: #FFFFFF !important; font-weight: 600 !important; }
.lnb-spec-value .accent { color: #00E676 !important; font-weight: 600 !important; }
.lnb-spec-value .warn   { color: #FFB300 !important; font-weight: 600 !important; }
.lnb-spec-value .danger { color: #FF5252 !important; font-weight: 600 !important; }
.lnb-spec-value .purple { color: #BB88FF !important; font-weight: 600 !important; }

/* 구분선 */
.lnb-divider { border: none; border-top: 1px solid #1E1E1E; margin: 20px 0; }

/* 체크리스트 */
.lnb-checklist { list-style: none; padding: 0; margin: 0; }
.lnb-checklist li {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 5px 0; font-size: 0.73rem !important;
    color: #AAAAAA !important; line-height: 1.5;
    border-bottom: 1px solid #1A1A1A;
}
.lnb-checklist li:last-child { border-bottom: none; }
.lnb-check-dot { width: 5px; height: 5px; background: #444; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }

/* 팁·경고 박스 */
.lnb-tip {
    background: #0F1F17; border: 1px solid #1A3A28;
    border-left: 3px solid #00E676;
    border-radius: 8px; padding: 11px 14px; margin-top: 8px;
}
.lnb-tip p { font-size: 0.72rem !important; color: #88BB99 !important; margin: 0; line-height: 1.6; }
.lnb-tip-label { font-size: 0.68rem !important; font-weight: 700 !important; color: #00E676 !important; letter-spacing: 0.05em; margin-bottom: 5px !important; }

.lnb-warn {
    background: #1F1500; border: 1px solid #3A2800;
    border-left: 3px solid #FFB300;
    border-radius: 8px; padding: 11px 14px; margin-top: 8px;
}
.lnb-warn p { font-size: 0.72rem !important; color: #BBAA77 !important; margin: 0; line-height: 1.6; }
.lnb-warn-label { font-size: 0.68rem !important; font-weight: 700 !important; color: #FFB300 !important; letter-spacing: 0.05em; margin-bottom: 5px !important; }

.lnb-danger {
    background: #1F0A0A; border: 1px solid #3A1010;
    border-left: 3px solid #FF5252;
    border-radius: 8px; padding: 11px 14px; margin-top: 8px;
}
.lnb-danger p { font-size: 0.72rem !important; color: #CC8888 !important; margin: 0; line-height: 1.6; }
.lnb-danger-label { font-size: 0.68rem !important; font-weight: 700 !important; color: #FF5252 !important; letter-spacing: 0.05em; margin-bottom: 5px !important; }

/* 오버레이 범례 */
.lnb-overlay-item {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 5px 0; border-bottom: 1px solid #1A1A1A;
    font-size: 0.73rem !important; color: #AAAAAA !important; line-height: 1.5;
}
.lnb-overlay-item:last-child { border-bottom: none; }
.lnb-overlay-dot { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; margin-top: 3px; }

/* 푸터 */
.lnb-footer {
    margin-top: 32px; padding-top: 16px;
    border-top: 1px solid #1E1E1E;
    font-size: 0.68rem !important; color: #444444 !important;
    text-align: center; line-height: 1.6;
}

#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }
.stImage  { display: flex; justify-content: center; }

/* 초기화 버튼 컬럼 수직 가운데 정렬 */
.st-emotion-cache-wfksaw {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 100% !important;
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


# ── LNB 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lnb-wrap">

      <!-- 로고 -->
      <div class="lnb-logo">
        <div class="lnb-logo-icon">✅</div>
        <div>
          <div class="lnb-logo-text">Check Mate</div>
          <div class="lnb-logo-sub">익스팬더블 배너 검수 가이드</div>
        </div>
      </div>

      <!-- 대표이미지 규격 -->
      <div class="lnb-section">
        <div class="lnb-section-header">
          <span class="lnb-section-icon">📌</span>
          <span class="lnb-section-title">대표이미지 규격</span>
        </div>
        <div class="lnb-card">
          <div class="lnb-card-title">
            대표이미지 (확장 전)
            <span class="lnb-card-badge">PNG / JPG</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">사이즈</span>
            <span class="lnb-spec-value"><strong>686 × 200 px</strong> (고정)</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">용량</span>
            <span class="lnb-spec-value"><span class="warn">150 KB 이하</span></span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">크롭 영역</span>
            <span class="lnb-spec-value">우측 <span class="purple">108 px</span> 크롭됨</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">화질 기준</span>
            <span class="lnb-spec-value">품질 점수 <span class="accent">60점 이상</span></span>
          </div>
        </div>
        <div class="lnb-warn">
          <div class="lnb-warn-label">⚠️ 크롭 주의</div>
          <p>문구·메인 상품·핵심 비주얼이 <strong style="color:#EEE;">우측 108px 크롭 영역</strong>에 걸리지 않도록 확인하세요.</p>
        </div>
      </div>

      <!-- 확장이미지 규격 -->
      <div class="lnb-section">
        <div class="lnb-section-header">
          <span class="lnb-section-icon">📐</span>
          <span class="lnb-section-title">확장이미지 규격</span>
        </div>
        <div class="lnb-card">
          <div class="lnb-card-title">
            확장이미지 (확장 후)
            <span class="lnb-card-badge">PNG / JPG</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">사이즈</span>
            <span class="lnb-spec-value"><strong>686 × 380 px</strong> (고정)</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">용량</span>
            <span class="lnb-spec-value"><span class="warn">150 KB 이하</span></span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">닫기버튼</span>
            <span class="lnb-spec-value">우상단 <span class="purple">68 px</span> 영역</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">화질 기준</span>
            <span class="lnb-spec-value">품질 점수 <span class="accent">60점 이상</span></span>
          </div>
        </div>
        <div class="lnb-warn">
          <div class="lnb-warn-label">⚠️ 닫기버튼 가독성</div>
          <p>우상단 닫기버튼 영역의 <strong style="color:#EEE;">배경 복잡도</strong>가 높으면 버튼이 잘 보이지 않습니다. 프리뷰에서 직접 확인하세요.</p>
        </div>
      </div>

      <hr class="lnb-divider">

      <!-- 동영상 규격 -->
      <div class="lnb-section">
        <div class="lnb-section-header">
          <span class="lnb-section-icon">🎬</span>
          <span class="lnb-section-title">동영상 규격</span>
        </div>
        <div class="lnb-card">
          <div class="lnb-card-title">
            동영상
            <span class="lnb-card-badge">MP4 / AVI</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">해상도</span>
            <span class="lnb-spec-value">최소 <strong>686 × 380 px</strong> 이상</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">비율</span>
            <span class="lnb-spec-value"><strong>16 : 9</strong> 유지 필수</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">영상 길이</span>
            <span class="lnb-spec-value"><span class="warn">60초 이내</span> 권장</span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">용량</span>
            <span class="lnb-spec-value"><span class="warn">30 MB 이하</span></span>
          </div>
          <div class="lnb-spec-row">
            <span class="lnb-spec-label">파일 형식</span>
            <span class="lnb-spec-value">MP4, AVI</span>
          </div>
        </div>
      </div>

      <hr class="lnb-divider">

      <!-- 공통 주의사항 -->
      <div class="lnb-section">
        <div class="lnb-section-header">
          <span class="lnb-section-icon">🚨</span>
          <span class="lnb-section-title">공통 주의사항</span>
        </div>
        <ul class="lnb-checklist">
          <li>
            <div class="lnb-check-dot"></div>
            <span><strong style="color:#FF5252;">광고 문구 삽입 금지</strong> — 광고 플래그는 시스템이 자동 부착합니다</span>
          </li>
          <li>
            <div class="lnb-check-dot"></div>
            <span>이미지 내 <strong style="color:#EEE;">"광고", "AD"</strong> 등의 텍스트를 직접 삽입하지 마세요</span>
          </li>
          <li>
            <div class="lnb-check-dot"></div>
            <span>화질 점수 <strong style="color:#FF5252;">60점 미만</strong> 시 고화질 원본으로 교체 필요</span>
          </li>
          <li>
            <div class="lnb-check-dot"></div>
            <span>교체 후에도 경고 지속 시 <strong style="color:#EEE;">UX디자인팀</strong>에 검수 요청</span>
          </li>
        </ul>
      </div>

      <!-- 푸터 -->
      <div class="lnb-footer">
        Check Mate · 익스팬더블 배너 검수<br>
        문의 : UX디자인팀
      </div>

    </div>
    """, unsafe_allow_html=True)


# ── 메인 UI ────────────────────────────────────────────────────
st.title("Check Mate : 익스팬더블 배너 검수")
st.caption("익스팬더블 광고 배너 소재 품질 및 규격 검수 프로그램")


def render_image_results(banner_type, uploaded):
    cfg = BANNER_SPECS[banner_type]
    exp_w, exp_h = cfg["size"]

    uploaded.seek(0)
    file_ext = Path(uploaded.name).suffix.lstrip(".").lower()
    if file_ext not in ALLOWED_IMAGE_EXT:
        st.warning("이미지 파일로 업로드 해주세요.")
        return

    image = Image.open(uploaded).convert("RGB")
    actual_w, actual_h = image.size
    file_size_kb = uploaded.size / 1024

    is_dim_valid = (actual_w, actual_h) == (exp_w, exp_h)
    is_size_valid = file_size_kb <= cfg["max_kb"]

    with st.spinner("이미지 품질을 분석 중입니다..."):
        is_blurry, is_pixelated, quality_score = evaluate_quality(image)

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
        is_quality_pass = quality_score >= QUALITY_THRESHOLD
        if is_quality_pass:
            st.markdown('<div class="check-pass">✅ 화질 양호</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-text">품질 점수: {quality_score:.0f}점'
                f'  |  기준: {QUALITY_THRESHOLD}점 이상</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="check-fail">⚠️ 화질 저하</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status-text">품질 점수: {quality_score:.0f}점'
                f'  |  기준: {QUALITY_THRESHOLD}점 이상</div>',
                unsafe_allow_html=True,
            )
            st.warning(
                f"화질 점수가 기준({QUALITY_THRESHOLD}점)에 미달되었습니다.\n\n"
                "고화질 원본 이미지로 교체해 주시고,\n"
                "동일한 경고가 뜬다면 UX디자인팀에 검수 요청을 해주세요."
            )

    st.divider()
    preview = apply_guide_overlay(image, banner_type)
    st.image(preview, width=actual_w)
    if banner_type == "대표이미지":
        guide_caption = (
            "대표이미지 가이드 프리뷰 — 우측 크롭 영역(보라색) 침범 여부를 직접 확인하세요"
        )
    else:
        guide_caption = (
            "확장이미지 가이드 프리뷰 — 우상단 닫기버튼 영역(보라색) 가독성을 직접 확인하세요"
        )
    st.markdown(
        f'<div class="guide-caption">{guide_caption}</div>',
        unsafe_allow_html=True,
    )


def render_video_results(uploaded):
    min_w, min_h = VIDEO_SPECS["min_size"]
    target_ratio = VIDEO_SPECS["aspect_ratio"]
    max_mb = VIDEO_SPECS["max_mb"]
    max_duration = VIDEO_SPECS["max_duration_sec"]
    allowed_formats = list(VIDEO_SPECS["formats"])

    uploaded.seek(0)
    file_ext = Path(uploaded.name).suffix.lstrip(".").lower()
    if file_ext not in allowed_formats:
        st.warning("영상 파일로 업로드 해주세요.")
        return

    file_size_mb = uploaded.size / (1024 * 1024)

    with st.spinner("동영상 정보를 분석 중입니다..."):
        meta = analyze_video(uploaded)

    if meta is None:
        st.error("동영상 파일을 읽을 수 없습니다. 파일이 손상되었거나 지원하지 않는 코덱일 수 있습니다.")
        return

    actual_w, actual_h = meta["width"], meta["height"]
    duration = meta["duration"]

    is_ratio_valid = abs(actual_w / actual_h - target_ratio) < 0.01 if actual_h > 0 else False
    is_min_size_valid = actual_w >= min_w and actual_h >= min_h
    is_dim_valid = is_ratio_valid and is_min_size_valid
    is_size_valid = file_size_mb <= max_mb
    is_format_valid = file_ext in allowed_formats
    is_duration_valid = duration <= max_duration

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        css = "check-pass" if is_dim_valid else "check-fail"
        label = "✅ 해상도 통과" if is_dim_valid else "❌ 해상도 오류"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        ratio_note = "16:9" if is_ratio_valid else f"{actual_w}:{actual_h}"
        st.markdown(
            f'<div class="status-text">업로드: {actual_w}×{actual_h}px ({ratio_note})'
            f'  |  최소: {min_w}×{min_h}px (16:9)</div>',
            unsafe_allow_html=True,
        )

    with col2:
        css = "check-pass" if is_size_valid else "check-fail"
        label = "✅ 용량 적합" if is_size_valid else "❌ 용량 초과"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">{file_size_mb:.1f} MB  |  제한: {max_mb} MB</div>',
            unsafe_allow_html=True,
        )

    with col3:
        css = "check-pass" if is_format_valid else "check-fail"
        label = "✅ 포맷 적합" if is_format_valid else "❌ 포맷 오류"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">업로드: {file_ext.upper()}  |  허용: MP4, AVI</div>',
            unsafe_allow_html=True,
        )

    with col4:
        css = "check-pass" if is_duration_valid else "check-fail"
        label = "✅ 길이 적합" if is_duration_valid else "⚠️ 길이 초과"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">{duration:.1f}초  |  권장: {max_duration}초 이내</div>',
            unsafe_allow_html=True,
        )
        if not is_duration_valid:
            st.warning(f"영상 길이가 {max_duration}초를 초과했습니다. 60초 이내로 편집해 주세요.")

    st.divider()
    uploaded.seek(0)
    st.video(uploaded)


def render_tab(banner_type):
    if banner_type == "대표이미지":
        guide_html = """
        <div class="guide-container">
          <div class="guide-row">
            <div class="guide-item">
              <div class="color-box" style="background:rgba(160,80,255,0.8);"></div>우측 크롭 영역 (108px)
            </div>
            <div class="warning-text" style="color:#B26BFF; font-size:1.05rem;">
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
              <div class="color-box" style="background:rgba(160,80,255,0.8);"></div>우상단 닫기버튼 영역
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

    uploaded = st.file_uploader(
        f"{banner_type} 시안 이미지를 업로드하세요",
        key=f"uploader_{banner_type}",
    )
    uploaded = resolve_single_upload(uploaded, f"uploader_{banner_type}")

    if not uploaded:
        return

    file_ext = Path(uploaded.name).suffix.lstrip(".").lower()
    if file_ext not in ALLOWED_IMAGE_EXT:
        st.warning("이미지 파일로 업로드 해주세요.")
        return

    render_image_results(banner_type, uploaded)


def render_video_tab():
    guide_html = """
    <div class="guide-container">
      <div class="guide-row">
        <div class="guide-item">📐 해상도: 16:9 비율  |  최소 686 × 380 px 이상</div>
        <div class="guide-item">⏱️ 영상 길이: 60초 이내 권장</div>
        <div class="guide-item">💾 용량: 30 MB 이하</div>
        <div class="guide-item">📁 포맷: MP4, AVI</div>
      </div>
    </div>
    """
    st.markdown(guide_html, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "동영상 시안을 업로드하세요",
        key="uploader_video",
    )
    uploaded = resolve_single_upload(uploaded, "uploader_video")

    if not uploaded:
        return

    render_video_results(uploaded)


# ── 일괄 업로드 ───────────────────────────────────────────────
st.subheader("일괄 검수")

# ── 사전 체크 공지 영역 ───────────────────────────────────────
st.markdown("""
<div class="precheck-box">
  <div class="precheck-header">
    <span class="precheck-header-icon">🚨</span>
    <span class="precheck-header-text">업로드 전 파일명을 확인해 주세요</span>
  </div>
  <ul class="precheck-list">
    <li class="precheck-item">
      <div class="precheck-num">1</div>
      <span class="precheck-text">
        대표 이미지 파일명에
        <span class="kw">대표이미지</span> 또는 <span class="kw">확장전</span>
        워딩이 포함되어 있어야 검수 가능합니다.
      </span>
    </li>
    <li class="precheck-item">
      <div class="precheck-num">2</div>
      <span class="precheck-text">
        확장 이미지 파일명에
        <span class="kw">확장형</span> 또는 <span class="kw">확장후</span>
        워딩이 포함되어 있어야 검수 가능합니다.
      </span>
    </li>
    <li class="precheck-item">
      <div class="precheck-num">3</div>
      <span class="precheck-text">
        동영상 파일 확장자는
        <span class="ext">.MP4</span> 또는 <span class="ext">.AVI</span>
        이어야 합니다.
      </span>
    </li>
  </ul>
</div>
""", unsafe_allow_html=True)

col_upload, col_reset = st.columns([0.95, 0.05])
with col_upload:
    _run = _uploader_run()
    batch_files = st.file_uploader(
        "소재 파일을 한 번에 선택하거나 드래그 앤 드롭하세요",
        accept_multiple_files=True,
        key=f"batch_uploader_all_{_run}",
    )
with col_reset:
    if st.button("초기화", key="batch_reset_btn"):
        # 카운터를 올려 uploader key를 교체 → 파일 목록까지 완전 초기화
        st.session_state["uploader_run"] = _uploader_run() + 1
        st.session_state.pop("batch_prev_fps", None)
        st.rerun()

if batch_files:
    batch_files = resolve_batch_upload(batch_files)
    assigned = classify_batch_files(batch_files)
    for w in assigned["warnings"]:
        st.warning(w)
    for name, w, h in assigned["unmatched"]:
        st.info(
            f"`{name}` ({w}×{h}px) 은(는) 파일명 키워드·규격으로 분류되지 않아 일괄 검수에서 제외되었습니다. "
            "아래 탭에서 개별 업로드하세요."
        )

    st.markdown("##### 일괄 검수 결과")
    with st.expander("📌 대표이미지", expanded=True):
        if assigned["대표이미지"]:
            st.caption(assigned["대표이미지"].name)
            render_image_results("대표이미지", assigned["대표이미지"])
        else:
            st.info(
                "대표이미지 파일이 없습니다. "
                "파일명(대표이미지·확장전) 또는 686×200 px 이미지를 업로드하세요."
            )

    with st.expander("📐 확장이미지", expanded=True):
        if assigned["확장이미지"]:
            st.caption(assigned["확장이미지"].name)
            render_image_results("확장이미지", assigned["확장이미지"])
        else:
            st.info(
                "확장이미지 파일이 없습니다. "
                "파일명(확장형·확장후) 또는 686×380 px 이미지를 업로드하세요."
            )

    with st.expander("🎬 동영상", expanded=True):
        if assigned["video"]:
            st.caption(assigned["video"].name)
            render_video_results(assigned["video"])
        else:
            st.info("MP4 또는 AVI 영상이 감지되지 않았습니다.")

st.divider()

# ── 개별 검수 (탭) ─────────────────────────────────────────────
st.subheader("개별 검수")

tab1, tab2, tab3 = st.tabs(["📌 대표이미지 검수", "📐 확장이미지 검수", "🎬 동영상 검수"])

with tab1:
    render_tab("대표이미지")

with tab2:
    render_tab("확장이미지")

with tab3:
    render_video_tab()