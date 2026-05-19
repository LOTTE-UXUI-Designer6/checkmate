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

VIDEO_SPECS = {
    "min_size": (686, 380),
    "aspect_ratio": 16 / 9,
    "max_mb": 30,
    "max_duration_sec": 60,
    "formats": ["mp4", "avi"],
}

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXT = set(VIDEO_SPECS["formats"])


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
    """대표(686×200), 확장(686×380), 영상(mp4/avi)으로 분류."""
    rep_size = BANNER_SPECS["대표이미지"]["size"]
    exp_size = BANNER_SPECS["확장이미지"]["size"]
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
            if (w, h) == rep_size:
                if result["대표이미지"] is not None:
                    result["warnings"].append(
                        f"대표이미지 규격 파일이 여러 개입니다. `{result['대표이미지'].name}` → `{f.name}`(으)로 교체됩니다."
                    )
                result["대표이미지"] = f
            elif (w, h) == exp_size:
                if result["확장이미지"] is not None:
                    result["warnings"].append(
                        f"확장이미지 규격 파일이 여러 개입니다. `{result['확장이미지'].name}` → `{f.name}`(으)로 교체됩니다."
                    )
                result["확장이미지"] = f
            else:
                result["unmatched"].append((f.name, w, h))
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
    """기존 파일이 있을 때 새로 올리면 새 파일만 사용(바꿔치기)."""
    if not files:
        st.session_state["batch_prev_fps"] = []
        return []

    file_list = list(files)
    current_fps = [(f.name, f.size) for f in file_list]
    prev_fps = st.session_state.get("batch_prev_fps", [])

    if prev_fps:
        new_fps = [fp for fp in current_fps if fp not in prev_fps]
        if new_fps:
            if not set(current_fps) & set(prev_fps):
                file_list = list(files)
            else:
                file_list = [f for f in file_list if (f.name, f.size) in new_fps]

    st.session_state["batch_prev_fps"] = [(f.name, f.size) for f in file_list]
    return file_list


def resolve_single_upload(uploaded, state_key):
    """단일 업로더: 새 파일 선택 시 이전 파일을 교체."""
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

    purple = (160,  80, 255,  90)   # 크롭·닫기버튼 영역 (보라톤)

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


def render_image_results(banner_type, uploaded):
    """이미지 검수 결과·프리뷰 (업로더 제외)."""
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
    preview = apply_guide_overlay(image, banner_type)
    if banner_type == "대표이미지":
        caption = "대표이미지 가이드 프리뷰 — 우측 크롭 영역(보라색) 침범 여부를 직접 확인하세요"
    else:
        caption = "확장이미지 가이드 프리뷰 — 우상단 닫기버튼 영역(보라색) 가독성을 직접 확인하세요"
    st.image(preview, caption=caption, width=actual_w)


def render_video_results(uploaded):
    """동영상 검수 결과·프리뷰 (업로더·가이드 제외)."""
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
    # 가이드 범례
    if banner_type == "대표이미지":
        guide_html = """
        <div class="guide-container">
          <div class="guide-row">
            <div class="guide-item">
              <div class="color-box" style="background:rgba(160,80,255,0.8);"></div>우측 크롭 영역 (108px)
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


# ── 일괄 업로드 (드래그 앤 드롭) ─────────────────────────────────
st.subheader("일괄 검수")
st.caption(
    "대표이미지 **686×200** px, 확장이미지 **686×380** px, 동영상 **MP4/AVI** 를 한 번에 올리면 "
    "해상도로 이미지를 구분하고 영상을 자동 검증합니다."
)
batch_files = st.file_uploader(
    "소재 파일을 한 번에 선택하거나 드래그 앤 드롭하세요",
    accept_multiple_files=True,
    key="batch_uploader_all",
)
if batch_files:
    batch_files = resolve_batch_upload(batch_files)
    assigned = classify_batch_files(batch_files)
    for w in assigned["warnings"]:
        st.warning(w)
    for name, w, h in assigned["unmatched"]:
        st.info(
            f"`{name}` ({w}×{h}px) 은(는) 대표·확장 규격과 달라 일괄 검수에 포함되지 않았습니다. "
            "아래 탭에서 개별 업로드하세요."
        )

    st.markdown("##### 일괄 검수 결과")
    with st.expander("📌 대표이미지", expanded=True):
        if assigned["대표이미지"]:
            st.caption(assigned["대표이미지"].name)
            render_image_results("대표이미지", assigned["대표이미지"])
        else:
            st.info("686×200 px 규격 이미지가 감지되지 않았습니다.")

    with st.expander("📐 확장이미지", expanded=True):
        if assigned["확장이미지"]:
            st.caption(assigned["확장이미지"].name)
            render_image_results("확장이미지", assigned["확장이미지"])
        else:
            st.info("686×380 px 규격 이미지가 감지되지 않았습니다.")

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
