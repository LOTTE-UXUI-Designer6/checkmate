import re
import unicodedata
import streamlit as st
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="Check Mate : 포커스뷰",
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


# ── 규격 정의 ──────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 686, 386

BANNER_SPECS = {
    "배너이미지": {
        "size": (CANVAS_W, CANVAS_H),
        "description": f"{CANVAS_W} × {CANVAS_H} px",
    },
}

VIDEO_SPECS = {
    "min_size": (CANVAS_W, CANVAS_H),
    "aspect_ratio": 16 / 9,
    "max_mb": 30,
    "max_duration_sec": 15,
    "formats": ["mp4", "avi"],
}

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXT = set(VIDEO_SPECS["formats"])
QUALITY_THRESHOLD = 60

# ── 배치 가이드 영역 정의 (686 × 386 기준) ──────────────────────
#
#  ┌──────────────────────────────────────────────────┐
#  │ [로고 영역]         x:24  y:24   w:160 h:60      │
#  │                                                  │
#  │         [메인 이미지 영역]                         │
#  │         x:163 y:40  w:360 h:220                  │
#  │                                                  │
#  │ [텍스트 영역]                        [버튼 위치]   │
#  │  x:24  y:250  w:300 h:80     x:480 y:295 w:180 h:55 │
#  └──────────────────────────────────────────────────┘
#
LAYOUT_ZONES = [
    {
        "name": "Main visual",
        "rect": (20, 20, 300, 366),
        "fill": (255, 200, 0, 70),
        "border": (255, 200, 0, 220),
    },
    {
        "name": "Brand Logo",
        "rect": (332, 68, 626, 122),
        "fill": (0, 200, 255, 80),
        "border": (0, 200, 255, 230),
    },
    {
        "name": "Text",
        "rect": (332, 150, 626, 219),
        "fill": (120, 80, 255, 75),
        "border": (160, 100, 255, 220),
    },
    {
        "name": "Button",
        "rect": (332, 250, 490, 298),
        "fill": (0, 230, 118, 85),
        "border": (0, 230, 118, 230),
    },
]

ZONE_LEGEND = [
    {"name": "메인 이미지 영역", "color": "#FFC800"},
    {"name": "로고 영역",        "color": "#00C8FF"},
    {"name": "텍스트 영역",      "color": "#7850FF"},
    {"name": "버튼 위치",        "color": "#00E676"},
]


def apply_layout_overlay(pil_image: Image.Image) -> Image.Image:
    """배치 가이드 컬러 딤드 오버레이를 이미지 위에 합성."""
    # 이미지를 686×386으로 리사이즈해서 좌표와 일치시킴
    display = pil_image.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
    canvas  = display.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()

    for zone in LAYOUT_ZONES:
        x1, y1, x2, y2 = zone["rect"]

        # 채우기
        draw.rectangle([(x1, y1), (x2, y2)], fill=zone["fill"])

        # 테두리 2px
        for offset in range(2):
            draw.rectangle(
                [(x1 + offset, y1 + offset), (x2 - offset, y2 - offset)],
                outline=zone["border"],
            )

        # 중앙 레이블
        label = zone["name"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        pad  = 4
        draw.rectangle(
            [(cx - tw // 2 - pad, cy - th // 2 - pad),
             (cx + tw // 2 + pad, cy + th // 2 + pad)],
            fill=(0, 0, 0, 160),
        )
        draw.text(
            (cx - tw // 2, cy - th // 2),
            label,
            font=font,
            fill=(255, 255, 255, 240),
        )

    result = Image.alpha_composite(canvas, overlay)
    return result.convert("RGB")


def _norm_filename(filename):
    return unicodedata.normalize("NFC", Path(filename).name)


def analyze_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        Path(tmp_path).unlink(missing_ok=True)
        return None

    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps if fps > 0 else 0
    cap.release()
    Path(tmp_path).unlink(missing_ok=True)

    return {"width": width, "height": height, "duration": duration, "fps": fps}


def resolve_single_upload(uploaded, state_key):
    fp_key = f"{state_key}_fp"
    if not uploaded:
        st.session_state.pop(fp_key, None)
        return None
    st.session_state[fp_key] = (uploaded.name, uploaded.size)
    return uploaded


def resolve_batch_upload(files):
    if not files:
        st.session_state["batch_prev_fps"] = []
        return []
    file_list = list(files)
    st.session_state["batch_prev_fps"] = [(f.name, f.size) for f in file_list]
    return file_list


def classify_batch_files(files):
    result = {"배너이미지": None, "video": None, "warnings": [], "unmatched": []}
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

            exp_w, exp_h = BANNER_SPECS["배너이미지"]["size"]
            name = _norm_filename(f.name)
            if (w, h) == (exp_w, exp_h):
                slot = "배너이미지"
            elif (re.search(r"686\s*[x×]\s*386", name, re.IGNORECASE)
                  or "배너" in name
                  or "banner" in name.lower()):
                slot = "배너이미지"
            else:
                slot = None

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
        for extra in video_candidates[:-1]:
            result["warnings"].append(
                f"영상 파일이 여러 개입니다. `{extra.name}` → `{video_candidates[-1].name}`(으)로 교체됩니다."
            )
        result["video"] = video_candidates[-1]

    return result


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

/* 탭 */
.stTabs [data-baseweb="tab-border"]    { background-color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #FFFFFF !important; }
button[data-baseweb="tab"] {
    color: #AAAAAA !important; font-size: 1rem !important; font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important; border-bottom: 2px solid #FFFFFF !important;
}

/* 검수 결과 */
.check-pass { font-size: 1.5rem; font-weight: 800; color: #00E676; }
.check-fail { font-size: 1.5rem; font-weight: 800; color: #FF5252; }
.status-text { font-size: 0.9rem; color: #AAAAAA; }

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
.warning-text { font-size: 0.85rem; font-weight: bold; display: flex; align-items: center; gap: 5px; }

/* 배치 가이드 범례 */
.layout-legend {
    display: flex; flex-wrap: wrap; gap: 14px;
    background: #1A1A1A; border: 1px solid #2E2E2E;
    border-radius: 10px; padding: 12px 18px; margin: 10px 0 16px 0;
}
.legend-item  { display: flex; align-items: center; gap: 7px; font-size: 0.82rem; color: #DDDDDD; }
.legend-dot   { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #111111 !important; border-right: 1px solid #333333;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #DDDDDD !important; }
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; margin-bottom: 20px !important; }
[data-testid="stSidebar"] li { color: #DDDDDD !important; margin-bottom: 8px !important; }

#MainMenu { visibility: hidden; }
header    { visibility: hidden; }
footer    { visibility: hidden; }
.stImage  { display: flex; justify-content: center; }
</style>
""", unsafe_allow_html=True)


# ── 범례 HTML ──────────────────────────────────────────────────
def legend_html():
    items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-dot" style="background:{z["color"]};"></div>'
        f'{z["name"]}'
        f'</div>'
        for z in ZONE_LEGEND
    )
    return f'<div class="layout-legend">{items}</div>'


# ── 배너 검수 렌더 ──────────────────────────────────────────────
def render_banner_results(uploaded, toggle_key: str):
    exp_w, exp_h = BANNER_SPECS["배너이미지"]["size"]

    uploaded.seek(0)
    file_ext = Path(uploaded.name).suffix.lstrip(".").lower()
    if file_ext not in ALLOWED_IMAGE_EXT:
        st.warning("이미지 파일로 업로드 해주세요.")
        return

    image = Image.open(uploaded).convert("RGB")
    actual_w, actual_h = image.size
    is_dim_valid = (actual_w, actual_h) == (exp_w, exp_h)

    with st.spinner("이미지 품질을 분석 중입니다..."):
        is_blurry, is_pixelated, quality_score = evaluate_quality(image)

    # 검수 결과
    col1, col2 = st.columns(2)
    with col1:
        css   = "check-pass" if is_dim_valid else "check-fail"
        label = "✅ 규격 통과" if is_dim_valid else "❌ 규격 오류"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">업로드: {actual_w}×{actual_h}px  |  권장: {exp_w}×{exp_h}px</div>',
            unsafe_allow_html=True,
        )
    with col2:
        q_pass  = quality_score >= QUALITY_THRESHOLD
        q_css   = "check-pass" if q_pass else "check-fail"
        q_label = "✅ 화질 양호" if q_pass else "⚠️ 화질 저하"
        st.markdown(f'<div class="{q_css}">{q_label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">품질 점수: {quality_score:.0f}점  |  기준: {QUALITY_THRESHOLD}점 이상</div>',
            unsafe_allow_html=True,
        )
        if not q_pass:
            st.warning(
                f"화질 점수가 기준({QUALITY_THRESHOLD}점)에 미달되었습니다.\n\n"
                "고화질 원본 이미지로 교체해 주시고,\n"
                "동일한 경고가 뜬다면 UX디자인팀에 검수 요청을 해주세요."
            )

    st.divider()

    # 배치 가이드 토글
    show_guide = st.toggle(
        "🗺️ 배치 가이드 보기",
        key=toggle_key,
        help="활성화하면 주요 이미지·로고·텍스트·버튼 배치 기준 영역이 컬러 딤드로 표시됩니다.",
    )

    if show_guide:
        st.markdown(legend_html(), unsafe_allow_html=True)
        preview = apply_layout_overlay(image)
        caption = "배치 가이드 오버레이 — 각 컬러 영역 안에 해당 요소가 위치하는지 확인하세요"
    else:
        preview = image
        caption = f"배너이미지 프리뷰 — {actual_w}×{actual_h}px"

    st.image(preview, caption=caption, use_column_width=False, width=min(actual_w, CANVAS_W))


# ── 동영상 검수 렌더 ────────────────────────────────────────────
def render_video_results(uploaded):
    min_w, min_h    = VIDEO_SPECS["min_size"]
    target_ratio    = VIDEO_SPECS["aspect_ratio"]
    max_mb          = VIDEO_SPECS["max_mb"]
    max_duration    = VIDEO_SPECS["max_duration_sec"]
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

    is_ratio_valid    = abs(actual_w / actual_h - target_ratio) < 0.01 if actual_h > 0 else False
    is_min_size_valid = actual_w >= min_w and actual_h >= min_h
    is_dim_valid      = is_ratio_valid and is_min_size_valid
    is_size_valid     = file_size_mb <= max_mb
    is_format_valid   = file_ext in allowed_formats
    is_duration_valid = duration <= max_duration

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        css   = "check-pass" if is_dim_valid else "check-fail"
        label = "✅ 해상도 통과" if is_dim_valid else "❌ 해상도 오류"
        ratio_note = "16:9" if is_ratio_valid else f"{actual_w}:{actual_h}"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">업로드: {actual_w}×{actual_h}px ({ratio_note})'
            f'  |  최소: {min_w}×{min_h}px (16:9)</div>',
            unsafe_allow_html=True,
        )
    with col2:
        css   = "check-pass" if is_size_valid else "check-fail"
        label = "✅ 용량 적합" if is_size_valid else "❌ 용량 초과"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">{file_size_mb:.1f} MB  |  제한: {max_mb} MB</div>',
            unsafe_allow_html=True,
        )
    with col3:
        css   = "check-pass" if is_format_valid else "check-fail"
        label = "✅ 포맷 적합" if is_format_valid else "❌ 포맷 오류"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">업로드: {file_ext.upper()}  |  허용: MP4, AVI</div>',
            unsafe_allow_html=True,
        )
    with col4:
        css   = "check-pass" if is_duration_valid else "check-fail"
        label = "✅ 길이 적합" if is_duration_valid else "⚠️ 길이 초과"
        st.markdown(f'<div class="{css}">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="status-text">{duration:.1f}초  |  권장: {max_duration}초 이내</div>',
            unsafe_allow_html=True,
        )
        if not is_duration_valid:
            st.warning(f"영상 길이가 {max_duration}초를 초과했습니다. 15초 이내로 편집해 주세요.")

    st.divider()
    uploaded.seek(0)
    st.video(uploaded)


# ── 메인 UI ────────────────────────────────────────────────────
st.title("Check Mate : 포커스뷰")
st.caption("포커스뷰 광고 배너 및 동영상 소재 품질 및 규격 사전 검증 프로그램")

# ── 일괄 검수 ─────────────────────────────────────────────────
st.subheader("일괄 검수")
st.caption(
    "배너이미지 **686×386** px 또는 파일명에 **배너·banner**, 동영상 **MP4/AVI** 를 한 번에 올리면 "
    "자동으로 분류·검증합니다."
)
batch_files = st.file_uploader(
    "소재 파일을 한 번에 선택하거나 드래그 앤 드롭하세요",
    accept_multiple_files=True,
    key="batch_uploader_all",
)
if batch_files:
    batch_files = resolve_batch_upload(batch_files)
    assigned    = classify_batch_files(batch_files)
    for w in assigned["warnings"]:
        st.warning(w)
    for name, w, h in assigned["unmatched"]:
        st.info(
            f"`{name}` ({w}×{h}px) 은(는) 파일명 키워드·규격으로 분류되지 않아 일괄 검수에서 제외되었습니다. "
            "아래 탭에서 개별 업로드하세요."
        )

    st.markdown("##### 일괄 검수 결과")
    with st.expander("🖼️ 배너이미지", expanded=True):
        if assigned["배너이미지"]:
            st.caption(assigned["배너이미지"].name)
            render_banner_results(assigned["배너이미지"], toggle_key="guide_toggle_batch")
        else:
            st.info("배너이미지 파일이 없습니다. 파일명(배너·banner) 또는 686×386 px 이미지를 업로드하세요.")

    with st.expander("🎬 동영상", expanded=True):
        if assigned["video"]:
            st.caption(assigned["video"].name)
            render_video_results(assigned["video"])
        else:
            st.info("MP4 또는 AVI 영상이 감지되지 않았습니다.")

st.divider()

# ── 개별 검수 (탭) ─────────────────────────────────────────────
st.subheader("개별 검수")
tab1, tab2 = st.tabs(["🖼️ 배너이미지 검수", "🎬 동영상 검수"])

with tab1:
    st.markdown("""
    <div class="guide-container">
      <div class="guide-row">
        <div class="guide-item">📐 규격: (가로) 686 × (세로) 386 px</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_banner = st.file_uploader("배너이미지 시안을 업로드하세요", key="uploader_banner")
    uploaded_banner = resolve_single_upload(uploaded_banner, "uploader_banner")

    if uploaded_banner:
        file_ext = Path(uploaded_banner.name).suffix.lstrip(".").lower()
        if file_ext not in ALLOWED_IMAGE_EXT:
            st.warning("이미지 파일로 업로드 해주세요.")
        else:
            render_banner_results(uploaded_banner, toggle_key="guide_toggle_tab")

with tab2:
    st.markdown("""
    <div class="guide-container">
      <div class="guide-row">
        <div class="guide-item">📐 비율: 16:9 유지  |  최소 686 × 386 px 이상</div>
        <div class="guide-item">⏱️ 영상 길이: 15초 이내</div>
        <div class="guide-item">💾 용량: 30 MB 이하</div>
        <div class="guide-item">📁 포맷: MP4, AVI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader("동영상 시안을 업로드하세요", key="uploader_video")
    uploaded_video = resolve_single_upload(uploaded_video, "uploader_video")

    if uploaded_video:
        render_video_results(uploaded_video)
