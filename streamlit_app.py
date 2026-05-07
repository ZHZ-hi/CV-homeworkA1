import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="图像处理工具", page_icon="🖼️", layout="wide")

PREVIEW_SIZE = (960, 640)
CHANNEL_PREVIEW_SIZE = (420, 280)


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def to_rgb(image):
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return image.astype(np.uint8)


def to_png_bytes(image):
    buffer = io.BytesIO()
    Image.fromarray(to_rgb(image)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_preview(image, canvas_size=PREVIEW_SIZE):
    image = to_rgb(image)
    canvas_w, canvas_h = canvas_size
    height, width = image.shape[:2]
    scale = min(canvas_w / width, canvas_h / height)
    return make_preview_at_scale(image, scale, canvas_size)


def make_preview_at_scale(image, scale, canvas_size=PREVIEW_SIZE):
    image = to_rgb(image)
    canvas_w, canvas_h = canvas_size
    height, width = image.shape[:2]
    preview_w = max(1, int(width * scale))
    preview_h = max(1, int(height * scale))

    resized = cv2.resize(image, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)
    x = (canvas_w - preview_w) // 2
    y = (canvas_h - preview_h) // 2
    canvas[y : y + preview_h, x : x + preview_w] = resized
    return canvas


def make_comparison_previews(original, result, canvas_size=PREVIEW_SIZE):
    original = to_rgb(original)
    result = to_rgb(result)
    canvas_w, canvas_h = canvas_size
    max_width = max(original.shape[1], result.shape[1])
    max_height = max(original.shape[0], result.shape[0])
    scale = min(canvas_w / max_width, canvas_h / max_height)
    return (
        make_preview_at_scale(original, scale, canvas_size),
        make_preview_at_scale(result, scale, canvas_size),
    )


def resize_image(image, width, height, method):
    interpolation = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def rotate_image(image, angle, method):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle_rad = np.radians(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_width = max(1, int(width * cos_a + height * sin_a))
    new_height = max(1, int(width * sin_a + height * cos_a))

    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_matrix[0, 2] += (new_width - width) / 2
    rotate_matrix[1, 2] += (new_height - height) / 2

    interpolation = cv2.INTER_NEAREST if method == "nearest" else cv2.INTER_LINEAR
    return cv2.warpAffine(
        image,
        rotate_matrix,
        (new_width, new_height),
        flags=interpolation,
        borderValue=(0, 0, 0),
    )


def split_color_space(image, color_space):
    if color_space == "RGB":
        r, g, b = cv2.split(image)
        return [("R 通道", r), ("G 通道", g), ("B 通道", b)]

    if color_space == "HSV":
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        return [("H 通道", h), ("S 通道", s), ("V 通道", v)]

    if color_space == "HLS":
        h, l, s = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
        return [("H 通道", h), ("L 通道", l), ("S 通道", s)]

    if color_space == "YUV":
        y, u, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
        return [("Y 通道", y), ("U 通道", u), ("V 通道", v)]

    y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb))
    return [("Y 通道", y), ("Cr 通道", cr), ("Cb 通道", cb)]


def adjust_channels(image, color_space, values):
    if color_space == "HSV":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        h = np.mod(h + values["h"], 180)
        s = np.clip(s * values["s"] / 100, 0, 255)
        v = np.clip(v * values["v"] / 100, 0, 255)
        return cv2.cvtColor(np.stack([h, s, v], axis=2).astype(np.uint8), cv2.COLOR_HSV2RGB)

    if color_space == "RGB":
        r, g, b = cv2.split(image.astype(np.float32))
        r = np.clip(r * values["r"] / 100, 0, 255)
        g = np.clip(g * values["g"] / 100, 0, 255)
        b = np.clip(b * values["b"] / 100, 0, 255)
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    if color_space == "YUV":
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32)
        y, u, v = cv2.split(yuv)
        y = np.clip(y * values["y"] / 100, 0, 255)
        u = np.clip((u - 128) * values["u"] / 100 + 128, 0, 255)
        v = np.clip((v - 128) * values["v"] / 100 + 128, 0, 255)
        return cv2.cvtColor(np.stack([y, u, v], axis=2).astype(np.uint8), cv2.COLOR_YUV2RGB)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y, cr, cb = cv2.split(ycrcb)
    y = np.clip(y * values["y"] / 100, 0, 255)
    cr = np.clip((cr - 128) * values["cr"] / 100 + 128, 0, 255)
    cb = np.clip((cb - 128) * values["cb"] / 100 + 128, 0, 255)
    return cv2.cvtColor(np.stack([y, cr, cb], axis=2).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def interpolation_controls(image):
    height, width = image.shape[:2]
    with st.sidebar.form("interpolation_form"):
        operation = st.selectbox("操作类型", ["按比例缩放", "指定尺寸", "旋转"])
        method_label = st.selectbox("插值方法", ["双线性插值", "最近邻插值"])
        method = "nearest" if method_label == "最近邻插值" else "bilinear"

        scale = 2.0
        target_width = min(800, width)
        target_height = min(600, height)
        angle = 45.0

        if operation == "按比例缩放":
            scale = st.number_input("缩放倍数", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        elif operation == "指定尺寸":
            target_width = st.number_input("目标宽度", min_value=1, max_value=5000, value=target_width)
            target_height = st.number_input("目标高度", min_value=1, max_value=5000, value=target_height)
        else:
            angle = st.number_input("旋转角度", min_value=-360.0, max_value=360.0, value=45.0, step=1.0)

        st.form_submit_button("应用变换", use_container_width=True)

    if operation == "按比例缩放":
        return resize_image(image, max(1, int(width * scale)), max(1, int(height * scale)), method)
    if operation == "指定尺寸":
        return resize_image(image, int(target_width), int(target_height), method)
    return rotate_image(image, angle, method)


def adjustment_controls(image):
    with st.sidebar.form("adjustment_form"):
        color_space = st.selectbox("调节空间", ["HSV", "RGB", "YUV", "YCrCb"])

        if color_space == "HSV":
            values = {
                "h": st.slider("H 色调偏移", -180, 180, 0),
                "s": st.slider("S 饱和度", 0, 200, 100),
                "v": st.slider("V 明度", 0, 200, 100),
            }
        elif color_space == "RGB":
            values = {
                "r": st.slider("R 红色", 0, 200, 100),
                "g": st.slider("G 绿色", 0, 200, 100),
                "b": st.slider("B 蓝色", 0, 200, 100),
            }
        elif color_space == "YUV":
            values = {
                "y": st.slider("Y 亮度", 0, 200, 100),
                "u": st.slider("U 色度", 0, 200, 100),
                "v": st.slider("V 色度", 0, 200, 100),
            }
        else:
            values = {
                "y": st.slider("Y 亮度", 0, 200, 100),
                "cr": st.slider("Cr 红色分量", 0, 200, 100),
                "cb": st.slider("Cb 蓝色分量", 0, 200, 100),
            }

        st.form_submit_button("应用调节", use_container_width=True)

    return adjust_channels(image, color_space, values)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        color: #475569;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }
    .info-row {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.1rem;
        flex-wrap: wrap;
    }
    .info-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
        background: #ffffff;
        min-width: 130px;
    }
    .info-label {
        color: #64748b;
        font-size: 0.78rem;
    }
    .info-value {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 650;
        margin-top: 0.1rem;
        word-break: break-all;
    }
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">图像处理工具</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">左侧设置参数，右侧稳定预览结果；下载按钮会保存完整处理图像。</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("控制面板")
    uploaded_file = st.file_uploader("上传图像", type=["jpg", "jpeg", "png", "bmp", "webp"])
    task = st.radio("处理任务", ["颜色空间", "插值变换", "通道调节"])
    st.divider()

if uploaded_file is None:
    st.info("请先在左侧上传一张图像。")
    st.stop()

image = load_image(uploaded_file)
height, width = image.shape[:2]

st.markdown(
    f"""
    <div class="info-row">
        <div class="info-box">
            <div class="info-label">文件名</div>
            <div class="info-value">{uploaded_file.name}</div>
        </div>
        <div class="info-box">
            <div class="info-label">图像尺寸</div>
            <div class="info-value">{width} x {height}</div>
        </div>
        <div class="info-box">
            <div class="info-label">颜色模式</div>
            <div class="info-value">RGB</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if task == "颜色空间":
    with st.sidebar:
        color_space = st.selectbox("颜色空间", ["RGB", "HSV", "HLS", "YUV", "YCrCb"])

    st.subheader("原图")
    st.image(make_preview(image), use_container_width=True)

    st.subheader(f"{color_space} 通道")
    channels = split_color_space(image, color_space)
    columns = st.columns(3)
    for column, (name, channel_image) in zip(columns, channels):
        with column:
            st.image(make_preview(channel_image, CHANNEL_PREVIEW_SIZE), caption=name, use_container_width=True)
            st.download_button(
                f"下载 {name}",
                to_png_bytes(channel_image),
                file_name=f"{color_space}_{name.replace(' ', '_')}.png",
                mime="image/png",
                key=f"download_{color_space}_{name}",
            )

elif task == "插值变换":
    result = interpolation_controls(image)
    original_preview, result_preview = make_comparison_previews(image, result)
    result_height, result_width = result.shape[:2]

    original_col, result_col = st.columns(2, gap="large")
    with original_col:
        st.subheader(f"原图 · {width} x {height}")
        st.image(original_preview, use_container_width=True)
    with result_col:
        st.subheader(f"处理结果 · {result_width} x {result_height}")
        st.image(result_preview, use_container_width=True)
        st.download_button(
            "下载处理结果",
            to_png_bytes(result),
            file_name="interpolation_result.png",
            mime="image/png",
        )

else:
    adjusted = adjustment_controls(image)
    original_preview, adjusted_preview = make_comparison_previews(image, adjusted)

    original_col, result_col = st.columns(2, gap="large")
    with original_col:
        st.subheader(f"原图 · {width} x {height}")
        st.image(original_preview, use_container_width=True)
    with result_col:
        st.subheader(f"处理结果 · {width} x {height}")
        st.image(adjusted_preview, use_container_width=True)
        st.download_button(
            "下载调节结果",
            to_png_bytes(adjusted),
            file_name="channel_adjust_result.png",
            mime="image/png",
        )
