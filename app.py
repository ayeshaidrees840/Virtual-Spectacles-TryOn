import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python

st.set_page_config(
    page_title="Magic Specs VTO",
    page_icon="🕶️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    .stApp h1 {
        color: #58a6ff;
        font-weight: 700;
    }
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        border: 2px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return background

    x1, x2 = max(0, x), min(bg_w, x + ol_w)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)

    overlay_x1, overlay_x2 = max(0, -x), ol_w - max(0, (x + ol_w) - bg_w)
    overlay_y1, overlay_y2 = max(0, -y), ol_h - max(0, (y + ol_h) - bg_h)

    if x2 > x1 and y2 > y1:
        alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        overlay_rgb = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
        background_rgb = background[y1:y2, x1:x2]

        background[y1:y2, x1:x2] = (alpha * overlay_rgb + (1 - alpha) * background_rgb).astype(np.uint8)
    return background

st.title("🕶️ Magic Specs VTO - Comparison Mode")
st.write("Compare the original photo with the virtual try-on result.")

# Left Sidebar / Control Panel
with st.sidebar:
    st.markdown("### 🛠️ Control Panel")
    person_file = st.file_uploader("1. Upload Person Image", type=['jpg', 'jpeg', 'png'])
    glasses_file = st.file_uploader("2. Upload Glasses (PNG)", type=['png'])

if person_file and glasses_file:
    file_bytes_person = np.asarray(bytearray(person_file.read()), dtype=np.uint8)
    original_frame = cv2.imdecode(file_bytes_person, cv2.IMREAD_COLOR)

    file_bytes_glasses = np.asarray(bytearray(glasses_file.read()), dtype=np.uint8)
    glasses_img = cv2.imdecode(file_bytes_glasses, cv2.IMREAD_UNCHANGED)

    # Process copy for overlay so original frame stays clean
    processed_frame = original_frame.copy()

    if os.path.exists('face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        h, w, _ = processed_frame.shape

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_bridge = landmarks[168]

            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)
            nx, ny = int(nose_bridge.x * w), int(nose_bridge.y * h)

            eye_distance = np.hypot(rx - lx, ry - ly)
            glasses_width = int(eye_distance * 1.65)
            
            aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)

            resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_LANCZOS4)
            angle = np.degrees(np.arctan2(ry - ly, rx - lx))

            M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), -angle, 1.0)
            rotated_glasses = cv2.warpAffine(
                resized_glasses, 
                M, 
                (glasses_width, glasses_height), 
                flags=cv2.INTER_LANCZOS4, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(0, 0, 0, 0)
            )

            top_left_x = nx - (glasses_width // 2)
            top_left_y = ny - (glasses_height // 2)

            processed_frame = overlay_transparent(processed_frame, rotated_glasses, top_left_x, top_left_y)

    # Conversion for rendering
    orig_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Display Options (Tabs & Side-by-Side)
    tab1, tab2 = st.tabs(["⚡ Side-by-Side Comparison", "🔀 Toggle View"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📷 Original Image (Without Glasses)")
            st.image(orig_rgb, use_container_width=True)
        with col_b:
            st.markdown("#### 🕶️ Try-On Output (With Glasses)")
            st.image(proc_rgb, use_container_width=True)

    with tab2:
        view_option = st.radio("Select View:", ["With Glasses", "Without Glasses"], horizontal=True)
        if view_option == "With Glasses":
            st.image(proc_rgb, caption="Virtual Spectacles Output", use_container_width=True)
        else:
            st.image(orig_rgb, caption="Original Input Image", use_container_width=True)
