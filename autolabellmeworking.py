import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageColor
import os
import tempfile
import shutil
import zipfile
import pandas as pd
import time
import json
import yaml
import random
from datetime import datetime
import io

st.set_page_config(
    page_title="AI-Labeler | YOLO Annotation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Main background and container styling */
        .main {
            background-color: #0e1117;
        }
        
        /* Custom card styling for metrics */
        .metric-card {
            background-color: #1e2130;
            border: 1px solid #3e445e;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2e7bcf;
        }
        
        .metric-label {
            font-size: 14px;
            color: #8a8d97;
            text-transform: uppercase;
        }

        /* Button styling refinements */
        .stButton>button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            border-color: #2e7bcf;
            color: #2e7bcf;
            box-shadow: 0 0 10px rgba(46, 123, 207, 0.4);
        }

        /* Sidebar enhancement */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #1e2130;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
        }

        /* Gallery Image Hover Effect */
        .gallery-item img {
            transition: transform 0.3s ease;
        }
        .gallery-item img:hover {
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """
    Initializes all necessary session state variables to ensure 
    the app remains stable through reruns and complex workflows.
    """
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None
    if 'edit_buffer' not in st.session_state:
        st.session_state.edit_buffer = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = {}
    if 'model_cache' not in st.session_state:
        st.session_state.model_cache = {}
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.25
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = 0.45
    if 'batch_processing' not in st.session_state:
        st.session_state.batch_processing = False

@st.cache_resource
def load_yolo_model(model_path):
    """
    Loads and caches YOLO models safely.
    Handles weights downloading automatically if standard names are used.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {str(e)}")
        return None

def get_unique_colors(n):
    """Generates distinct RGB colors for bounding boxes."""
    colors = []
    for i in range(n):
        h = int(360 * i / n)
        colors.append(f"hsl({h}, 70%, 50%)")
    return colors

def process_frame_with_ai(frame_bgr, models, conf, iou):
    """
    Inference logic: Aggregates results from multiple models and
    returns a unified list of detections in YOLO format.
    """
    aggregated_labels = []
    
    for model_name, model in models.items():
        results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                class_id = int(classes[i])
                x_c, y_c, w, h = boxes[i]
                # Format: <class_id> <x_center> <y_center> <width> <height>
                label_str = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                aggregated_labels.append(label_str)
                
    # Return unique labels only
    return list(set(aggregated_labels))

def render_labels_on_image(pil_img, labels_list, class_names):
    """
    Draws custom styled bounding boxes and labels on a PIL image.
    Used for both static review and live editing feedback.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    h_img, w_img, _ = img_rgb.shape
    draw_img = img_rgb.copy()
    
    # Palette definition (20 distinct colors)
    palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 166), (255, 155, 71), 
        (255, 118, 229), (255, 144, 30), (255, 106, 0), (255, 81, 151),
        (0, 190, 255), (0, 215, 185), (0, 209, 143), (0, 188, 121),
        (102, 219, 0), (148, 255, 0), (189, 255, 0), (241, 255, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0)
    ]

    for label in labels_list:
        try:
            data = label.split()
            cls_id = int(data[0])
            xc, yc, w, h = map(float, data[1:5])
            
            # Convert normalized to pixel coords
            x1 = int((xc - w/2) * w_img)
            y1 = int((yc - h/2) * h_img)
            x2 = int((xc + w/2) * w_img)
            y2 = int((yc + h/2) * h_img)
            
            color = palette[cls_id % len(palette)]
            name = class_names.get(cls_id, f"Class {cls_id}")
            
            # Draw primary box
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label tag
            text = f"{name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(draw_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(draw_img, text, (x1 + 5, y1 - 7), font, font_scale, (255, 255, 255), thickness)
            
        except Exception:
            continue
            
    return draw_img

def build_sidebar():
    with st.sidebar:
        st.image("https://placehold.co/400x120/161b22/2e7bcf?text=AI-LABELER+PRO&font=playfair-display", use_container_width=True)
        st.markdown("### 🛠️ Global Settings")
        
        with st.expander("Model Configuration", expanded=True):
            MODEL_OPTIONS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"]
            selected_models = st.multiselect(
                "Inference Engine(s)", 
                MODEL_OPTIONS, 
                default=["yolov8n.pt"],
                help="Select one or more YOLO models to detect objects. Predictions will be merged."
            )
            
            st.session_state.conf_threshold = st.slider("Confidence", 0.05, 1.0, 0.25, 0.05)
            st.session_state.iou_threshold = st.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)

        # Load models and aggregate class names
        active_models = {}
        all_names = {}
        for path in selected_models:
            m = load_yolo_model(path)
            if m:
                active_models[path] = m
                if m.names:
                    all_names.update(m.names)
        
        st.session_state.class_names = all_names

        # System Statistics Card
        st.markdown("### 📊 Dataset Overview")
        total_imgs = len(st.session_state.processed_data)
        total_boxes = sum([len(d['labels']) for d in st.session_state.processed_data])
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Images</div>
                <div class="metric-value">{total_imgs}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Annotations</div>
                <div class="metric-value">{total_boxes}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Reset Workspace", type="secondary", use_container_width=True):
            st.session_state.processed_data = []
            st.rerun()

        return active_models

def tab_image_upload(active_models):
    st.markdown("### 🖼️ Batch Image Auto-Labeling")
    st.write("Upload high-resolution images to automatically generate YOLO annotations.")
    
    files = st.file_uploader("Select JPG/PNG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if files:
        if st.button("🚀 Process Batch", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, f in enumerate(files):
                status_text.info(f"Analyzing {f.name}...")
                
                # Convert file to image
                img_pil = Image.open(f).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
                # Inference
                labels = process_frame_with_ai(
                    img_bgr, 
                    active_models, 
                    st.session_state.conf_threshold,
                    st.session_state.iou_threshold
                )
                
                # Store
                st.session_state.processed_data.append({
                    'original_image': img_pil,
                    'labels': labels,
                    'filename': f"{datetime.now().strftime('%H%M%S')}_{f.name}"
                })
                
                progress_bar.progress((i + 1) / len(files))
            
            status_text.success(f"Successfully processed {len(files)} images!")
            st.balloons()

def tab_live_capture(active_models):
    st.markdown("### 🎥 Interactive AI-Webcam")
    st.info("The live feed uses the first selected model for performance. Use 'Capture' to save high-res annotations.")
    
    col_cam, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        run_cam = st.toggle("Enable Camera Feed", value=False)
        st.divider()
        capture_btn = st.button("📸 CAPTURE FRAME", use_container_width=True)
        st.caption("Captures currently visible frame with full multi-model detection.")

    if run_cam:
        # Standard Streamlit CV2 loop
        cap = cv2.VideoCapture(0)
        frame_holder = col_cam.empty()
        
        # Performance check
        if not active_models:
            st.warning("No models loaded for detection.")
            run_cam = False

        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            
            # High-speed preview (uses 1st model)
            p_model = list(active_models.values())[0]
            results = p_model.predict(frame, conf=st.session_state.conf_threshold, verbose=False)
            preview_frame = results[0].plot()
            
            # Display
            frame_holder.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
            
            # Handle capture trigger
            if capture_btn:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                labels = process_frame_with_ai(frame, active_models, st.session_state.conf_threshold, 0.45)
                
                st.session_state.processed_data.append({
                    'original_image': img_pil,
                    'labels': labels,
                    'filename': f"capture_{int(time.time())}.jpg"
                })
                st.toast("Frame Saved to Collection!")
                # Reset capture button logic (Streamlit specific)
                break
                
            time.sleep(0.01)
        
        cap.release()

def tab_review_and_edit():
    if not st.session_state.processed_data:
        st.info("Your collection is currently empty. Start by uploading images or using the camera.")
        return

    # Logical Branch: Detail Editor or Gallery
    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        buffer = st.session_state.edit_buffer
        
        st.markdown(f"### ✏️ Editing: `{buffer['filename']}`")
        
        e_col1, e_col2 = st.columns([2, 1])
        
        with e_col2:
            st.markdown("#### Bounding Boxes")
            
            if st.button("➕ Add Manual Annotation", use_container_width=True):
                buffer['labels'].append("0 0.5 0.5 0.2 0.2")
                st.rerun()

            to_delete = []
            for i, label_str in enumerate(buffer['labels']):
                with st.expander(f"📦 Box {i+1}", expanded=True):
                    parts = label_str.split()
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # Manual Tuning Sliders
                    n_cid = st.number_input("Class ID", 0, 100, cid, key=f"ecid_{i}")
                    n_xc = st.slider("X-Center", 0.0, 1.0, xc, 0.001, key=f"exc_{i}")
                    n_yc = st.slider("Y-Center", 0.0, 1.0, yc, 0.001, key=f"eyc_{i}")
                    n_w = st.slider("Width", 0.0, 1.0, w, 0.001, key=f"ew_{i}")
                    n_h = st.slider("Height", 0.0, 1.0, h, 0.001, key=f"eh_{i}")
                    
                    # Update buffer immediately
                    buffer['labels'][i] = f"{n_cid} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}"
                    
                    if st.button(f"🗑️ Delete Box {i+1}", key=f"edel_{i}"):
                        to_delete.append(i)
            
            for i in sorted(to_delete, reverse=True):
                buffer['labels'].pop(i)
                st.rerun()

            st.divider()
            s1, s2 = st.columns(2)
            if s1.button("💾 SAVE CHANGES", type="primary", use_container_width=True):
                st.session_state.processed_data[idx] = buffer
                st.session_state.edit_index = None
                st.toast("Success: Labels updated.")
                st.rerun()
            if s2.button("❌ DISCARD", use_container_width=True):
                st.session_state.edit_index = None
                st.rerun()

        with e_col1:
            # Visual Feedback
            preview = render_labels_on_image(
                buffer['original_image'], 
                buffer['labels'], 
                st.session_state.class_names
            )
            st.image(preview, use_container_width=True)

    else:
        # Gallery View
        st.markdown("### 🔍 Annotation Gallery")
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.processed_data):
            with cols[i % 4]:
                thumb = render_labels_on_image(item['original_image'], item['labels'], st.session_state.class_names)
                st.image(thumb, caption=f"ID: {i} | {len(item['labels'])} labels", use_container_width=True)
                
                c1, c2 = st.columns(2)
                if c1.button("Edit", key=f"g_ed_{i}", use_container_width=True):
                    st.session_state.edit_index = i
                    st.session_state.edit_buffer = {
                        'original_image': item['original_image'],
                        'labels': item['labels'].copy(),
                        'filename': item['filename']
                    }
                    st.rerun()
                if c2.button("Del", key=f"g_de_{i}", use_container_width=True):
                    st.session_state.processed_data.pop(i)
                    st.rerun()

def tab_export_and_download():
    if not st.session_state.processed_data:
        st.info("Capture or upload data to enable export features.")
        return

    st.markdown("### 📊 Dataset Analytics")
    
    # Calculate Class Distribution
    dist = []
    for item in st.session_state.processed_data:
        for l in item['labels']:
            dist.append(int(l.split()[0]))
    
    if dist:
        df_dist = pd.Series(dist).value_counts().reset_index()
        df_dist.columns = ['ID', 'Count']
        df_dist['Name'] = df_dist['ID'].map(lambda x: st.session_state.class_names.get(x, f"ID {x}"))
        st.bar_chart(df_dist, x='Name', y='Count')
    
    st.divider()
    st.markdown("### 📦 YOLO Export Wizard")
    
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        ds_name = st.text_input("Project Name", "my_yolo_project")
        split = st.slider("Train/Val Split Ratio", 0.5, 0.95, 0.8)
    
    with col_set2:
        st.info("This will generate a ZIP containing images, .txt label files, and a data.yaml configuration.")
        
        if st.button("🏗️ PREPARE EXPORT ZIP", use_container_width=True, type="primary"):
            with st.spinner("Building dataset structure..."):
                with tempfile.TemporaryDirectory() as tmp:
                    # Folder Structure
                    root = os.path.join(tmp, ds_name)
                    for s in ['train', 'val']:
                        os.makedirs(os.path.join(root, s, 'images'), exist_ok=True)
                        os.makedirs(os.path.join(root, s, 'labels'), exist_ok=True)
                    
                    # Split logic
                    data = st.session_state.processed_data.copy()
                    random.shuffle(data)
                    split_i = int(len(data) * split)
                    
                    train_set = data[:split_i]
                    val_set = data[split_i:]
                    
                    for set_name, items in [('train', train_set), ('val', val_set)]:
                        for item in items:
                            fname_base = item['filename'].replace(" ", "_")
                            # Save Image
                            img_path = os.path.join(root, set_name, 'images', fname_base)
                            item['original_image'].save(img_path)
                            # Save Txt
                            txt_name = os.path.splitext(fname_base)[0] + ".txt"
                            txt_path = os.path.join(root, set_name, 'labels', txt_name)
                            with open(txt_path, 'w') as f:
                                f.write("\n".join(item['labels']))
                    
                    # Generate YAML
                    names_list = [st.session_state.class_names.get(i, f"class_{i}") for i in range(max(dist)+1)] if dist else []
                    yaml_data = {
                        'train': './train/images',
                        'val': './val/images',
                        'nc': len(names_list),
                        'names': names_list
                    }
                    with open(os.path.join(root, 'data.yaml'), 'w') as yf:
                        yaml.dump(yaml_data, yf)
                    
                    # Zip
                    zip_file = shutil.make_archive(os.path.join(tmp, "dataset_export"), 'zip', root)
                    
                    with open(zip_file, "rb") as f:
                        st.download_button(
                            label="⬇️ DOWNLOAD ZIP ARCHIVE",
                            data=f,
                            file_name=f"{ds_name}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

def main():
    apply_custom_styles()
    init_session_state()
    
    st.title("🤖 AI-Labeler Pro")
    st.caption("A high-performance assistant for creating YOLO datasets using multi-model fusion.")
    
    # Build sidebar and get models
    active_models = build_sidebar()
    
    # Navigation
    tabs = st.tabs([
        "📥 Upload & Auto-Label", 
        "🎥 Live Detection", 
        "🔍 Review & Refine", 
        "🚀 Export Dataset"
    ])
    
    with tabs[0]:
        tab_image_upload(active_models)
        
    with tabs[1]:
        tab_live_capture(active_models)
        
    with tabs[2]:
        tab_review_and_edit()
        
    with tabs[3]:
        tab_export_and_download()

    st.markdown("---")
    st.caption("Advanced Computer Vision Workflow | v3.0 stable")

if __name__ == "__main__":
    main()
