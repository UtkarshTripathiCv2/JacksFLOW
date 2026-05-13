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
    page_title="DOG Vision System | Advanced AI Annotation",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Gradient background for the main area */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }
        
        /* Glassmorphism containers */
        div[data-testid="stVerticalBlock"] > div:has(div.metric-card) {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
        }

        /* Custom metric cards */
        .metric-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #38bdf8;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #38bdf8;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 5px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Button Enhancements */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
            transform: scale(1.02);
        }

        /* Tabs customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border-radius: 4px;
            color: #94a3b8;
            font-weight: 600;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #38bdf8;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #38bdf8;
            border-bottom: 2px solid #38bdf8;
        }

        /* Image hover zoom */
        .gallery-img {
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .gallery-img:hover {
            filter: brightness(1.1);
        }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """
    Ensures all critical variables are initialized and persistent 
    across Streamlit's reactive execution cycles.
    """
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None
    if 'edit_buffer' not in st.session_state:
        st.session_state.edit_buffer = None
    if 'class_names' not in st.session_state:
        # Default COCO classes or user-defined
        st.session_state.class_names = {}
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.25
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = 0.45
    if 'active_models' not in st.session_state:
        st.session_state.active_models = []

@st.cache_resource
def load_yolo_model(model_path):
    """
    Safely loads YOLOv8/v9/v10 models.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to initialize engine: {str(e)}")
        return None

def process_frame_with_ai(frame_bgr, models, conf, iou):
    """
    Multi-model inference engine.
    Runs detection through all active models and merges findings.
    """
    aggregated_labels = []
    
    for model_name, model in models.items():
        # Verbose=False to keep logs clean
        results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                class_id = int(classes[i])
                x_c, y_c, w, h = boxes[i]
                # Standard YOLO format: <class_id> <x_center> <y_center> <width> <height>
                label_str = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                aggregated_labels.append(label_str)
                
    # Return unique labels (merging identical boxes from different models)
    return list(set(aggregated_labels))

def render_labels_on_image(pil_img, labels_list, class_names):
    """
    Sophisticated drawing function.
    Maps numeric IDs to text names and applies a consistent color palette.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    h_img, w_img, _ = img_rgb.shape
    draw_img = img_rgb.copy()
    
    # Designer Color Palette
    palette = [
        (255, 56, 56), (0, 215, 185), (0, 188, 121), (255, 155, 71),
        (255, 118, 229), (0, 190, 255), (102, 219, 0), (255, 106, 0),
        (148, 255, 0), (255, 0, 255), (0, 255, 255), (189, 255, 0)
    ]

    for label in labels_list:
        try:
            data = label.split()
            cls_id = int(data[0])
            xc, yc, w, h = map(float, data[1:5])
            
            # Map coordinates
            x1 = int((xc - w/2) * w_img)
            y1 = int((yc - h/2) * h_img)
            x2 = int((xc + w/2) * w_img)
            y2 = int((yc + h/2) * h_img)
            
            color = palette[cls_id % len(palette)]
            # Retrieve human readable name
            name = class_names.get(cls_id, f"ID: {cls_id}")
            
            # 1. Draw outer glow / shadow
            cv2.rectangle(draw_img, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 1)
            # 2. Main Box
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 3)
            
            # 3. Dynamic Tag sizing
            text = f"{name}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Label Background
            cv2.rectangle(draw_img, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
            # Text shadow
            cv2.putText(draw_img, text, (x1 + 6, y1 - 6), font, font_scale, (0, 0, 0), thickness + 1)
            # Main Text
            cv2.putText(draw_img, text, (x1 + 5, y1 - 7), font, font_scale, (255, 255, 255), thickness)
            
        except Exception:
            continue
            
    return draw_img

def build_sidebar():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🐕 DOG VISION</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>High-Precision Intelligence Suite</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### 🧬 Intelligence Configuration")
        
        with st.expander("Active Inference Engines", expanded=True):
            MODEL_CATALOGUE = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"]
            selected_paths = st.multiselect(
                "Select Model(s)", 
                MODEL_CATALOGUE, 
                default=["yolov8n.pt"],
                help="Running multiple models increases recall but reduces performance."
            )
            
            st.session_state.conf_threshold = st.slider("Min Confidence", 0.05, 1.0, 0.25, 0.05)
            st.session_state.iou_threshold = st.slider("Overlap (IOU)", 0.1, 1.0, 0.45, 0.05)

        # Build active model dict and extract global class list
        active_instances = {}
        merged_names = {}
        for p in selected_paths:
            m = load_yolo_model(p)
            if m:
                active_instances[p] = m
                if m.names:
                    merged_names.update(m.names)
        
        st.session_state.class_names = merged_names

        st.markdown("### 📊 Live Ecosystem Metrics")
        total_items = len(st.session_state.processed_data)
        total_anno = sum([len(d['labels']) for d in st.session_state.processed_data])
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Processed Images</div>
                <div class="metric-value">{total_items}</div>
            </div>
            <div style="margin-top:15px;" class="metric-card">
                <div class="metric-label">Total Detections</div>
                <div class="metric-value">{total_anno}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        if st.button("🔥 Purge System Cache", type="secondary", use_container_width=True):
            st.session_state.processed_data = []
            st.rerun()

        return active_instances

def tab_batch_upload(active_engines):
    st.markdown("## 📥 High-Volume Ingestion")
    st.caption("Upload folders of raw imagery to trigger automated multi-model labeling.")
    
    files = st.file_uploader(
        "Ingest Image Assets", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True,
        help="Supports individual images or batch selection."
    )
    
    if files:
        if st.button("⚡ EXECUTE BATCH AUTO-LABEL", use_container_width=True):
            progress_bar = st.progress(0)
            status = st.empty()
            
            for i, f in enumerate(files):
                status.info(f"Scanning Asset: {f.name}...")
                
                # Image transformation
                pil_raw = Image.open(f).convert("RGB")
                cv2_frame = cv2.cvtColor(np.array(pil_raw), cv2.COLOR_RGB2BGR)
                
                # Intelligent Inference
                predictions = process_frame_with_ai(
                    cv2_frame, 
                    active_engines, 
                    st.session_state.conf_threshold,
                    st.session_state.iou_threshold
                )
                
                # Repository storage
                st.session_state.processed_data.append({
                    'original_image': pil_raw,
                    'labels': predictions,
                    'filename': f"{datetime.now().strftime('%m%d%H%M')}_{f.name}"
                })
                
                progress_bar.progress((i + 1) / len(files))
            
            status.success(f"System Check: {len(files)} assets ingested successfully.")
            st.balloons()

def tab_live_intelligence(active_engines):
    st.markdown("## 🎥 Live Visual Intelligence")
    st.info("Direct sensor feed integration. Use triggers below to capture high-fidelity samples.")
    
    col_v, col_c = st.columns([3, 1])
    
    with col_c:
        is_active = st.toggle("Activate Sensors", value=False)
        st.divider()
        snap_trigger = st.button("📸 SNAPSHOT FRAME", use_container_width=True, type="primary")
        st.markdown("""
        **Operational Note:**
        Preview uses optimized 1-pass inference. Captured frames undergo full multi-engine fusion.
        """)

    if is_active:
        v_cap = cv2.VideoCapture(0)
        v_placeholder = col_v.empty()
        
        if not active_engines:
            st.error("No intelligence engines active. Enable via sidebar.")
            is_active = False

        while is_active:
            success, raw_frame = v_cap.read()
            if not success: break
            
            # Preview Logic (Fastest model)
            fast_engine = list(active_engines.values())[0]
            preview_res = fast_engine.predict(raw_frame, conf=st.session_state.conf_threshold, verbose=False)
            preview_render = preview_res[0].plot()
            
            v_placeholder.image(cv2.cvtColor(preview_render, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if snap_trigger:
                rgb_pil = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
                full_inference = process_frame_with_ai(raw_frame, active_engines, st.session_state.conf_threshold, 0.45)
                
                st.session_state.processed_data.append({
                    'original_image': rgb_pil,
                    'labels': full_inference,
                    'filename': f"sensor_snap_{int(time.time())}.jpg"
                })
                st.toast("Intelligence Sample Captured!")
                break
                
            time.sleep(0.01)
        
        v_cap.release()

def tab_data_studio():
    if not st.session_state.processed_data:
        st.markdown("<div style='text-align:center; padding:50px;'><h3>Sensor Buffer Empty</h3><p>Ingest data via Batch or Live modules to begin processing.</p></div>", unsafe_allow_html=True)
        return

    # Logic Switching: Detailed Annotation vs Gallery
    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        work_item = st.session_state.edit_buffer
        
        st.markdown(f"## 🛠️ Annotation Studio: `{work_item['filename']}`")
        
        s_col1, s_col2 = st.columns([2, 1])
        
        with s_col2:
            st.markdown("#### Layer Control")
            
            if st.button("➕ Inject Manual Label", use_container_width=True):
                work_item['labels'].append("0 0.5 0.5 0.2 0.2")
                st.rerun()

            purge_list = []
            for i, label_str in enumerate(work_item['labels']):
                with st.expander(f"📦 OBJECT {i+1}", expanded=True):
                    l_data = label_str.split()
                    c_id = int(l_data[0])
                    xc, yc, w, h = map(float, l_data[1:5])
                    
                    # Mapping logic for presentation
                    current_name = st.session_state.class_names.get(c_id, f"Class {c_id}")
                    st.info(f"Identity: **{current_name}**")
                    
                    new_id = st.number_input("ID Mapping", 0, 999, c_id, key=f"id_{i}")
                    new_xc = st.slider("X-Axis Center", 0.0, 1.0, xc, 0.001, key=f"xc_{i}")
                    new_yc = st.slider("Y-Axis Center", 0.0, 1.0, yc, 0.001, key=f"yc_{i}")
                    new_w = st.slider("Object Width", 0.0, 1.0, w, 0.001, key=f"w_{i}")
                    new_h = st.slider("Object Height", 0.0, 1.0, h, 0.001, key=f"h_{i}")
                    
                    # Live string update
                    work_item['labels'][i] = f"{new_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
                    
                    if st.button(f"🗑️ Delete Object {i+1}", key=f"del_box_{i}"):
                        purge_list.append(i)
            
            for i in sorted(purge_list, reverse=True):
                work_item['labels'].pop(i)
                st.rerun()

            st.divider()
            b1, b2 = st.columns(2)
            if b1.button("💾 COMMIT EDITS", type="primary", use_container_width=True):
                st.session_state.processed_data[idx] = work_item
                st.session_state.edit_index = None
                st.toast("Database updated.")
                st.rerun()
            if b2.button("🚫 DISCARD", use_container_width=True):
                st.session_state.edit_index = None
                st.rerun()

        with s_col1:
            # Immersive Visual Feedback
            render_preview = render_labels_on_image(
                work_item['original_image'], 
                work_item['labels'], 
                st.session_state.class_names
            )
            st.image(render_preview, use_container_width=True, caption="Visual Sandbox Output")

    else:
        # High-End Gallery View
        st.markdown("## 🔍 Knowledge Repository")
        
        search_q = st.text_input("Filter assets by filename...", "")
        
        filtered_data = [d for d in st.session_state.processed_data if search_q.lower() in d['filename'].lower()]
        
        g_cols = st.columns(4)
        for i, item in enumerate(filtered_data):
            # Find original index in session_state
            orig_idx = next(idx for idx, d in enumerate(st.session_state.processed_data) if d['filename'] == item['filename'])
            
            with g_cols[i % 4]:
                t_preview = render_labels_on_image(item['original_image'], item['labels'], st.session_state.class_names)
                st.image(t_preview, use_container_width=True)
                st.markdown(f"**{item['filename'][:20]}...**")
                st.caption(f"{len(item['labels'])} objects identified")
                
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("Edit", key=f"gs_ed_{i}", use_container_width=True):
                    st.session_state.edit_index = orig_idx
                    st.session_state.edit_buffer = {
                        'original_image': item['original_image'],
                        'labels': item['labels'].copy(),
                        'filename': item['filename']
                    }
                    st.rerun()
                if c_btn2.button("Del", key=f"gs_de_{i}", use_container_width=True):
                    st.session_state.processed_data.pop(orig_idx)
                    st.rerun()

def tab_export_wizard():
    if not st.session_state.processed_data:
        st.warning("Empty Dataset: Cannot generate deployment package.")
        return

    st.markdown("## 🚀 Neural Deployment Wizard")
    
    # 1. Real-time Distribution Chart
    st.subheader("Asset Distribution Analysis")
    all_class_ids = []
    for item in st.session_state.processed_data:
        for l in item['labels']:
            all_class_ids.append(int(l.split()[0]))
    
    if all_class_ids:
        series = pd.Series(all_class_ids).value_counts().reset_index()
        series.columns = ['ID', 'Instance Count']
        series['Class Name'] = series['ID'].map(lambda x: st.session_state.class_names.get(x, f"Unknown_{x}"))
        st.bar_chart(series, x='Class Name', y='Instance Count', color="#38bdf8")
    
    st.divider()
    
    # 2. Packaging logic
    st.subheader("Dataset Packaging")
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        p_name = st.text_input("Project Codename", "DOG_VIS_01")
        split_val = st.select_slider("Validation Allocation (%)", options=[10, 20, 30, 40, 50], value=20)
    
    with p_col2:
        st.info("Generating standard YOLOv8 folder structure with normalized bounding box txt files.")
        
        if st.button("🏗️ COMPILE ZIP PACKAGE", type="primary", use_container_width=True):
            with st.spinner("Compiling Neural Package..."):
                with tempfile.TemporaryDirectory() as base_tmp:
                    # Directory Architecture
                    sys_root = os.path.join(base_tmp, p_name)
                    for folder in ['train', 'val']:
                        os.makedirs(os.path.join(sys_root, folder, 'images'), exist_ok=True)
                        os.makedirs(os.path.join(sys_root, folder, 'labels'), exist_ok=True)
                    
                    # Split implementation
                    dataset = st.session_state.processed_data.copy()
                    random.shuffle(dataset)
                    v_count = int(len(dataset) * (split_val / 100))
                    
                    split_map = {
                        'val': dataset[:v_count],
                        'train': dataset[v_count:]
                    }
                    
                    for split_tag, items in split_map.items():
                        for sample in items:
                            safe_name = sample['filename'].replace(" ", "_")
                            # 1. Save Image
                            img_p = os.path.join(sys_root, split_tag, 'images', safe_name)
                            sample['original_image'].save(img_p)
                            # 2. Save Annotations
                            txt_name = os.path.splitext(safe_name)[0] + ".txt"
                            txt_p = os.path.join(sys_root, split_tag, 'labels', txt_name)
                            with open(txt_p, 'w') as tf:
                                tf.write("\n".join(sample['labels']))
                    
                    # 3. YAML Config Generation
                    # Map names up to the maximum found index
                    max_id = max(all_class_ids) if all_class_ids else 0
                    c_names_list = [st.session_state.class_names.get(k, f"class_{k}") for k in range(max_id + 1)]
                    
                    config = {
                        'path': f'./{p_name}',
                        'train': 'train/images',
                        'val': 'val/images',
                        'nc': len(c_names_list),
                        'names': c_names_list
                    }
                    
                    with open(os.path.join(sys_root, 'data.yaml'), 'w') as yf:
                        yaml.dump(config, yf, default_flow_style=False)
                    
                    # 4. ZIP Archiving
                    target_zip = shutil.make_archive(os.path.join(base_tmp, "output"), 'zip', sys_root)
                    
                    with open(target_zip, "rb") as final_f:
                        st.download_button(
                            label="⬇️ DOWNLOAD DEPLOYMENT PACKAGE",
                            data=final_f,
                            file_name=f"{p_name}_YOLO_DATASET.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

def main():
    apply_custom_styles()
    init_session_state()
    
    # Hero Section
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); padding: 30px; border-radius: 20px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 30px;">
            <h1 style='margin:0; color:#38bdf8; font-size: 2.5rem;'>DOG Vision System</h1>
            <p style='color:#94a3b8; font-size: 1.1rem;'>Professional Grade Object Detection & Annotation Workflow</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Global Config via Sidebar
    active_engines = build_sidebar()
    
    # Navigation System
    t1, t2, t3, t4 = st.tabs([
        "💎 BATCH INGESTION", 
        "📡 LIVE SENSORS", 
        "🎨 DATA STUDIO", 
        "⚡ EXPORT WIZARD"
    ])
    
    with t1:
        tab_batch_upload(active_engines)
        
    with t2:
        tab_live_intelligence(active_engines)
        
    with t3:
        tab_data_studio()
        
    with t4:
        tab_export_wizard()

    # Footer
    st.markdown("---")
    st.caption("DOG Vision System | Institutional Release v4.2-LTS | Developed for High-Performance CV")

if __name__ == "__main__":
    main()
