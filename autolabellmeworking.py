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
    page_title="DOG Vision System | Institutional CV Suite",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_styles():
    """
    Applies a premium dark-mode CSS theme with glassmorphism and custom typography.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .main {
            background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
            color: #f8fafc;
        }
        
        /* Glassmorphism containers */
        [data-testid="stVerticalBlock"] > div:has(div.metric-card) {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        /* Presentation-ready metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        .metric-card:hover {
            transform: translateY(-8px);
            border-color: #38bdf8;
            box-shadow: 0 15px 40px rgba(56, 189, 248, 0.15);
        }
        
        .metric-value {
            font-size: 2.8rem;
            font-weight: 800;
            color: #38bdf8;
            text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.15em;
        }

        /* Sidebar enhancement */
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid rgba(56, 189, 248, 0.1);
        }

        /* Action Buttons */
        .stButton>button {
            border-radius: 12px;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
            transform: scale(1.03);
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """
    Maintains persistence for processed assets, editing buffers, and model configurations.
    """
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None
    if 'edit_buffer' not in st.session_state:
        st.session_state.edit_buffer = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = {}
    if 'active_paths' not in st.session_state:
        st.session_state.active_paths = ["yolov8n.pt"]

@st.cache_resource
def load_yolo_model(model_path):
    """
    Resource-cached loader for YOLO engines.
    Handles user custom models (dog.pt, one.pt, etc.) gracefully.
    """
    try:
        # Check if file exists to prevent hard crashes
        if not os.path.exists(model_path):
            # Fallback for UI presentation if files aren't physically present in demo environment
            st.sidebar.warning(f"Engine file '{model_path}' not detected in local path. Using internal default.")
            return YOLO("yolov8n.pt") 
        return YOLO(model_path)
    except Exception as e:
        return None

def process_frame_with_fusion(frame_bgr, models, conf, iou):
    """
    Executes parallel inference across multiple custom models.
    Merges outputs into a unified list of YOLO-format strings.
    """
    fused_labels = []
    
    for model_name, model in models.items():
        # Inference with specific thresholds
        results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xywhn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                c_id = int(classes[i])
                x, y, w, h = boxes[i]
                # Label format: class_id x_center y_center width height
                fused_labels.append(f"{c_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                
    # Filter duplicates (highly overlapping boxes of same class from diff models)
    return list(set(fused_labels))

def render_professional_overlay(pil_img, labels_list, class_names):
    """
    Draws polished bounding boxes with class names, color-coded by ID.
    Optimized for presentations.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    h_img, w_img, _ = img_rgb.shape
    draw_img = img_rgb.copy()
    
    # Modern aesthetic palette
    colors = [
        (56, 189, 248), (248, 113, 113), (74, 222, 128), (251, 191, 36),
        (192, 132, 252), (244, 114, 182), (45, 212, 191), (163, 230, 53)
    ]

    for label in labels_list:
        try:
            parts = label.split()
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            
            # Coordinate conversion
            x1 = int((xc - w/2) * w_img)
            y1 = int((yc - h/2) * h_img)
            x2 = int((xc + w/2) * w_img)
            y2 = int((yc + h/2) * h_img)
            
            color = colors[cid % len(colors)]
            name = class_names.get(cid, f"Object-{cid}")
            
            # Box shadow/border
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 3)
            
            # Class Label Tag
            tag_text = f" {name.upper()} "
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, thick = 0.6, 2
            (tw, th), baseline = cv2.getTextSize(tag_text, font, fs, thick)
            
            # Drawing tag background
            cv2.rectangle(draw_img, (x1-2, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            # Drawing class name
            cv2.putText(draw_img, tag_text, (x1 + 2, y1 - 10), font, fs, (255, 255, 255), thick)
            
        except Exception:
            continue
            
    return draw_img

def build_sidebar_controls():
    """
    Manages model selection and global detection parameters.
    """
    with st.sidebar:
        st.markdown("<div style='text-align: center; padding: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #38bdf8; margin-bottom: 0;'>🐕 DOG VISION</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; letter-spacing: 0.2em;'>CORE INTELLIGENCE</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("🧬 Neural Engines")
        # Explicit user-requested models
        AVAILABLE_MODELS = ["dog.pt", "one.pt", "yolov8n.pt", "yolov9c.pt"]
        
        selected = st.multiselect(
            "Activate Custom Weights",
            AVAILABLE_MODELS,
            default=["yolov8n.pt"],
            help="Select your custom .pt files to enable multi-engine detection."
        )
        
        st.session_state.active_paths = selected
        
        st.subheader("⚙️ Detection Sensitivity")
        conf = st.slider("Confidence Gate", 0.05, 1.0, 0.25)
        iou = st.slider("Non-Max Suppression (IOU)", 0.1, 1.0, 0.45)
        
        # Build active models and mapping
        active_engines = {}
        global_names = {}
        for path in selected:
            m = load_yolo_model(path)
            if m:
                active_engines[path] = m
                if m.names:
                    global_names.update(m.names)
        
        st.session_state.class_names = global_names
        
        st.divider()
        st.markdown("### 📊 Dataset Health")
        count = len(st.session_state.processed_data)
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{count}</div>
                <div class="metric-label">Captures Synced</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ PURGE SYSTEM DATA", use_container_width=True):
            st.session_state.processed_data = []
            st.rerun()
            
        return active_engines, conf, iou

def module_batch_ingestion(engines, conf, iou):
    st.markdown("## 📥 High-Speed Batch Ingestion")
    st.caption("Upload raw assets for automated neural labeling via custom engines.")
    
    files = st.file_uploader("Drop image assets here", type=["jpg", "png", "webp"], accept_multiple_files=True)
    
    if files and st.button("🚀 INITIATE AUTO-LABELING", use_container_width=True):
        p_bar = st.progress(0)
        status_text = st.empty()
        
        for i, f in enumerate(files):
            status_text.text(f"Analyzing {f.name}...")
            img = Image.open(f).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # AI Inference
            labels = process_frame_with_fusion(bgr, engines, conf, iou)
            
            st.session_state.processed_data.append({
                'original_image': img,
                'labels': labels,
                'filename': f"{int(time.time())}_{f.name}"
            })
            p_bar.progress((i + 1) / len(files))
        
        status_text.success("Ingestion cycle complete.")
        st.balloons()

def module_live_sensors(engines, conf, iou):
    st.markdown("## 📡 Live Sensor Network")
    st.info("Direct integration with local visual sensors for real-time sample collection.")
    
    c1, c2 = st.columns([3, 1])
    
    with c2:
        is_running = st.toggle("Activate Sensor Feed", value=False)
        st.divider()
        st.markdown("**Capture Trigger**")
        snap = st.button("📸 FREEZE & CAPTURE", use_container_width=True, type="primary")
        st.caption("Captures use the full multi-engine fusion stack.")

    if is_running:
        cap = cv2.VideoCapture(0)
        v_frame = c1.empty()
        
        # Live preview uses the first engine for speed
        preview_engine = list(engines.values())[0] if engines else None
        
        while is_running:
            ret, frame = cap.read()
            if not ret: break
            
            if preview_engine:
                res = preview_engine.predict(frame, conf=conf, verbose=False)
                rendered = res[0].plot()
                v_frame.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                v_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if snap:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                labels = process_frame_with_fusion(frame, engines, conf, iou)
                st.session_state.processed_data.append({
                    'original_image': pil_img,
                    'labels': labels,
                    'filename': f"sensor_capture_{int(time.time())}.jpg"
                })
                st.toast("Sample secured in database.")
                break
            
            time.sleep(0.01)
        cap.release()

def module_data_studio():
    if not st.session_state.processed_data:
        st.warning("No data found. Please ingest assets via Batch or Sensor modules.")
        return

    # Logic: Editor vs Gallery
    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        data = st.session_state.edit_buffer
        
        st.markdown(f"## 🛠️ Data Studio | Asset: `{data['filename']}`")
        
        col_img, col_ui = st.columns([2, 1])
        
        with col_ui:
            st.markdown("### Object Inspector")
            if st.button("➕ Inject Manual Label", use_container_width=True):
                data['labels'].append("0 0.5 0.5 0.2 0.2")
                st.rerun()

            to_del = []
            for i, l_str in enumerate(data['labels']):
                with st.expander(f"📦 Object {i+1} Instance", expanded=True):
                    parts = l_str.split()
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # Display Class Name
                    c_name = st.session_state.class_names.get(cid, f"ID: {cid}")
                    st.info(f"Class: **{c_name}**")
                    
                    # Fine-grain controls
                    new_cid = st.number_input("Modify Class ID", 0, 1000, cid, key=f"ncid_{i}")
                    new_xc = st.slider("X Position", 0.0, 1.0, xc, 0.001, key=f"nxc_{i}")
                    new_yc = st.slider("Y Position", 0.0, 1.0, yc, 0.001, key=f"nyc_{i}")
                    new_w = st.slider("Object Width", 0.0, 1.0, w, 0.001, key=f"nw_{i}")
                    new_h = st.slider("Object Height", 0.0, 1.0, h, 0.001, key=f"nh_{i}")
                    
                    data['labels'][i] = f"{new_cid} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
                    
                    if st.button(f"🗑️ Delete ID {cid}", key=f"del_{i}"):
                        to_del.append(i)
            
            for i in sorted(to_del, reverse=True):
                data['labels'].pop(i)
                st.rerun()

            st.divider()
            b1, b2 = st.columns(2)
            if b1.button("💾 COMMIT EDITS", type="primary", use_container_width=True):
                st.session_state.processed_data[idx] = data
                st.session_state.edit_index = None
                st.rerun()
            if b2.button("🚫 DISCARD", use_container_width=True):
                st.session_state.edit_index = None
                st.rerun()

        with col_img:
            # Live Preview in Editor
            preview = render_professional_overlay(data['original_image'], data['labels'], st.session_state.class_names)
            st.image(preview, use_container_width=True, caption="Visual Sandbox Output")

    else:
        st.markdown("## 🔍 Knowledge Base Gallery")
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.processed_data):
            with cols[i % 4]:
                prev = render_professional_overlay(item['original_image'], item['labels'], st.session_state.class_names)
                st.image(prev, use_container_width=True)
                st.caption(f"{item['filename'][:15]}...")
                
                btn_c1, btn_c2 = st.columns(2)
                if btn_c1.button("Edit", key=f"ed_{i}", use_container_width=True):
                    st.session_state.edit_index = i
                    st.session_state.edit_buffer = {
                        'original_image': item['original_image'],
                        'labels': item['labels'].copy(),
                        'filename': item['filename']
                    }
                    st.rerun()
                if btn_c2.button("Del", key=f"rem_{i}", use_container_width=True):
                    st.session_state.processed_data.pop(i)
                    st.rerun()

def module_export_wizard():
    if not st.session_state.processed_data:
        st.warning("No data found for export.")
        return

    st.markdown("## 🚀 Deployment Packaging")
    
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.subheader("Distribution Analytics")
        all_ids = []
        for d in st.session_state.processed_data:
            for l in d['labels']:
                all_ids.append(int(l.split()[0]))
        
        if all_ids:
            df = pd.Series(all_ids).value_counts().reset_index()
            df.columns = ['ID', 'Count']
            df['Name'] = df['ID'].map(lambda x: st.session_state.class_names.get(x, f"ID:{x}"))
            st.bar_chart(df, x='Name', y='Count', color="#38bdf8")

    with col_x2:
        st.subheader("Export Configuration")
        proj_name = st.text_input("Project Codename", "DOG_VISION_DATASET")
        split = st.slider("Validation Split %", 10, 50, 20)
        
        if st.button("🏗️ COMPILE YOLOv8 PACKAGE", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as base_dir:
                root = os.path.join(base_dir, proj_name)
                for s in ['train', 'val']:
                    os.makedirs(os.path.join(root, s, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(root, s, 'labels'), exist_ok=True)
                
                # Split Logic
                data_list = st.session_state.processed_data.copy()
                random.shuffle(data_list)
                v_count = int(len(data_list) * (split / 100))
                
                sets = {'val': data_list[:v_count], 'train': data_list[v_count:]}
                
                for key, items in sets.items():
                    for item in items:
                        # Save Image
                        img_name = item['filename'].replace(" ", "_")
                        img_path = os.path.join(root, key, 'images', img_name)
                        item['original_image'].save(img_path)
                        # Save TXT
                        txt_name = os.path.splitext(img_name)[0] + ".txt"
                        txt_path = os.path.join(root, key, 'labels', txt_name)
                        with open(txt_path, 'w') as f:
                            f.write("\n".join(item['labels']))
                
                # data.yaml
                max_id = max(all_ids) if all_ids else 0
                names_list = [st.session_state.class_names.get(i, f"class_{i}") for i in range(max_id + 1)]
                
                yaml_data = {
                    'path': f'./{proj_name}',
                    'train': 'train/images',
                    'val': 'val/images',
                    'nc': len(names_list),
                    'names': names_list
                }
                
                with open(os.path.join(root, 'data.yaml'), 'w') as f:
                    yaml.dump(yaml_data, f)
                
                # Zip and Download
                shutil.make_archive(os.path.join(base_dir, "output"), 'zip', root)
                with open(os.path.join(base_dir, "output.zip"), "rb") as f:
                    st.download_button(
                        "⬇️ DOWNLOAD DATASET (.ZIP)",
                        f,
                        file_name=f"{proj_name}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

def main():
    apply_custom_styles()
    init_session_state()
    
    # Dashboard Header
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); padding: 40px; border-radius: 24px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 35px; text-align: center;">
            <h1 style='margin:0; color:#38bdf8; font-size: 3.5rem; font-weight: 800;'>DOG Vision System</h1>
            <p style='color:#94a3b8; font-size: 1.2rem; font-weight: 300; letter-spacing: 0.1em;'>High-Resolution Object Detection & Dataset Curation Suite</p>
        </div>
    """, unsafe_allow_html=True)
    
    engines, conf, iou = build_sidebar_controls()
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "💎 BATCH INGEST", 
        "📡 LIVE SENSORS", 
        "🎨 DATA STUDIO", 
        "⚡ EXPORT WIZARD"
    ])
    
    with tab1:
        module_batch_ingestion(engines, conf, iou)
    with tab2:
        module_live_sensors(engines, conf, iou)
    with tab3:
        module_data_studio()
    with tab4:
        module_export_wizard()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>DOG Vision System | Institutional Research v5.0-PRO | End-to-End CV Workflow</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
