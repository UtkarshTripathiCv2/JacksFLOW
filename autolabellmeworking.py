import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageColor
import os
import tempfile
import shutil
import pandas as pd
import time
import yaml
import random
import io
import zipfile

st.set_page_config(
    page_title="DOG Vision System | Dataset Engineering",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_styles():
    """
    Applies a high-end dark-mode CSS theme with glassmorphism and institutional design tokens.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .main {
            background: radial-gradient(circle at top right, #0f172a, #020617);
            color: #f8fafc;
        }
        
        /* Premium Card Containers */
        [data-testid="stVerticalBlock"] > div:has(div.metric-card) {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.4);
            margin-bottom: 20px;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(56, 189, 248, 0.15);
            padding: 20px;
            border-radius: 18px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 3rem;
            font-weight: 800;
            color: #38bdf8;
            text-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            margin-top: 5px;
        }

        /* Sidebar & Navigation Styling */
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid rgba(56, 189, 248, 0.15);
        }
        
        .class-badge {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
            margin: 2px;
            display: inline-block;
        }

        .stButton>button {
            border-radius: 14px;
            background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stButton>button:hover {
            box-shadow: 0 0 30px rgba(59, 130, 246, 0.4);
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """
    Maintains the global database for processed frames, active labels, and edit buffers.
    """
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = []
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None
    if 'edit_buffer' not in st.session_state:
        st.session_state.edit_buffer = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = {}
    if 'model_objects' not in st.session_state:
        st.session_state.model_objects = {}

@st.cache_resource
def load_yolo_engine(model_path):
    """
    Safely loads YOLO weights. Handles missing local files by using defaults.
    """
    try:
        if not os.path.exists(model_path):
            # If the user's specific .pt isn't present, we notify but don't crash
            return None
        return YOLO(model_path)
    except Exception:
        return None

def execute_fusion_inference(image_bgr, engines, conf_thresh, iou_thresh):
    """
    Runs an ensemble detection across all selected engines.
    Merges results into a unique YOLO-formatted label list.
    """
    aggregated_labels = []
    
    for name, model in engines.items():
        if model is None: continue
        
        # Perform Inference
        results = model.predict(image_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
        res = results[0]
        
        if len(res.boxes) > 0:
            boxes = res.boxes.xywhn.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                c_id = int(classes[i])
                x, y, w, h = boxes[i]
                # Label format: class_id x_center y_center width height
                label_str = f"{c_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                aggregated_labels.append(label_str)
                
    # Remove identical duplicates if multiple models find the same box
    return list(set(aggregated_labels))

def draw_professional_bboxes(pil_img, labels_list, class_map):
    """
    Renders high-contrast bounding boxes with class name strings.
    """
    frame = np.array(pil_img.convert("RGB"))
    h, w, _ = frame.shape
    
    # Designer palette for high-visibility
    palette = [
        (56, 189, 248), (248, 113, 113), (163, 230, 53), (251, 191, 36),
        (192, 132, 252), (244, 114, 182), (45, 212, 191), (255, 255, 255)
    ]
    
    for label in labels_list:
        try:
            parts = label.split()
            cid = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            
            # Un-normalize coordinates
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            color = palette[cid % len(palette)]
            name = class_map.get(cid, f"Object {cid}").upper()
            
            # Drawing the Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Drawing the Label Tag
            text = f" {name} "
            font = cv2.FONT_HERSHEY_DUPLEX
            fs, thick = 0.5, 1
            (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
            
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 7), font, fs, (0, 0, 0), thick)
            
        except Exception:
            continue
            
    return frame

def build_controller():
    """
    Global control panel for the Vision System.
    """
    with st.sidebar:
        st.markdown("<div style='text-align: center; padding: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color: #38bdf8; margin-bottom: 0;'>🐕 DOG VISION</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.75rem; letter-spacing: 0.3em;'>SYSTEM CORE v5.0</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("🧬 Intelligence Engines")
        # Custom model requests
        AVAILABLE = ["dog.pt", "one.pt", "yolov8n.pt", "yolov9c.pt"]
        selected_paths = st.multiselect("Active Weights", AVAILABLE, default=["yolov8n.pt"])
        
        st.subheader("⚙️ Vision Parameters")
        conf = st.slider("Confidence Gate", 0.01, 1.0, 0.25)
        iou = st.slider("IOU Sensitivity", 0.1, 1.0, 0.45)
        
        # Load models and extract class names
        active_engines = {}
        unified_class_map = {}
        
        for path in selected_paths:
            m = load_yolo_engine(path)
            if m:
                active_engines[path] = m
                if m.names:
                    unified_class_map.update(m.names)
            else:
                st.warning(f"File '{path}' not found in root directory.")
                
        st.session_state.class_names = unified_class_map
        st.session_state.model_objects = active_engines
        
        st.divider()
        st.subheader("🔍 Class Dictionary")
        if unified_class_map:
            st.markdown("Use these IDs for manual editing:")
            # Displaying classes in a pretty badge format
            cols = st.columns(1)
            with cols[0]:
                for cid, name in sorted(unified_class_map.items()):
                    st.markdown(f"<span class='class-badge'>{cid}: {name}</span>", unsafe_allow_html=True)
        else:
            st.info("Load a model to see classes.")
            
        st.divider()
        if st.button("🗑️ PURGE SYSTEM MEMORY", use_container_width=True):
            st.session_state.processed_data = []
            st.rerun()
            
    return active_engines, conf, iou

def module_batch_ingestion(engines, conf, iou):
    st.markdown("## 📥 High-Capacity Batch Ingestion")
    st.caption("Upload raw asset sets for recursive neural auto-labeling.")
    
    files = st.file_uploader("Drop image assets here (JPG/PNG/WEBP)", type=["jpg", "png", "webp", "jpeg"], accept_multiple_files=True)
    
    if files:
        if st.button("🚀 EXECUTE AUTO-LABELING CYCLE", use_container_width=True):
            p_bar = st.progress(0)
            status = st.empty()
            
            for i, f in enumerate(files):
                status.text(f"Analyzing Visual Data: {f.name}")
                img = Image.open(f).convert("RGB")
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Multi-engine fusion
                labels = execute_fusion_inference(bgr, engines, conf, iou)
                
                st.session_state.processed_data.append({
                    'original_image': img,
                    'labels': labels,
                    'filename': f"{int(time.time())}_{f.name}"
                })
                p_bar.progress((i + 1) / len(files))
                
            status.success(f"Ingestion successful. {len(files)} assets added to Knowledge Base.")
            st.balloons()

def module_data_studio():
    if not st.session_state.processed_data:
        st.warning("Knowledge Base is empty. Please ingest data via the Ingestion module.")
        return

    # Sub-Router: Editor vs Gallery
    if st.session_state.edit_index is not None:
        idx = st.session_state.edit_index
        buffer = st.session_state.edit_buffer
        
        st.markdown(f"## 🛠️ Data Studio | Asset: `{buffer['filename']}`")
        
        c1, c2 = st.columns([2, 1])
        
        with c2:
            st.markdown("### Object Inspector")
            if st.button("➕ Inject New Instance", use_container_width=True):
                buffer['labels'].append("0 0.5 0.5 0.2 0.2")
                st.rerun()

            to_del = []
            for i, l_str in enumerate(buffer['labels']):
                with st.expander(f"📦 Instance {i+1}", expanded=True):
                    try:
                        p = l_str.split()
                        cid = int(p[0])
                        xc, yc, bw, bh = map(float, p[1:])
                        
                        # Show current name
                        cname = st.session_state.class_names.get(cid, f"ID {cid}")
                        st.markdown(f"Detected as: **{cname.upper()}**")
                        
                        # Fine-tuning
                        new_cid = st.number_input("Class ID", 0, 999, cid, key=f"cid_{i}")
                        new_xc = st.slider("X Position", 0.0, 1.0, xc, 0.001, key=f"xc_{i}")
                        new_yc = st.slider("Y Position", 0.0, 1.0, yc, 0.001, key=f"yc_{i}")
                        new_bw = st.slider("Width", 0.0, 1.0, bw, 0.001, key=f"bw_{i}")
                        new_bh = st.slider("Height", 0.0, 1.0, bh, 0.001, key=f"bh_{i}")
                        
                        buffer['labels'][i] = f"{new_cid} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}"
                        
                        if st.button(f"🗑️ Remove Instance", key=f"del_{i}"):
                            to_del.append(i)
                    except:
                        st.error("Malformed label string.")
            
            for i in sorted(to_del, reverse=True):
                buffer['labels'].pop(i)
                st.rerun()

            st.divider()
            b1, b2 = st.columns(2)
            if b1.button("💾 COMMIT EDITS", type="primary", use_container_width=True):
                st.session_state.processed_data[idx] = buffer
                st.session_state.edit_index = None
                st.rerun()
            if b2.button("🚫 DISCARD", use_container_width=True):
                st.session_state.edit_index = None
                st.rerun()

        with c1:
            overlay = draw_professional_bboxes(buffer['original_image'], buffer['labels'], st.session_state.class_names)
            st.image(overlay, use_container_width=True, caption="Visual Sandbox Preview")

    else:
        st.markdown("## 🔍 Knowledge Base Gallery")
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.processed_data):
            with cols[i % 4]:
                ov = draw_professional_bboxes(item['original_image'], item['labels'], st.session_state.class_names)
                st.image(ov, use_container_width=True)
                st.caption(f"{item['filename'][:20]}...")
                
                b1, b2 = st.columns(2)
                if b1.button("Edit", key=f"e_{i}", use_container_width=True):
                    st.session_state.edit_index = i
                    st.session_state.edit_buffer = {
                        'original_image': item['original_image'],
                        'labels': list(item['labels']), # Deep copy
                        'filename': item['filename']
                    }
                    st.rerun()
                if b2.button("Del", key=f"d_{i}", use_container_width=True):
                    st.session_state.processed_data.pop(i)
                    st.rerun()

def module_export_wizard():
    if not st.session_state.processed_data:
        st.warning("No data detected for export configuration.")
        return

    st.markdown("## 🚀 Deployment Packaging & Analytics")
    
    col_a, col_b = st.columns([2, 3])
    
    with col_a:
        st.subheader("📊 Class Distribution")
        all_ids = []
        for d in st.session_state.processed_data:
            for l in d['labels']:
                try:
                    all_ids.append(int(l.split()[0]))
                except: continue
        
        if all_ids:
            df = pd.Series(all_ids).value_counts().reset_index()
            df.columns = ['ID', 'Count']
            df['Label'] = df['ID'].map(lambda x: st.session_state.class_names.get(x, f"ID:{x}"))
            st.bar_chart(df, x='Label', y='Count', color="#38bdf8")
        else:
            st.info("No detections found to analyze.")

    with col_b:
        st.subheader("📦 Export Configuration")
        proj_name = st.text_input("Project Name", "DOG_VISION_DATASET")
        split_val = st.slider("Validation Split (%)", 5, 50, 20)
        
        st.info("Format: YOLOv8 standard (Images & Labels subfolders with data.yaml)")
        
        if st.button("🏗️ COMPILE & DOWNLOAD DATASET (.ZIP)", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Structure: root/train/images, root/train/labels...
                root = os.path.join(tmp_dir, proj_name)
                for s in ['train', 'val']:
                    os.makedirs(os.path.join(root, s, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(root, s, 'labels'), exist_ok=True)
                
                # Split Logic
                data_pool = list(st.session_state.processed_data)
                random.shuffle(data_pool)
                v_size = int(len(data_pool) * (split_val / 100))
                
                sets = {'val': data_pool[:v_size], 'train': data_pool[v_size:]}
                
                for mode, items in sets.items():
                    for item in items:
                        # Sanitize filename
                        fname = item['filename'].replace(" ", "_")
                        # Save Image
                        img_path = os.path.join(root, mode, 'images', fname)
                        item['original_image'].save(img_path)
                        # Save Label TXT
                        txt_name = os.path.splitext(fname)[0] + ".txt"
                        txt_path = os.path.join(root, mode, 'labels', txt_name)
                        with open(txt_path, 'w') as f:
                            f.write("\n".join(item['labels']))
                
                # Create data.yaml
                max_id = max(all_ids) if all_ids else 0
                names_list = [st.session_state.class_names.get(i, f"class_{i}") for i in range(max_id + 1)]
                
                yaml_content = {
                    'path': f'../{proj_name}',
                    'train': 'train/images',
                    'val': 'val/images',
                    'nc': len(names_list),
                    'names': names_list
                }
                
                with open(os.path.join(root, 'data.yaml'), 'w') as f:
                    yaml.dump(yaml_content, f, default_flow_style=False)
                
                # Zip the root
                zip_buffer = io.BytesIO()
                shutil.make_archive(os.path.join(tmp_dir, "dist"), 'zip', root)
                
                with open(os.path.join(tmp_dir, "dist.zip"), "rb") as f:
                    st.download_button(
                        label="⬇️ DOWNLOAD PREPARED DATASET",
                        data=f,
                        file_name=f"{proj_name}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

def main():
    apply_custom_styles()
    init_session_state()
    
    # Hero Section
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.4); padding: 50px; border-radius: 30px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 40px; text-align: center;">
            <h1 style='margin:0; color:#38bdf8; font-size: 4rem; font-weight: 800; letter-spacing: -0.02em;'>DOG Vision System</h1>
            <p style='color:#94a3b8; font-size: 1.3rem; font-weight: 300; letter-spacing: 0.15em; margin-top: 10px;'>Institutional Intelligence for Automated Visual Labeling</p>
        </div>
    """, unsafe_allow_html=True)
    
    engines, conf, iou = build_controller()
    
    # Primary Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "💎 BATCH INGESTION", 
        "🎨 DATA STUDIO", 
        "⚡ EXPORT & ANALYTICS"
    ])
    
    with tab1:
        module_batch_ingestion(engines, conf, iou)
    with tab2:
        module_data_studio()
    with tab3:
        module_export_wizard()

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>DOG Vision System | Distributed AI Training Suite v5.0-PRO | © 2024 Institutional CV Research</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
