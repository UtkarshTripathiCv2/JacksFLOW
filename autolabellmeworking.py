import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import shutil
import zipfile
import pandas as pd

# --- Configuration ---
st.set_page_config(
    page_title="YOLO Annotation Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path, caching it for performance."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        # Show a less intrusive error in the sidebar
        st.sidebar.error(f"Error loading '{model_path}': {e}", icon="⚠️")
        return None

# --- Core Processing Function ---
def process_frame(frame, model):
    """
    Performs object detection on a single frame (image) and returns the annotated frame
    and a list of YOLO-formatted label strings.
    """
    results = model(frame, verbose=False)
    result = results[0]
    annotated_frame = result.plot()
    
    labels = []
    if len(result.boxes) > 0:
        boxes = result.boxes.xywhn
        classes = result.boxes.cls
        
        for i in range(len(boxes)):
            class_id = int(classes[i])
            x_center, y_center, width, height = boxes[i]
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
    return annotated_frame, labels

# --- NEW HELPER: Draw boxes from label strings ---
def draw_boxes_on_image(pil_image, labels_list, class_names):
    """
    Draws bounding boxes from a list of YOLO label strings onto a PIL image.
    Returns an np.array (RGB) with boxes drawn.
    """
    frame_rgb = np.array(pil_image.convert("RGB"))
    img_h, img_w, _ = frame_rgb.shape
    
    # Create a copy to draw on
    draw_frame = frame_rgb.copy()
    
    for label_str in labels_list:
        try:
            parts = label_str.split()
            class_id = int(parts[0])
            x_c_norm = float(parts[1])
            y_c_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])
            
            # Un-normalize coordinates
            x_c_px = x_c_norm * img_w
            y_c_px = y_c_norm * img_h
            w_px = w_norm * img_w
            h_px = h_norm * img_h
            
            x1 = int(x_c_px - (w_px / 2))
            y1 = int(y_c_px - (h_px / 2))
            x2 = int(x_c_px + (w_px / 2))
            y2 = int(y_c_px + (h_px / 2))
            
            # Get class name
            label_name = class_names.get(class_id, f"ID: {class_id}")
            
            # Draw rectangle (red, thickness 2)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw text
            cv2.putText(draw_frame, label_name, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        except Exception as e:
            # Log error to console, not to Streamlit UI
            print(f"Error drawing box for label '{label_str}': {e}")
    
    return draw_frame


# --- Session State Initialization ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
# Index for image being edited
if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None
# --- NEW: Buffer for editing ---
if 'edit_buffer' not in st.session_state:
    st.session_state.edit_buffer = None
# --- NEW: Store all class names ---
if 'class_names' not in st.session_state:
    st.session_state.class_names = {}

# --- UI Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # --- NEW: Multiple Model Selection ---
    MODEL_LIST = ["yolov8n.pt", "one.pt", "two.pt", "three.pt", "four.pt", "dog.pt"]
    
    selected_model_paths = st.multiselect(
        "Select Models to Run", 
        MODEL_LIST, 
        default=MODEL_LIST[0]
    )

    if not selected_model_paths:
        st.error("Please select at least one model to run.")
        st.stop()

    live_preview_model_path = st.selectbox(
        "Select Model for Live Preview", 
        selected_model_paths, 
        index=0,
        help="This model will be used for the live webcam feed for performance."
    )
    
    # Load the live model for webcam and sidebar display
    live_model = load_yolo_model(live_preview_model_path)
    
    # Load all selected models (cached) and combine their class names
    all_class_names = {}
    loaded_models = {}
    
    for path in selected_model_paths:
        model = load_yolo_model(path)
        if model:
            loaded_models[path] = model
            if model.names:
                all_class_names.update(model.names)

    st.session_state.class_names = all_class_names

    if live_model:
        st.success(f"Live preview model '{live_preview_model_path}' loaded.")
        st.subheader("Live Model Classes")
        if live_model.names:
            class_df = pd.DataFrame(live_model.names.items(), columns=["Class ID", "Name"])
            st.dataframe(class_df, use_container_width=True)
        else:
            st.info("This model does not have class names embedded.")
    else:
        st.error("Live preview model could not be loaded. Please check the path.")

# --- Main App Interface ---
st.title("🤖 YOLO Annotation Assistant")
st.write("Upload images, use your webcam, then review, edit, and export your complete YOLO dataset.")

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Upload", 
    "📹 Live Video", 
    "📦 Review, Edit & Export" # Tab name changed
])

# --- Image Upload Tab (MODIFIED) ---
with tab1:
    st.header("Upload Image Files")
    if not loaded_models:
        st.warning("Please load at least one model in the sidebar to begin.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} images with {len(loaded_models)} model(s)...")
        col1, col2 = st.columns(2)
        
        for uploaded_file in uploaded_files:
            original_image = Image.open(uploaded_file).convert("RGB")
            frame_rgb = np.array(original_image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # --- NEW: Run all selected models ---
            all_labels = []
            for model_name, model in loaded_models.items():
                _, labels = process_frame(frame_bgr, model)
                all_labels.extend(labels)

            # --- NEW: Draw combined predictions ---
            annotated_frame_rgb = draw_boxes_on_image(
                original_image, 
                all_labels, 
                st.session_state.class_names
            )

            with col1:
                st.image(original_image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
            with col2:
                st.image(annotated_frame_rgb, caption=f"Combined Predictions: {uploaded_file.name}", use_container_width=True)

            st.session_state.processed_data.append({
                'original_image': original_image,
                'labels': all_labels, # Store the combined list
                'filename': uploaded_file.name
            })
        
        st.success(f"Processed and added {len(uploaded_files)} images to the collection.")
        st.info("Navigate to the 'Review, Edit & Export' tab to see your collection.")


# --- Live Video Tab (MODIFIED) ---
with tab2:
    st.header("Live Webcam Feed")
    if not live_model:
        st.warning("Please load a live preview model in the sidebar to begin.")
        st.stop()
        
    run = st.checkbox('Start Webcam', key='run_webcam')
    
    if run:
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Could not open webcam. Please grant browser permission.")
        else:
            if st.button("📸 Capture & Add Frame"):
                success, frame_bgr = camera.read()
                if success:
                    # --- NEW: Run all models on capture ---
                    all_labels = []
                    for model_name, model in loaded_models.items():
                        _, labels = process_frame(frame_bgr, model)
                        all_labels.extend(labels)
                    
                    original_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    timestamp = f"frame_{len(st.session_state.processed_data) + 1}.jpg"
                    
                    st.session_state.processed_data.append({
                        'original_image': original_image,
                        'labels': all_labels, # Store combined labels
                        'filename': timestamp
                    })
                    st.success(f"Frame '{timestamp}' captured and added with {len(all_labels)} total detections!")
                    
                    # --- NEW: Show combined annotated frame ---
                    annotated_frame_rgb = draw_boxes_on_image(
                        original_image, 
                        all_labels, 
                        st.session_state.class_names
                    )
                    FRAME_WINDOW.image(annotated_frame_rgb)
                else:
                    st.error("Failed to capture frame from webcam.")

            while run:
                success, frame_bgr = camera.read()
                if not success:
                    st.error("Failed to capture frame from webcam.")
                    break
                
                # Live feed uses *only* the live_model
                annotated_frame, _ = process_frame(frame_bgr, live_model)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame_rgb)
        
            camera.release()
            st.write("Webcam stopped.")
            
# --- Review, Edit & Export Tab (MODIFIED) ---
with tab3:
    st.header("Review, Edit & Export")

    if not st.session_state.processed_data:
        st.info("You have no items in your collection. Upload images or capture frames to begin.")
        st.stop()

    st.info(f"You have {len(st.session_state.processed_data)} items in your collection.")
    st.markdown("---")

    # --- 1. Dataset Configuration (MODIFIED for combined class list) ---
    st.subheader("1. Dataset Configuration")
    dataset_name = st.text_input("Enter Dataset Name", "my_dataset")
    
    generate_yaml = st.checkbox("Include data.yaml file", value=True)
    class_names_input = ""
    
    if generate_yaml:
        default_names = ""
        all_class_names = st.session_state.get('class_names', {})
        if all_class_names:
            # Sort by class ID (key) to get the correct order
            sorted_names = [v for k, v in sorted(all_class_names.items())]
            default_names = "\n".join(sorted_names)
        else:
            default_names = "class1\nclass2\n..."
            st.warning("Could not auto-detect class names. Please define them manually.")
            
        class_names_input = st.text_area(
            "Class Names (one per line)", 
            value=default_names, 
            height=150,
            help="Ensure this list is in the correct order and matches the class IDs from your model(s)."
        )

    st.markdown("---")

    # --- 2. Prepare & Download (No logic changes) ---
    st.subheader("2. Prepare & Download Dataset")
    
    if st.button(f"📦 Prepare & Download '{dataset_name}.zip'"):
        with st.spinner("Zipping your dataset... Please wait."):
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_root = os.path.join(tmpdir, dataset_name)
                images_dir = os.path.join(dataset_root, 'images')
                labels_dir = os.path.join(dataset_root, 'labels')
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                # Save all images and *updated* labels
                for data in st.session_state.processed_data:
                    original_image = data['original_image']
                    labels = data['labels']
                    base_filename = data['filename']
                    
                    img_path = os.path.join(images_dir, base_filename)
                    original_image.save(img_path)
                    
                    if labels:
                        label_filename = os.path.splitext(base_filename)[0] + '.txt'
                        label_path = os.path.join(labels_dir, label_filename)
                        with open(label_path, 'w') as f:
                            f.write("\n".join(labels))

                if generate_yaml:
                    class_names_list = [name.strip() for name in class_names_input.split("\n") if name.strip()]
                    num_classes = len(class_names_list)
                    
                    yaml_content = f"""
path: .
train: images
val: images

nc: {num_classes}
names: {class_names_list}
"""
                    yaml_path = os.path.join(dataset_root, 'data.yaml')
                    with open(yaml_path, 'w') as f:
                        f.write(yaml_content)

                zip_path = os.path.join(tmpdir, f"{dataset_name}")
                shutil.make_archive(zip_path, 'zip', dataset_root)
                
                zip_filename = f"{dataset_name}.zip"
                zip_file_path = f"{zip_path}.zip"

                with open(zip_file_path, "rb") as fp:
                    st.download_button(
                        label="✅ Download Complete! Click here.",
                        data=fp,
                        file_name=zip_filename,
                        mime="application/zip",
                        key='download_zip_button'
                    )
        st.success("Zip file prepared successfully!")
    st.markdown("---")

    # --- 3. Review & Edit Gallery / Detailed Editor (No logic changes) ---
    
    # Check if we are in "Edit Mode"
    if st.session_state.edit_index is not None:
        
        # --- DETAILED EDITOR VIEW ---
        st.subheader("Detailed Bounding Box Editor")
        
        # --- Button Bar ---
        btn_cols = st.columns([1, 1, 3]) 
        with btn_cols[0]:
            if st.button("💾 Save and Go Back", use_container_width=True, type="primary"):
                st.session_state.processed_data[st.session_state.edit_index] = st.session_state.edit_buffer
                st.session_state.edit_index = None
                st.session_state.edit_buffer = None
                st.toast("Changes saved!", icon="💾") 
                st.rerun() 
        
        with btn_cols[1]:
            if st.button("↩️ Cancel (Discard Changes)", use_container_width=True):
                st.session_state.edit_index = None
                st.session_state.edit_buffer = None
                st.toast("Changes discarded.", icon="↩️") 
                st.rerun() 
        
        if st.session_state.edit_index is None:
            st.stop() 

        try:
            edit_data = st.session_state.edit_buffer 
            original_image = edit_data['original_image']
            labels = edit_data['labels']
        except (IndexError, TypeError, AttributeError):
            st.session_state.edit_index = None
            st.session_state.edit_buffer = None
            st.rerun()

        
        img_col, controls_col = st.columns([2, 1])
        
        # --- Controls Column (Sliders, Buttons) ---
        with controls_col:
            st.subheader(f"Editing: {edit_data['filename']}")
            
            if st.button("Add New Box"):
                st.session_state.edit_buffer['labels'].append("0 0.5 0.5 0.2 0.2")
                st.toast("New box added!", icon="➕") 
                st.rerun()

            box_indices_to_delete = []
            
            for i, label_str in enumerate(labels):
                st.markdown(f"---")
                st.markdown(f"**Box {i+1}**")
                
                try:
                    parts = label_str.split()
                    class_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except Exception:
                    st.error(f"Error parsing label: {label_str}")
                    continue

                key_prefix = f"edit_{st.session_state.edit_index}_{i}"
                
                new_class_id = st.number_input("Class ID", min_value=0, value=class_id, key=f"{key_prefix}_class")
                new_x_c = st.slider("X Center", 0.0, 1.0, value=x_c, step=0.001, key=f"{key_prefix}_x")
                new_y_c = st.slider("Y Center", 0.0, 1.0, value=y_c, step=0.001, key=f"{key_prefix}_y")
                new_w = st.slider("Width", 0.0, 1.0, value=w, step=0.001, key=f"{key_prefix}_w")
                new_h = st.slider("Height", 0.0, 1.0, value=h, step=0.001, key=f"{key_prefix}_h")

                new_label_str = f"{new_class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}"
                st.session_state.edit_buffer['labels'][i] = new_label_str
                
                if st.button(f"Delete Box {i+1}", key=f"{key_prefix}_del", type="primary"):
                    box_indices_to_delete.append(i)

            if box_indices_to_delete:
                for i in sorted(box_indices_to_delete, reverse=True):
                    st.session_state.edit_buffer['labels'].pop(i)
                st.toast(f"Deleted {len(box_indices_to_delete)} box(es)!", icon="🗑️") 
                st.rerun()

        # --- Image Column (Live Preview) ---
        with img_col:
            updated_labels = st.session_state.edit_buffer['labels']
            image_with_boxes = draw_boxes_on_image(
                original_image, 
                updated_labels, 
                st.session_state.get('class_names', {})
            )
            st.image(image_with_boxes, caption="Live Preview of Edits", use_container_width=True)

    else:
        
        # --- GALLERY VIEW ---
        st.subheader("3. Review & Edit Gallery")
        
        if st.button("🗑️ Clear All Collected Data"):
            st.session_state.processed_data = []
            st.session_state.edit_index = None
            st.session_state.edit_buffer = None 
            st.toast("All data cleared!", icon="🗑️") 
            st.rerun()

        st.warning("Clicking 'Remove' is permanent for this session.")
        
        indices_to_remove = []
        cols_per_row = 4
        cols = st.columns(cols_per_row)
        
        for i, data in enumerate(st.session_state.processed_data):
            with cols[i % cols_per_row]:
                
                # Draw the *current* boxes on the image for the gallery
                image_with_boxes = draw_boxes_on_image(
                    data['original_image'], 
                    data['labels'], 
                    st.session_state.get('class_names', {})
                )
                st.image(image_with_boxes, caption=data['filename'], use_container_width=True)
                
                # Add Edit and Remove buttons
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button("Edit", key=f"edit_{i}"):
                        st.session_state.edit_index = i # Set index
                        st.session_state.edit_buffer = data.copy() 
                        st.session_state.edit_buffer['labels'] = data['labels'].copy()
                        st.rerun()
                
                with b_col2:
                    if st.button("Remove", key=f"remove_{i}", type="primary"):
                        indices_to_remove.append(i)

        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                st.session_state.processed_data.pop(index)
            st.session_state.edit_index = None 
            st.session_state.edit_buffer = None 
            st.toast(f"Removed {len(indices_to_remove)} image(s)!", icon="🗑️") 
            st.rerun()

