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
        st.error(f"Error loading model '{model_path}': {e}")
        st.error("Please ensure the model path is correct and the file is accessible.")
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

# --- UI Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    model_path = st.text_input("YOLO Model Path (.pt)", "yolov8n.pt")
    
    model = load_yolo_model(model_path)
    
    # Store class names in session state for the editor
    class_names = {}
    if model and model.names:
        class_names = model.names
        st.session_state.class_names = class_names
    else:
        st.session_state.class_names = {} # Ensure it exists

    if model:
        st.success(f"Model '{model_path}' loaded successfully.")
        st.subheader("Model Classes")
        if class_names:
            class_df = pd.DataFrame(class_names.items(), columns=["Class ID", "Name"])
            st.dataframe(class_df, use_container_width=True)
        else:
            st.info("This model does not have class names embedded.")
    else:
        st.error("Model could not be loaded. Please check the path.")

# --- Main App Interface ---
st.title("🤖 YOLO Annotation Assistant")
st.write("Upload images, use your webcam, then review, edit, and export your complete YOLO dataset.")

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Upload", 
    "📹 Live Video", 
    "📦 Review, Edit & Export" # Tab name changed
])

# --- Image Upload Tab (No Changes) ---
with tab1:
    st.header("Upload Image Files")
    if not model:
        st.warning("Please load a model in the sidebar to begin.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} images...")
        col1, col2 = st.columns(2)
        
        for uploaded_file in uploaded_files:
            original_image = Image.open(uploaded_file).convert("RGB")
            frame_rgb = np.array(original_image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            annotated_frame, labels = process_frame(frame_bgr, model)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            with col1:
                st.image(original_image, caption=f"Original: {uploaded_file.name}", use_container_width=True)
            with col2:
                st.image(annotated_frame_rgb, caption=f"Annotated: {uploaded_file.name}", use_container_width=True)

            st.session_state.processed_data.append({
                'original_image': original_image,
                'labels': labels,
                'filename': uploaded_file.name
            })
        
        st.success(f"Processed and added {len(uploaded_files)} images to the collection.")
        st.info("Navigate to the 'Review, Edit & Export' tab to see your collection.")


# --- Live Video Tab (No Changes) ---
with tab2:
    st.header("Live Webcam Feed")
    if not model:
        st.warning("Please load a model in the sidebar to begin.")
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
                    annotated_frame, labels = process_frame(frame_bgr, model)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    original_image = Image.fromarray(frame_rgb)
                    timestamp = f"frame_{len(st.session_state.processed_data) + 1}.jpg"
                    
                    st.session_state.processed_data.append({
                        'original_image': original_image,
                        'labels': labels,
                        'filename': timestamp
                    })
                    st.success(f"Frame '{timestamp}' captured and added!")
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(annotated_frame_rgb)
                else:
                    st.error("Failed to capture frame from webcam.")

            while run:
                success, frame_bgr = camera.read()
                if not success:
                    st.error("Failed to capture frame from webcam.")
                    break
                
                annotated_frame, _ = process_frame(frame_bgr, model)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated_frame_rgb)
        
            camera.release()
            st.write("Webcam stopped.")
            
# --- Review, Edit & Export Tab (HEAVILY MODIFIED) ---
with tab3:
    st.header("Review, Edit & Export")

    if not st.session_state.processed_data:
        st.info("You have no items in your collection. Upload images or capture frames to begin.")
        st.stop()

    st.info(f"You have {len(st.session_state.processed_data)} items in your collection.")
    st.markdown("---")

    # --- 1. Dataset Configuration ---
    st.subheader("1. Dataset Configuration")
    dataset_name = st.text_input("Enter Dataset Name", "my_dataset")
    
    generate_yaml = st.checkbox("Include data.yaml file", value=True)
    class_names_input = ""
    
    if generate_yaml:
        default_names = ""
        if 'class_names' in st.session_state and st.session_state.class_names:
            default_names = "\n".join(st.session_state.class_names.values())
        else:
            default_names = "class1\nclass2\n..."
            st.warning("Could not auto-detect class names. Please define them manually.")
            
        class_names_input = st.text_area(
            "Class Names (one per line)", 
            value=default_names, 
            height=150,
            help="Ensure this list is in the correct order and matches the class IDs from your model."
        )

    st.markdown("---")

    # --- 2. Prepare & Download ---
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
        st.success("Zip file prepared successfully!") # This one is fine, it's after the zip
    st.markdown("---")

    # --- 3. Review & Edit Gallery / Detailed Editor ---
    
    # Check if we are in "Edit Mode"
    if st.session_state.edit_index is not None:
        
        # --- DETAILED EDITOR VIEW ---
        st.subheader("Detailed Bounding Box Editor")
        
        # --- NEW Button Bar ---
        btn_cols = st.columns([1, 1, 3]) # Give buttons proportional space
        with btn_cols[0]:
            if st.button("💾 Save and Go Back", use_container_width=True, type="primary"):
                # Save: Copy from buffer to main list
                st.session_state.processed_data[st.session_state.edit_index] = st.session_state.edit_buffer
                # Clear buffer and index
                st.session_state.edit_index = None
                st.session_state.edit_buffer = None
                st.toast("Changes saved!", icon="💾") # Use toast for feedback
                st.rerun() # Go back to gallery
        
        with btn_cols[1]:
            if st.button("↩️ Cancel (Discard Changes)", use_container_width=True):
                # Discard: Just clear buffer and index
                st.session_state.edit_index = None
                st.session_state.edit_buffer = None
                st.toast("Changes discarded.", icon="↩️") # Use toast for feedback
                st.rerun() # Go back to gallery
        
        # Need to check if state was cleared before proceeding
        if st.session_state.edit_index is None:
            st.stop() # Stop rendering this page

        # Get the data for the selected image *from the buffer*
        try:
            edit_data = st.session_state.edit_buffer 
            original_image = edit_data['original_image']
            labels = edit_data['labels']
        except (IndexError, TypeError, AttributeError):
            # This can happen if state is weird
            st.session_state.edit_index = None
            st.session_state.edit_buffer = None
            st.rerun()

        
        img_col, controls_col = st.columns([2, 1])
        
        # --- Controls Column (Sliders, Buttons) ---
        with controls_col:
            st.subheader(f"Editing: {edit_data['filename']}")
            
            if st.button("Add New Box"):
                # Modify the buffer
                st.session_state.edit_buffer['labels'].append("0 0.5 0.5 0.2 0.2")
                st.toast("New box added!", icon="➕") # Use toast for feedback
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

                # Update the label string in the *buffer* immediately for real-time preview
                new_label_str = f"{new_class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}"
                st.session_state.edit_buffer['labels'][i] = new_label_str
                
                if st.button(f"Delete Box {i+1}", key=f"{key_prefix}_del", type="primary"):
                    box_indices_to_delete.append(i)

            if box_indices_to_delete:
                for i in sorted(box_indices_to_delete, reverse=True):
                    # Modify the buffer
                    st.session_state.edit_buffer['labels'].pop(i)
                st.toast(f"Deleted {len(box_indices_to_delete)} box(es)!", icon="🗑️") # Use toast
                st.rerun()

        # --- Image Column (Live Preview) ---
        with img_col:
            # Read from the buffer
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
            st.session_state.edit_buffer = None # Clear buffer
            st.toast("All data cleared!", icon="🗑️") # Use toast
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
                        # --- NEW: Create the edit buffer ---
                        # Create a copy of the data to edit
                        st.session_state.edit_buffer = data.copy() 
                        # Also create a copy of the labels list
                        st.session_state.edit_buffer['labels'] = data['labels'].copy()
                        st.rerun()
                
                with b_col2:
                    if st.button("Remove", key=f"remove_{i}", type="primary"):
                        indices_to_remove.append(i)

        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                st.session_state.processed_data.pop(index)
            st.session_state.edit_index = None # Ensure edit index is cleared
            st.session_state.edit_buffer = None # Ensure buffer is cleared
            st.toast(f"Removed {len(indices_to_remove)} image(s)!", icon="🗑️") # Use toast
            st.rerun()

