import os
import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageOps
import pickle
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="Face Recognition Hub",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply Outfit font */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Premium Title Styling */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f1f3f5;
    padding: 6px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    color: #FF4B4B !important;
}
</style>
""", unsafe_allow_html=True)

# Absolute path to the known faces file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_FILE = os.path.join(BASE_DIR, 'known_faces.pkl')

# Known face encodings and names (in-memory cache)
known_face_encodings = []
known_face_names = []

# WebRTC cache variables to optimize real-time streaming performance
webrtc_frame_count = 0
webrtc_last_locations = []
webrtc_last_names = []

# Function to save known faces
def save_known_faces():
    try:
        with open(KNOWN_FACES_FILE, 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        return True
    except Exception as e:
        st.error(f"Error saving database: {e}")
        return False

# Function to load known faces
def load_known_faces():
    global known_face_encodings, known_face_names
    try:
        if os.path.exists(KNOWN_FACES_FILE) and os.path.getsize(KNOWN_FACES_FILE) > 0:
            with open(KNOWN_FACES_FILE, 'rb') as f:
                known_face_encodings, known_face_names = pickle.load(f)
        else:
            known_face_encodings = []
            known_face_names = []
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
        known_face_encodings = []
        known_face_names = []
        st.warning("Database empty or initialized for the first time.")

# Helper to convert input image format to safe RGB numpy array (uint8)
def to_rgb_numpy(image):
    if hasattr(image, 'convert'):
        try:
            # Correct orientation from mobile cameras using EXIF data
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        # Convert to RGB mode
        image = image.convert('RGB')
    
    img_np = np.array(image, dtype=np.uint8)
    
    if img_np.ndim == 2:
        # Grayscale to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.ndim == 3:
        if img_np.shape[2] == 4:
            # RGBA to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif img_np.shape[2] == 1:
            # 1-channel to RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
    return img_np

# Helper function to add a new face
def add_new_face(image, name):
    try:
        image_np = to_rgb_numpy(image)
        # Check shape and dtype to ensure it is valid
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            st.error(f"Invalid image format: shape is {image_np.shape}")
            return False
            
        # Get locations and encodings
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            # Fallback to upsampling to find smaller/blurry faces
            face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2)
            
        encodings = face_recognition.face_encodings(image_np, face_locations)
        
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error processing image for registration: {e}")
        return False

# Helper function to recognize faces in an image (Upload Mode)
def recognize_faces(image_np, tolerance=0.6):
    try:
        image_np = to_rgb_numpy(image_np)
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            # Fallback to upsampling to find smaller/blurry faces
            face_locations = face_recognition.face_locations(image_np, number_of_times_to_upsample=2)
            
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        # Make a copy to draw on
        annotated_image = image_np.copy()
        face_names = []
        
        for face_encoding in face_encodings:
            name = "Face Not Registered"
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if face_distances.size > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw rectangles around faces and display names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw elegant bounding box
            cv2.rectangle(annotated_image, (left, top), (right, bottom), (255, 75, 75), 2)
            cv2.rectangle(annotated_image, (left, bottom - 30), (right, bottom), (255, 75, 75), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(annotated_image, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        return annotated_image, len(face_locations)
    except Exception as e:
        st.error(f"Error performing face recognition: {e}")
        return image_np, 0

# Load known faces at startup
load_known_faces()

# Sidebar for database status and registration
st.sidebar.markdown("### 👤 Face Database Manager")

# Display registered count
st.sidebar.markdown(f"Registered Faces: **{len(known_face_names)}**")

# Tolerance Slider
tolerance = st.sidebar.slider(
    "Recognition Sensitivity (Tolerance)", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.6, 
    step=0.05, 
    help="Lower values are stricter (fewer false matches). Higher values are more lenient (easier to match)."
)

# Export tolerance to global scope for WebRTC thread to read
global_tolerance = tolerance

if len(known_face_names) > 0:
    with st.sidebar.expander("Show Registered Names"):
        # Unique list of registered names
        unique_names = list(set(known_face_names))
        for un in sorted(unique_names):
            st.write(f"- {un}")

# Add New Face Form
st.sidebar.markdown("---")
st.sidebar.markdown("#### Register New Face")
uploaded_file = st.sidebar.file_uploader("Upload portrait photo", type=["jpg", "jpeg", "png"])
new_name = st.sidebar.text_input("Person's Name")

if st.sidebar.button("Register Face", key="add_face_button", use_container_width=True):
    if uploaded_file and new_name.strip():
        # Open image and convert to RGB format
        image = Image.open(uploaded_file).convert("RGB")
        # Process and register
        success = add_new_face(image, new_name.strip())
        if success:
            save_success = save_known_faces()
            if save_success:
                st.sidebar.success(f"Successfully registered {new_name.strip()}!")
                st.rerun()
            else:
                st.sidebar.error("Face extracted but failed to write database file.")
        else:
            st.sidebar.error("Could not locate any clear face in the uploaded image. Please try another photo.")
    else:
        st.sidebar.warning("Please provide both an image and a valid name.")

# Option to reset database
if len(known_face_names) > 0:
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Database", key="reset_db_button", use_container_width=True, type="secondary"):
        known_face_encodings = []
        known_face_names = []
        save_known_faces()
        st.sidebar.info("Database reset successfully.")
        st.rerun()

# Main Application Layout
st.markdown('<div class="main-title">Face Recognition Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced real-time face detection & identification system optimized for local and cloud runtimes.</div>', unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2 = st.tabs(["📸 Real-Time Camera Stream", "📤 Upload Image for Detection"])

with tab1:
    st.markdown("### 📸 Real-Time Video Face Detection")
    st.markdown("The stream runs **directly in the browser** and automatically displays names for registered faces, or **Face Not Registered** for unknown individuals, without requiring snapshots.")
    
    # RTC Configuration using public Google STUN server for Streamlit Cloud NAT traversal
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # WebRTC Video frame processor callback function
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        global webrtc_frame_count, webrtc_last_locations, webrtc_last_names
        webrtc_frame_count += 1
        
        img = frame.to_ndarray(format="bgr24")
        
        # Only run face recognition on every 3rd frame to optimize performance
        if webrtc_frame_count % 3 == 0 or webrtc_frame_count == 1:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Downsize image to 1/4 size for 16x faster processing
            small_img = cv2.resize(rgb_img, (0, 0), fx=0.25, fy=0.25)
            
            # Detect face locations on the smaller image
            face_locations = face_recognition.face_locations(small_img)
            if not face_locations:
                # Fallback to upsampling to find smaller faces
                face_locations = face_recognition.face_locations(small_img, number_of_times_to_upsample=2)
                
            face_encodings = face_recognition.face_encodings(small_img, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Face Not Registered"
                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=global_tolerance)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                face_names.append(name)
                
            # Scale coordinates back up and store in cache
            scaled_locations = []
            for (top, right, bottom, left) in face_locations:
                scaled_locations.append((top * 4, right * 4, bottom * 4, left * 4))
                
            webrtc_last_locations = scaled_locations
            webrtc_last_names = face_names

        # Draw bounding boxes (from cache or new detection) on the current frame
        for (top, right, bottom, left), name in zip(webrtc_last_locations, webrtc_last_names):
            # Draw elegant red bounding box (75, 75, 255 in BGR is red)
            cv2.rectangle(img, (left, top), (right, bottom), (75, 75, 255), 2)
            cv2.rectangle(img, (left, bottom - 30), (right, bottom), (75, 75, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Start the WebRTC stream
    webrtc_streamer(
        key="realtime-face-recognition",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with tab2:
    st.markdown("### 📤 Upload Image")
    st.markdown("Upload any photo containing faces to run detection and name identification.")
    
    uploaded_image_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"], key="recognize_uploader")
    if uploaded_image_file is not None:
        image = Image.open(uploaded_image_file).convert("RGB")
        image_np = np.array(image)
        
        with st.spinner("Analyzing uploaded photo..."):
            annotated_image, count = recognize_faces(image_np, tolerance=tolerance)
            
        st.success(f"Analysis complete: Detected {count} face(s).")
        st.image(annotated_image, caption="Processed Image", use_container_width=True)
