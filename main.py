#!/usr/bin/env python3
"""
Streamlit app to display AWS Kinesis Video Stream with Object Detection
"""

import os
os.environ["STREAMLIT_WATCH_FOR_CHANGES"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import boto3
from datetime import datetime
from dotenv import load_dotenv
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-southeast-2')
STREAM_NAME = os.getenv('STREAM_NAME', 'my-stream1')


@st.cache_resource
def load_yolo_model():
    """Load YOLO model (will download automatically if not present)"""
    try:
        # Using YOLOv8n (nano) for faster inference
        # Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        model = YOLO('yolov8n.pt')  # This will auto-download on first run
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None


def get_hls_streaming_url(stream_name, playback_mode='LIVE'):
    """
    Get HLS streaming URL from Kinesis Video Streams
    
    Args:
        stream_name: Name of the Kinesis video stream
        playback_mode: 'LIVE' or 'ON_DEMAND'
    
    Returns:
        HLS streaming URL
    """
    try:
        # Create Kinesis Video client
        kvs_client = boto3.client(
            'kinesisvideo',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Get the endpoint for the stream
        response = kvs_client.get_data_endpoint(
            StreamName=stream_name,
            APIName='GET_HLS_STREAMING_SESSION_URL'
        )
        
        endpoint = response['DataEndpoint']
        
        # Create Kinesis Video Archived Media client
        kvam_client = boto3.client(
            'kinesis-video-archived-media',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
            endpoint_url=endpoint
        )
        
        # Get HLS streaming session URL
        hls_response = kvam_client.get_hls_streaming_session_url(
            StreamName=stream_name,
            PlaybackMode=playback_mode,
            HLSFragmentSelector={
                'FragmentSelectorType': 'SERVER_TIMESTAMP' if playback_mode == 'LIVE' else 'PRODUCER_TIMESTAMP'
            },
            ContainerFormat='FRAGMENTED_MP4',
            DiscontinuityMode='ALWAYS',
            DisplayFragmentTimestamp='ALWAYS',
            Expires=43200  # URL expires in 12 hours
        )
        
        return hls_response['HLSStreamingSessionURL']
    
    except Exception as e:
        st.error(f"Error getting HLS URL: {str(e)}")
        return None


def process_video_stream(hls_url, model, frame_placeholder, stats_placeholder, stop_flag, target_objects, conf_threshold):
    """
    Process video stream with YOLO object detection
    
    Args:
        hls_url: HLS streaming URL
        model: YOLO model
        frame_placeholder: Streamlit placeholder for video frames
        stats_placeholder: Streamlit placeholder for statistics
        stop_flag: List containing boolean to stop processing
        target_objects: List of object classes to detect
        conf_threshold: Confidence threshold for detection
    """
    cap = cv2.VideoCapture(hls_url)
    
    if not cap.isOpened():
        st.error("Failed to open video stream")
        return
    
    frame_count = 0
    detection_counts = {obj: 0 for obj in target_objects}
    fps_time = time.time()
    fps = 0
    
    # Color map for different objects
    color_map = {
        'laptop': (255, 165, 0),  # Orange
        'book': (0, 255, 255),    # Cyan
        'knife': (0, 0, 255),     # Red
        'scissors': (255, 0, 255) # Magenta
    }
    
    while cap.isOpened() and not stop_flag[0]:
        ret, frame = cap.read()
        
        if not ret:
            st.warning("Stream ended or connection lost")
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=conf_threshold)
        
        # Track detected classes in this frame
        detected_classes_in_frame = set()
        
        # Draw bounding boxes and labels
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Get class name
                class_name = model.names[cls]
                
                # Only process target objects
                if class_name in target_objects:
                    detection_counts[class_name] += 1
                    detected_classes_in_frame.add(class_name)
                    
                    # Get color for this object
                    color = color_map.get(class_name, (0, 255, 0))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label = f"{class_name}: {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - label_height - 10), 
                        (x1 + label_width, y1), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
        
        # Calculate FPS
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            fps = frame_count / (current_time - fps_time)
            frame_count = 0
            fps_time = current_time
        
        # Add FPS counter to frame
        cv2.putText(
            frame, 
            f"FPS: {fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update statistics (cumulative)
        stats_html = ""
        for obj, count in detection_counts.items():
            stats_html += f"**{obj.title()}:** {count}  \n"
        stats_placeholder.markdown(stats_html)

        # Show detected classes for this frame
        if detected_classes_in_frame:
            st.markdown(
                "**Detected in this frame:** " +
                ", ".join(sorted(detected_classes_in_frame))
            )
        else:
            st.markdown("**Detected in this frame:** None")
    
    cap.release()


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Object Detection - Kinesis Stream",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Object Detection System")
    st.markdown("Real-time object detection from AWS Kinesis Video Stream using YOLOv8")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info(
            f"**Stream Info:**  \n"
            f"- **Region:** `{AWS_REGION}`  \n"
            f"- **Stream:** `{STREAM_NAME}`  \n"
            f"- **Model:** YOLOv8n (COCO dataset)"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Settings")
        
        # Object selection
        available_objects = ['laptop', 'book', 'knife', 'scissors', 'cell phone', 'mouse', 'keyboard']
        target_objects = st.multiselect(
            "Select objects to detect:",
            available_objects,
            default=['laptop', 'book', 'knife']
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
        
        st.markdown("---")
        st.markdown("### 📊 Detection Stats")
        stats_placeholder = st.empty()
        
        st.markdown("---")
        if st.button("🔄 Refresh Stream", type="primary", use_container_width=True):
            st.rerun()
    
    # Load YOLO model
    with st.spinner("Loading YOLOv8 model (downloading if needed)..."):
        model = load_yolo_model()
    
    if model is None:
        st.error("❌ Failed to load YOLO model.")
        return
    
    st.success("✅ YOLOv8 model loaded successfully!")
    
    # Show available classes
    with st.expander("📋 Available Object Classes (COCO dataset)"):
        classes = list(model.names.values())
        st.write(", ".join(classes))
    
    # Get HLS URL
    with st.spinner("Connecting to video stream..."):
        hls_url = get_hls_streaming_url(STREAM_NAME, 'LIVE')
    
    if not hls_url:
        st.error("❌ Failed to get streaming URL. Please check your AWS configuration.")
        return
    
    st.success("✅ Stream connected successfully!")
    
    # Create placeholders
    frame_placeholder = st.empty()
    
    # Control buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        start_button = st.button("▶️ Start Detection", type="primary")
    with col2:
        stop_button = st.button("⏹️ Stop Detection")
    
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if start_button:
        if not target_objects:
            st.warning("⚠️ Please select at least one object to detect!")
        else:
            st.session_state.processing = True
            stop_flag = [False]
            
            # Start processing
            with st.spinner("Processing video stream..."):
                try:
                    process_video_stream(
                        hls_url, 
                        model, 
                        frame_placeholder, 
                        stats_placeholder, 
                        stop_flag,
                        target_objects,
                        conf_threshold
                    )
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
            
            st.session_state.processing = False
    
    if stop_button:
        st.session_state.processing = False
        st.info("Stream stopped by user")
    
    # Show HLS URL in expander
    with st.expander("🔗 HLS Stream URL (for debugging)"):
        st.code(hls_url, language="text")
        st.caption("⚠️ This URL expires after 12 hours")


if __name__ == "__main__":
    main()  
    
    
    
# #!/usr/bin/env python3
# """
# Streamlit app to display AWS Kinesis Video Stream with Fall Detection
# """

# import os
# os.environ["STREAMLIT_WATCH_FOR_CHANGES"] = "false"
# os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# import streamlit as st
# import boto3
# from datetime import datetime
# from dotenv import load_dotenv
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time
# from threading import Thread
# import queue

# # Load environment variables
# load_dotenv()

# # AWS Configuration
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-southeast-2')
# STREAM_NAME = os.getenv('STREAM_NAME', 'my-stream1')

# # Model path
# MODEL_PATH = "fall_model/best.pt"


# @st.cache_resource
# def load_yolo_model():
#     """Load YOLO model (cached)"""
#     try:
#         model = YOLO(MODEL_PATH)
#         return model
#     except Exception as e:
#         st.error(f"Error loading YOLO model: {str(e)}")
#         return None


# def get_hls_streaming_url(stream_name, playback_mode='LIVE'):
#     """
#     Get HLS streaming URL from Kinesis Video Streams
    
#     Args:
#         stream_name: Name of the Kinesis video stream
#         playback_mode: 'LIVE' or 'ON_DEMAND'
    
#     Returns:
#         HLS streaming URL
#     """
#     try:
#         # Create Kinesis Video client
#         kvs_client = boto3.client(
#             'kinesisvideo',
#             aws_access_key_id=AWS_ACCESS_KEY_ID,
#             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#             region_name=AWS_REGION
#         )
        
#         # Get the endpoint for the stream
#         response = kvs_client.get_data_endpoint(
#             StreamName=stream_name,
#             APIName='GET_HLS_STREAMING_SESSION_URL'
#         )
        
#         endpoint = response['DataEndpoint']
        
#         # Create Kinesis Video Archived Media client
#         kvam_client = boto3.client(
#             'kinesis-video-archived-media',
#             aws_access_key_id=AWS_ACCESS_KEY_ID,
#             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#             region_name=AWS_REGION,
#             endpoint_url=endpoint
#         )
        
#         # Get HLS streaming session URL
#         hls_response = kvam_client.get_hls_streaming_session_url(
#             StreamName=stream_name,
#             PlaybackMode=playback_mode,
#             HLSFragmentSelector={
#                 'FragmentSelectorType': 'SERVER_TIMESTAMP' if playback_mode == 'LIVE' else 'PRODUCER_TIMESTAMP'
#             },
#             ContainerFormat='FRAGMENTED_MP4',
#             DiscontinuityMode='ALWAYS',
#             DisplayFragmentTimestamp='ALWAYS',
#             Expires=43200  # URL expires in 12 hours
#         )
        
#         return hls_response['HLSStreamingSessionURL']
    
#     except Exception as e:
#         st.error(f"Error getting HLS URL: {str(e)}")
#         return None


# def process_video_stream(hls_url, model, frame_placeholder, stats_placeholder, stop_flag):
#     """
#     Process video stream with YOLO fall detection
    
#     Args:
#         hls_url: HLS streaming URL
#         model: YOLO model
#         frame_placeholder: Streamlit placeholder for video frames
#         stats_placeholder: Streamlit placeholder for statistics
#         stop_flag: List containing boolean to stop processing
#     """
#     cap = cv2.VideoCapture(hls_url)
    
#     if not cap.isOpened():
#         st.error("Failed to open video stream")
#         return
    
#     frame_count = 0
#     fall_count = 0
#     fps_time = time.time()
#     fps = 0
    
#     while cap.isOpened() and not stop_flag[0]:
#         ret, frame = cap.read()
        
#         if not ret:
#             st.warning("Stream ended or connection lost")
#             break
        
#         frame_count += 1
        
#         # Run YOLO detection every frame (adjust skip rate if needed)
#         if frame_count % 1 == 0:  # Process every frame
#             results = model(frame, conf=0.5)
            
#             # Draw bounding boxes and labels
#             for result in results:
#                 boxes = result.boxes
#                 for box in boxes:
#                     # Get box coordinates
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                     conf = float(box.conf[0])
#                     cls = int(box.cls[0])
                    
#                     # Get class name
#                     class_name = model.names[cls]
                    
#                     # Check if it's a fall detection
#                     if 'fall' in class_name.lower():
#                         fall_count += 1
#                         color = (0, 0, 255)  # Red for fall
#                         thickness = 3
#                     else:
#                         color = (0, 255, 0)  # Green for other detections
#                         thickness = 2
                    
#                     # Draw bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
#                     # Draw label background
#                     label = f"{class_name}: {conf:.2f}"
#                     (label_width, label_height), _ = cv2.getTextSize(
#                         label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
#                     )
#                     cv2.rectangle(
#                         frame, 
#                         (x1, y1 - label_height - 10), 
#                         (x1 + label_width, y1), 
#                         color, 
#                         -1
#                     )
                    
#                     # Draw label text
#                     cv2.putText(
#                         frame, 
#                         label, 
#                         (x1, y1 - 5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.6, 
#                         (255, 255, 255), 
#                         2
#                     )
        
#         # Calculate FPS
#         current_time = time.time()
#         if current_time - fps_time >= 1.0:
#             fps = frame_count / (current_time - fps_time)
#             frame_count = 0
#             fps_time = current_time
        
#         # Add FPS counter to frame
#         cv2.putText(
#             frame, 
#             f"FPS: {fps:.1f}", 
#             (10, 30), 
#             cv2.FONT_HERSHEY_SIMPLEX, 
#             1, 
#             (0, 255, 0), 
#             2
#         )
        
#         # Convert BGR to RGB for Streamlit
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Display frame
#         frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
#         # Update statistics
#         stats_placeholder.metric("Falls Detected", fall_count)
    
#     cap.release()


# def main():
#     """Main Streamlit application"""
    
#     st.set_page_config(
#         page_title="Fall Detection - Kinesis Stream",
#         page_icon="🚨",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     st.title("🚨 Fall Detection System")
#     st.markdown("Real-time fall detection from AWS Kinesis Video Stream using YOLO")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
#         st.info(
#             f"**Stream Info:**  \n"
#             f"- **Region:** `{AWS_REGION}`  \n"
#             f"- **Stream:** `{STREAM_NAME}`  \n"
#             f"- **Model:** `{MODEL_PATH}`"
#         )
        
#         if st.button("🔄 Refresh Stream", type="primary", use_container_width=True):
#             st.rerun()
        
#         st.markdown("---")
#         st.markdown("### 📊 Detection Stats")
#         stats_placeholder = st.empty()
    
#     # Load YOLO model
#     with st.spinner("Loading YOLO model..."):
#         model = load_yolo_model()
    
#     if model is None:
#         st.error("❌ Failed to load YOLO model. Please check the model path.")
#         return
    
#     st.success("✅ YOLO model loaded successfully!")
    
#     # Get HLS URL
#     with st.spinner("Connecting to video stream..."):
#         hls_url = get_hls_streaming_url(STREAM_NAME, 'LIVE')
    
#     if not hls_url:
#         st.error("❌ Failed to get streaming URL. Please check your AWS configuration.")
#         return
    
#     st.success("✅ Stream connected successfully!")
    
#     # Create placeholders
#     frame_placeholder = st.empty()
    
#     # Control buttons
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         start_button = st.button("▶️ Start Detection", type="primary")
#     with col2:
#         stop_button = st.button("⏹️ Stop Detection")
    
#     # Initialize session state
#     if 'processing' not in st.session_state:
#         st.session_state.processing = False
    
#     if start_button:
#         st.session_state.processing = True
#         stop_flag = [False]
        
#         # Start processing in main thread (Streamlit limitation)
#         with st.spinner("Processing video stream..."):
#             try:
#                 process_video_stream(hls_url, model, frame_placeholder, stats_placeholder, stop_flag)
#             except Exception as e:
#                 st.error(f"Error during processing: {str(e)}")
        
#         st.session_state.processing = False
    
#     if stop_button:
#         st.session_state.processing = False
#         st.info("Stream stopped by user")
    
#     # Show HLS URL in expander
#     with st.expander("🔗 HLS Stream URL (for debugging)"):
#         st.code(hls_url, language="text")
#         st.caption("⚠️ This URL expires after 12 hours")


# if __name__ == "__main__":
#     main()