#!/usr/bin/env python3
"""
Streamlit app to display AWS Kinesis Video Stream
"""

# Install dependencies:
# - Using requirements file: pip install -r requirements.txt
# - Or single command: pip install streamlit boto3 python-dotenv
# On some systems use: pip3 install --user streamlit boto3 python-dotenv

import streamlit as st
import boto3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'ap-southeast-2')
STREAM_NAME = os.getenv('STREAM_NAME', 'my-stream1')


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


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Kinesis Video Stream Viewer",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎥 AWS Kinesis Video Stream Viewer")
    st.markdown("Live video stream from your RTSP camera via AWS Kinesis Video Streams")
    
    # Only play the configured stream, no user input
    stream_name = STREAM_NAME
    playback_mode = 'LIVE'
    video_width, video_height = 1280, 720  # Fixed large size

    # --- Stream Info Panel ---
    st.info(
        f"**Stream Info:**  \n"
        f"- **Region:** `{AWS_REGION}`  \n"
        f"- **Stream Name:** `{stream_name}`  \n"
        f"- **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Refresh button
    if st.button("🔄 Refresh Stream", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Main content area: just the video
    with st.spinner("Loading video stream..."):
        hls_url = get_hls_streaming_url(stream_name, playback_mode)
    
    if hls_url:
        st.success("✅ Stream connected successfully!")
        video_html = f"""
        <link href="https://vjs.zencdn.net/8.10.0/video-js.css" rel="stylesheet" />
        <script src="https://vjs.zencdn.net/8.10.0/video.min.js"></script>
        <video
            id="kinesis-video"
            class="video-js vjs-default-skin vjs-big-play-centered"
            controls
            autoplay
            muted
            preload="auto"
            width="{video_width}"
            height="{video_height}"
            data-setup='{{}}'
            style="max-width: 100%; border-radius: 12px; box-shadow: 0 2px 16px #0003;"
        >
            <source src="{hls_url}" type="application/x-mpegURL" />
            <p class="vjs-no-js">
                To view this video please enable JavaScript, and consider upgrading to a
                web browser that supports HTML5 video
            </p>
        </video>
        <script>
            var player = videojs('kinesis-video');
            player.play();
        </script>
        """
        st.components.v1.html(video_html, height=video_height + 50)
        # Show HLS URL in expander (for debugging)
        with st.expander("🔗 HLS Stream URL (for debugging)"):
            st.code(hls_url, language="text")
            st.caption("⚠ This URL expires after 12 hours")
    else:
        st.error("❌ Failed to load video stream. Please check your configuration.")


if __name__ == "__main__":
    main()