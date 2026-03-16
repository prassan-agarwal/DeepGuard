import os
import sys
import streamlit as st
import tempfile
import time
import torch
from pathlib import Path

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from inference.detect_single_video import process_single_video

def main():
    st.set_page_config(
        page_title="Deepfake Video Detector",
        page_icon="🕵️",
        layout="centered"
    )

    st.title("🕵️ Deepfake Video Detector")
    st.markdown("Upload a video to analyze if it's real or AI-generated.")

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("🔍 Analyze Video", type="primary"):
            with st.spinner("Analyzing video... This may take a moment."):
                # Save the uploaded file to a temporary location
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name
                tfile.close()

                try:
                    # Capture stdout to get the prediction from process_single_video
                    # Alternatively, we could modify process_single_video to return the result, 
                    # but capturing stdout works without modifying the original file
                    
                    from runpy import run_path
                    import contextlib
                    import io

                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        process_single_video(temp_video_path)
                    
                    output = f.getvalue()
                    
                    # Parse output
                    if "RESULT: FAKE" in output:
                        st.error("🚨 THIS VIDEO APPEARS TO BE A DEEPFAKE 🚨")
                        st.markdown("### Analysis Details:")
                        
                        # Extract probability and confidence
                        for line in output.split('\n'):
                            if "Probability of being Fake:" in line:
                                st.write(f"**{line.strip()}**")
                            if "Confidence:" in line:
                                st.write(f"**{line.strip()}**")
                                
                    elif "RESULT: REAL" in output:
                        st.success("✅ THIS VIDEO APPEARS TO BE REAL")
                        st.markdown("### Analysis Details:")
                        
                        # Extract probability and confidence
                        for line in output.split('\n'):
                            if "Probability of being Fake:" in line:
                                st.write(f"**{line.strip()}**")
                            if "Confidence:" in line:
                                st.write(f"**{line.strip()}**")
                    else:
                        st.warning("⚠️ Could not determine result. Check the logs below.")
                        st.text_area("Analysis Logs", output, height=200)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)

if __name__ == "__main__":
    main()
