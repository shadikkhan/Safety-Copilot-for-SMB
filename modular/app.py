"""
Safety Copilot - Streamlit Web Application

Run with: streamlit run app.py
"""
import streamlit as st
import cv2
import os
import sys
import tempfile
import time
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modular.models import ModelManager, get_device
from modular.risk_engine import RiskEngine
from modular.visualization import (
    plot_risk_over_time, 
    plot_violations_summary,
    bgr_to_rgb
)
from modular.detectors import PPEDetector
from modular.llm_reporter import LLMReporter, generate_quick_summary


# Page configuration
st.set_page_config(
    page_title="Safety Copilot",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-low { color: #28a745; }
    .risk-medium { color: #ffc107; }
    .risk-high { color: #dc3545; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load models with caching."""
    with st.spinner("Loading AI models... This may take a moment."):
        manager = ModelManager()
        manager.load_all()
        return manager


def get_risk_color(score: int) -> str:
    """Get color class based on risk score."""
    if score < 30:
        return "risk-low"
    elif score < 60:
        return "risk-medium"
    return "risk-high"


def main():
    # Header
    st.markdown('<p class="main-header">🦺 Safety Copilot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Workplace Safety Monitoring for SMBs</p>',
        unsafe_allow_html=True
    )
    
    # Device info
    device = get_device()
    st.sidebar.markdown(f"**Device:** {'🚀 GPU' if device != 'cpu' else '💻 CPU'}")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode",
        ["🏠 Home", "📷 Image Analysis", "🎬 Video Analysis", "📹 Live Camera", "📊 Reports"]
    )
    
    # Load models
    try:
        model_manager = load_models()
        st.sidebar.success("✅ Models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        st.error("Failed to load AI models. Please check that model files exist.")
        return
    
    # Display Models Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Active Models")
    with st.sidebar.expander("View Models", expanded=False):
        st.markdown("**Base Detection:**")
        st.code("YOLOv8n (yolov8n.pt)", language=None)
        st.markdown("**PPE Detection:**")
        st.code("Custom PPE Model (ppe_yolov8n_best.pt)", language=None)
        st.markdown("**Pose Estimation:**")
        st.code("YOLOv8n-Pose (yolov8n-pose.pt)", language=None)
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    enable_pose = st.sidebar.checkbox("Enable Pose Analysis", value=True)
    show_heatmap = st.sidebar.checkbox("Show Risk Heatmap", value=True)
    
    # Route to pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📷 Image Analysis":
        show_image_analysis(model_manager, confidence)
    elif page == "🎬 Video Analysis":
        show_video_analysis(model_manager, confidence, enable_pose)
    elif page == "📹 Live Camera":
        show_live_camera(model_manager, confidence, enable_pose)
    elif page == "📊 Reports":
        show_reports_page()


def show_home_page():
    """Display home page with overview."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Detection Capabilities")
        st.markdown("""
        - **PPE Violations**: Helmet & vest detection
        - **Running Detection**: Identifies unsafe running
        - **Forklift Proximity**: Warns of danger zones
        - **Bad Lifting Posture**: Ergonomic analysis
        - **Slip Risk**: Motion instability detection
        """)
    
    with col2:
        st.markdown("### 📈 Real-time Analytics")
        st.markdown("""
        - Live risk scoring
        - Cumulative violation tracking
        - Risk heatmap visualization
        - Frame-by-frame analysis
        - Exportable reports
        """)
    
    with col3:
        st.markdown("### 🎯 Use Cases")
        st.markdown("""
        - Construction sites
        - Warehouses & logistics
        - Manufacturing floors
        - Industrial facilities
        - Safety compliance audits
        """)
    
    st.markdown("---")
    st.info("👈 Select a mode from the sidebar to begin analysis")


def show_image_analysis(model_manager, confidence):
    """Image analysis page for PPE detection."""
    st.header("📷 Image Analysis - PPE Detection")
    
    # Initialize session state for image analysis
    if 'image_analysis_report' not in st.session_state:
        st.session_state.image_analysis_report = None
    if 'image_llm_report' not in st.session_state:
        st.session_state.image_llm_report = None
    
    uploaded_files = st.file_uploader(
        "Upload construction/warehouse images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Clear previous reports when new files uploaded
        if st.button("🔄 Analyze Images", type="primary"):
            st.session_state.image_analysis_report = None
            st.session_state.image_llm_report = None
            
            ppe_detector = PPEDetector(model_manager.ppe_model, confidence)
            
            total_violations = 0
            total_persons = 0
            total_helmets = 0
            total_vests = 0
            all_violation_counts = {}
            image_results = []
            
            for uploaded_file in uploaded_files:
                st.markdown(f"### 📁 {uploaded_file.name}")
                
                # Read image
                uploaded_file.seek(0)  # Reset file pointer
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect
                with st.spinner("Analyzing..."):
                    result = ppe_detector.detect(image)
                    annotated = ppe_detector.draw_violations(image, result)
                
                # Display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original**")
                    st.image(bgr_to_rgb(image), use_container_width=True)
                
                with col2:
                    st.markdown("**Analysis Result**")
                    st.image(bgr_to_rgb(annotated), use_container_width=True)
                
                # Stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Persons", len(result.persons))
                col2.metric("Helmets", len(result.helmets))
                col3.metric("Vests", len(result.vests))
                col4.metric("Violations", len(result.violations), 
                           delta=None if not result.violations else "⚠️",
                           delta_color="inverse")
                
                # Aggregate stats
                total_persons += len(result.persons)
                total_helmets += len(result.helmets)
                total_vests += len(result.vests)
                total_violations += len(result.violations)
                
                # Count violation types
                for v in result.violations:
                    all_violation_counts[v.violation_type] = all_violation_counts.get(v.violation_type, 0) + 1
                
                # Store result for report
                image_results.append({
                    'filename': uploaded_file.name,
                    'persons': len(result.persons),
                    'helmets': len(result.helmets),
                    'vests': len(result.vests),
                    'violations': [v.violation_type for v in result.violations]
                })
                
                if result.violations:
                    st.error(f"⚠️ Violations: {', '.join(v.violation_type for v in result.violations)}")
                else:
                    st.success("✅ All safety requirements met!")
                
                st.markdown("---")
            
            # Save report to session state
            st.session_state.image_analysis_report = {
                'total_images': len(uploaded_files),
                'total_persons': total_persons,
                'total_helmets': total_helmets,
                'total_vests': total_vests,
                'total_violations': total_violations,
                'violation_counts': all_violation_counts,
                'image_results': image_results,
                'final_risk_score': min(total_violations * 10, 100)  # Simple risk score
            }
    
    # Display report if available
    if st.session_state.image_analysis_report:
        report = st.session_state.image_analysis_report
        
        st.markdown("---")
        st.markdown("### 📊 Image Analysis Report")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images Analyzed", report['total_images'])
        col2.metric("Total Persons", report['total_persons'])
        col3.metric("Total Violations", report['total_violations'])
        col4.metric("Risk Score", report['final_risk_score'])
        
        # Violations summary
        if report['violation_counts']:
            st.markdown("### Violations Detected")
            fig = plot_violations_summary(report['violation_counts'])
            st.pyplot(fig)
            
            # Breakdown
            st.markdown("**Violation Breakdown:**")
            for v_type, count in report['violation_counts'].items():
                st.write(f"- **{v_type}**: {count}")
        else:
            st.success("✅ No violations detected across all images!")
        
        # AI-Generated Safety Report
        st.markdown("---")
        st.markdown("### 🤖 AI Safety Report")
        
        # Quick summary
        quick_summary = generate_quick_summary(
            report['violation_counts'], 
            report['final_risk_score']
        )
        st.markdown(quick_summary)
        
        # LLM detailed report
        if report['violation_counts']:
            col1, col2 = st.columns([2, 1])
            with col1:
                generate_btn = st.button("📝 Generate Detailed AI Report", key="image_llm_btn")
            with col2:
                if st.button("🗑️ Clear Results", key="clear_image_report"):
                    st.session_state.image_analysis_report = None
                    st.session_state.image_llm_report = None
                    st.rerun()
            
            if generate_btn:
                llm = LLMReporter()
                
                if not llm.is_available():
                    st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
                else:
                    with st.spinner("🧠 Generating AI safety report... This may take a minute."):
                        llm_report = llm.generate_report(
                            violation_counts=report['violation_counts'],
                            total_frames=report['total_images'],
                            duration_seconds=0,  # N/A for images
                            final_risk_score=report['final_risk_score'],
                            context="construction/industrial site (image analysis)"
                        )
                    st.session_state.image_llm_report = llm_report
            
            # Display LLM report if available
            if st.session_state.image_llm_report:
                llm_report = st.session_state.image_llm_report
                
                st.markdown("#### 📋 Executive Summary")
                st.info(llm_report.summary)
                
                st.markdown("#### ⚠️ Risk Assessment")
                st.warning(llm_report.risk_assessment)
                
                st.markdown("#### 💡 Recommendations")
                for rec in llm_report.recommendations:
                    st.write(f"• {rec}")
                
                st.markdown("#### 💰 Business Impact")
                st.write(llm_report.business_impact)
                
                st.markdown("#### ✅ Action Items")
                for i, item in enumerate(llm_report.action_items, 1):
                    st.write(f"{i}. {item}")
                
                with st.expander("📄 View Full Report"):
                    st.markdown(llm_report.raw_response)


def show_video_analysis(model_manager, confidence, enable_pose):
    """Video analysis page."""
    st.header("🎬 Video Analysis")
    
    # Initialize session state for video analysis
    if 'video_analysis_report' not in st.session_state:
        st.session_state.video_analysis_report = None
    if 'video_llm_report' not in st.session_state:
        st.session_state.video_llm_report = None
    
    uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
    
    col1, col2 = st.columns([3, 1])
    with col2:
        max_frames = st.number_input("Max frames (0=all)", min_value=0, value=300)
        process_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    if uploaded_video and process_btn:
        # Clear previous reports
        st.session_state.video_analysis_report = None
        st.session_state.video_llm_report = None
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
        
        try:
            # Initialize engine
            pose_model = model_manager.pose_model if enable_pose else None
            engine = RiskEngine(
                model_manager.base_model,
                model_manager.ppe_model,
                pose_model,
                confidence
            )
            
            # Video capture
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if max_frames > 0:
                total_frames = min(total_frames, max_frames)
            
            # UI elements
            st.markdown("### Processing...")
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            ret, first_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            engine.reset(first_frame.shape)
            
            risk_log = []
            frame_id = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames > 0 and frame_id >= max_frames:
                    break
                
                result = engine.process_frame(frame, frame_id)
                risk_log.append(result.risk_score)
                
                # Update UI every 5 frames
                if frame_id % 5 == 0:
                    progress_bar.progress(frame_id / total_frames)
                    frame_placeholder.image(
                        bgr_to_rgb(result.annotated_frame),
                        use_container_width=True
                    )
                    
                    with metrics_placeholder.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Frame", f"{frame_id}/{total_frames}")
                        c2.metric("Risk Score", result.risk_score)
                        c3.metric("Persons", result.persons_detected)
                
                frame_id += 1
            
            cap.release()
            progress_bar.progress(1.0)
            
            # Save results to session state
            st.session_state.video_analysis_report = {
                'total_frames': frame_id,
                'final_risk_score': engine.risk_score,
                'duration': frame_id/fps,
                'fps': fps,
                'violation_counts': dict(engine.violation_counts),
                'risk_log': risk_log
            }
            st.success("✅ Analysis complete! See results below.")
            
        finally:
            os.unlink(video_path)
    
    # Display results if available (persists after button clicks)
    if st.session_state.video_analysis_report:
        report_data = st.session_state.video_analysis_report
        
        st.markdown("---")
        st.markdown("### 📊 Video Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", report_data['total_frames'])
        col2.metric("Final Risk Score", report_data['final_risk_score'])
        col3.metric("Duration", f"{report_data['duration']:.1f}s")
        col4.metric("Avg FPS", f"{report_data['fps']:.1f}")
        
        # Violations summary
        st.markdown("### Violations Detected")
        if report_data['violation_counts']:
            fig = plot_violations_summary(report_data['violation_counts'])
            st.pyplot(fig)
            
            # Violation breakdown
            st.markdown("**Violation Breakdown:**")
            for v_type, count in report_data['violation_counts'].items():
                st.write(f"- **{v_type}**: {count}")
        else:
            st.success("✅ No violations detected!")
        
        # Risk over time
        if report_data['risk_log']:
            st.markdown("### Risk Over Time")
            fig = plot_risk_over_time(report_data['risk_log'])
            st.pyplot(fig)
        
        # AI-Generated Safety Report
        st.markdown("---")
        st.markdown("### 🤖 AI Safety Report")
        
        # Quick summary (no LLM)
        quick_summary = generate_quick_summary(
            report_data['violation_counts'], 
            report_data['final_risk_score']
        )
        st.markdown(quick_summary)
        
        # LLM detailed report
        if report_data['violation_counts']:
            col1, col2 = st.columns([2, 1])
            with col1:
                generate_btn = st.button("📝 Generate Detailed AI Report", key="video_llm_btn")
            with col2:
                if st.button("🗑️ Clear Results", key="clear_video_report"):
                    st.session_state.video_analysis_report = None
                    st.session_state.video_llm_report = None
                    st.rerun()
            
            if generate_btn:
                llm = LLMReporter()
                
                if not llm.is_available():
                    st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
                else:
                    with st.spinner("🧠 Generating AI safety report... This may take a minute."):
                        llm_report = llm.generate_report(
                            violation_counts=report_data['violation_counts'],
                            total_frames=report_data['total_frames'],
                            duration_seconds=report_data['duration'],
                            final_risk_score=report_data['final_risk_score'],
                            context="construction/industrial site"
                        )
                    # Store in session state
                    st.session_state.video_llm_report = llm_report
            
            # Display LLM report if available
            if st.session_state.video_llm_report:
                llm_report = st.session_state.video_llm_report
                
                st.markdown("#### 📋 Executive Summary")
                st.info(llm_report.summary)
                
                st.markdown("#### ⚠️ Risk Assessment")
                st.warning(llm_report.risk_assessment)
                
                st.markdown("#### 💡 Recommendations")
                for rec in llm_report.recommendations:
                    st.write(f"• {rec}")
                
                st.markdown("#### 💰 Business Impact")
                st.write(llm_report.business_impact)
                
                st.markdown("#### ✅ Action Items")
                for i, item in enumerate(llm_report.action_items, 1):
                    st.write(f"{i}. {item}")
                
                with st.expander("📄 View Full Report"):
                    st.markdown(llm_report.raw_response)


def show_live_camera(model_manager, confidence, enable_pose):
    """Live camera feed analysis."""
    st.header("📹 Live Camera Feed")
    
    st.warning("⚠️ Live camera requires webcam access and may not work in all browsers.")
    
    camera_index = st.selectbox("Camera", [0, 1, 2], index=0)
    
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶️ Start", type="primary", use_container_width=True)
    stop_btn = col2.button("⏹️ Stop", use_container_width=True)
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'live_cam_report' not in st.session_state:
        st.session_state.live_cam_report = None
    if 'live_llm_report' not in st.session_state:
        st.session_state.live_llm_report = None
    if 'live_session_data' not in st.session_state:
        st.session_state.live_session_data = None
    
    if start_btn:
        st.session_state.camera_running = True
        st.session_state.live_cam_report = None  # Clear previous report
        st.session_state.live_llm_report = None  # Clear previous LLM report
        # Initialize live session tracking
        st.session_state.live_session_data = {
            'start_time': time.time(),
            'frame_count': 0,
            'risk_log': [],
            'violation_counts': {},
            'last_risk_score': 0
        }
    
    if stop_btn and st.session_state.camera_running:
        st.session_state.camera_running = False
        # Generate report from live session data
        if st.session_state.live_session_data:
            session = st.session_state.live_session_data
            duration = time.time() - session['start_time']
            st.session_state.live_cam_report = {
                'total_frames': session['frame_count'],
                'final_risk_score': session['last_risk_score'],
                'duration': duration,
                'violation_counts': session['violation_counts'],
                'risk_log': session['risk_log']
            }
            st.session_state.live_session_data = None
        st.rerun()
    
    if st.session_state.camera_running:
        pose_model = model_manager.pose_model if enable_pose else None
        engine = RiskEngine(
            model_manager.base_model,
            model_manager.ppe_model,
            pose_model,
            confidence
        )
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error("Failed to open camera")
            st.session_state.camera_running = False
            return
        
        ret, first_frame = cap.read()
        if ret:
            engine.reset(first_frame.shape)
        
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        violation_placeholder = st.empty()
        
        frame_id = 0
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = engine.process_frame(frame, frame_id)
            
            # Update session state incrementally (so data persists on stop)
            if st.session_state.live_session_data:
                st.session_state.live_session_data['frame_count'] = frame_id + 1
                st.session_state.live_session_data['risk_log'].append(result.risk_score)
                st.session_state.live_session_data['violation_counts'] = dict(engine.violation_counts)
                st.session_state.live_session_data['last_risk_score'] = engine.risk_score
            
            frame_placeholder.image(
                bgr_to_rgb(result.annotated_frame),
                use_container_width=True
            )
            
            with metrics_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frame", frame_id)
                c2.metric("Risk Score", result.risk_score)
                c3.metric("Persons", result.persons_detected)
                c4.metric("Violations", len(result.violations))
            
            # Show current violations
            if engine.violation_counts:
                with violation_placeholder.container():
                    total_v = sum(engine.violation_counts.values())
                    st.warning(f"⚠️ **{total_v} violation(s) detected**: {', '.join(f'{k}({v})' for k, v in engine.violation_counts.items())}")
            
            frame_id += 1
            time.sleep(0.03)  # ~30 fps
        
        cap.release()
        
        # If loop ended naturally (not via stop button), save report
        if st.session_state.live_session_data:
            session = st.session_state.live_session_data
            duration = time.time() - session['start_time']
            st.session_state.live_cam_report = {
                'total_frames': session['frame_count'],
                'final_risk_score': session['last_risk_score'],
                'duration': duration,
                'violation_counts': session['violation_counts'],
                'risk_log': session['risk_log']
            }
            st.session_state.live_session_data = None
        st.session_state.camera_running = False
    
    # Display report if available
    if st.session_state.live_cam_report:
        report = st.session_state.live_cam_report
        
        st.markdown("---")
        st.markdown("### 📊 Live Camera Analysis Report")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", report['total_frames'])
        col2.metric("Final Risk Score", report['final_risk_score'])
        col3.metric("Duration", f"{report['duration']:.1f}s")
        col4.metric("Total Violations", sum(report['violation_counts'].values()))
        
        # Violations summary
        st.markdown("### Violations Detected")
        if report['violation_counts']:
            fig = plot_violations_summary(report['violation_counts'])
            st.pyplot(fig)
            
            # Show violation breakdown
            st.markdown("**Violation Breakdown:**")
            for violation_type, count in report['violation_counts'].items():
                st.write(f"- **{violation_type}**: {count}")
        else:
            st.success("✅ No violations detected during monitoring!")
        
        # Risk over time
        if report['risk_log']:
            st.markdown("### Risk Score Over Time")
            fig = plot_risk_over_time(report['risk_log'])
            st.pyplot(fig)
        
        # AI-Generated Safety Report
        st.markdown("---")
        st.markdown("### 🤖 AI Safety Report")
        
        # Quick summary
        quick_summary = generate_quick_summary(
            report['violation_counts'], 
            report['final_risk_score']
        )
        st.markdown(quick_summary)
        
        # LLM detailed report
        if report['violation_counts']:
            generate_live_btn = st.button("📝 Generate Detailed AI Report", key="live_llm_btn")
            
            if generate_live_btn:
                llm = LLMReporter()
                
                if not llm.is_available():
                    st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
                else:
                    with st.spinner("🧠 Generating AI safety report... This may take a minute."):
                        llm_report = llm.generate_report(
                            violation_counts=report['violation_counts'],
                            total_frames=report['total_frames'],
                            duration_seconds=report['duration'],
                            final_risk_score=report['final_risk_score'],
                            context="construction/industrial site"
                        )
                    # Store in session state
                    st.session_state.live_llm_report = llm_report
            
            # Display LLM report if available
            if st.session_state.live_llm_report:
                llm_report = st.session_state.live_llm_report
                
                st.markdown("#### 📋 Executive Summary")
                st.info(llm_report.summary)
                
                st.markdown("#### ⚠️ Risk Assessment")
                st.warning(llm_report.risk_assessment)
                
                st.markdown("#### 💡 Recommendations")
                for rec in llm_report.recommendations:
                    st.write(f"• {rec}")
                
                st.markdown("#### 💰 Business Impact")
                st.write(llm_report.business_impact)
                
                st.markdown("#### ✅ Action Items")
                for i, item in enumerate(llm_report.action_items, 1):
                    st.write(f"{i}. {item}")
                
                with st.expander("📄 View Full Report"):
                    st.markdown(llm_report.raw_response)
        
        # Clear report button
        if st.button("🗑️ Clear Report"):
            st.session_state.live_cam_report = None
            st.session_state.live_llm_report = None
            st.rerun()


def show_reports_page():
    """Reports and analytics page with aggregated analysis data."""
    st.header("📊 Reports & Analytics")
    
    # Collect all available reports
    has_video_report = st.session_state.get('video_analysis_report') is not None
    has_live_report = st.session_state.get('live_cam_report') is not None
    has_any_report = has_video_report or has_live_report
    
    if not has_any_report:
        st.info("📭 No analysis data available yet. Run Video or Live Camera analysis first to see reports here.")
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **📷 Image Analysis** - Upload construction site images to detect PPE violations
        2. **🎬 Video Analysis** - Process recorded video files for comprehensive safety analysis
        3. **📹 Live Camera** - Real-time monitoring with your webcam
        
        After running any analysis, return here to view aggregated reports and insights.
        """)
        return
    
    # Dashboard Overview
    st.markdown("### 📈 Analysis Dashboard")
    
    # Aggregate statistics
    total_frames = 0
    total_violations = 0
    all_violation_counts = {}
    max_risk_score = 0
    total_duration = 0
    
    reports = []
    
    if has_video_report:
        vr = st.session_state.video_analysis_report
        reports.append(('Video Analysis', vr))
        total_frames += vr.get('total_frames', 0)
        total_duration += vr.get('duration', 0)
        max_risk_score = max(max_risk_score, vr.get('final_risk_score', 0))
        for v_type, count in vr.get('violation_counts', {}).items():
            all_violation_counts[v_type] = all_violation_counts.get(v_type, 0) + count
            total_violations += count
    
    if has_live_report:
        lr = st.session_state.live_cam_report
        reports.append(('Live Camera', lr))
        total_frames += lr.get('total_frames', 0)
        total_duration += lr.get('duration', 0)
        max_risk_score = max(max_risk_score, lr.get('final_risk_score', 0))
        for v_type, count in lr.get('violation_counts', {}).items():
            all_violation_counts[v_type] = all_violation_counts.get(v_type, 0) + count
            total_violations += count
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📹 Sessions Analyzed", len(reports))
    col2.metric("🖼️ Total Frames", f"{total_frames:,}")
    col3.metric("⏱️ Total Duration", f"{total_duration:.1f}s")
    col4.metric("⚠️ Total Violations", total_violations)
    
    st.markdown("---")
    
    # Violations breakdown
    st.markdown("### 🚨 Violations Summary")
    
    if all_violation_counts:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_violations_summary(all_violation_counts)
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Violation Types:**")
            for v_type, count in sorted(all_violation_counts.items(), key=lambda x: -x[1]):
                severity = "🔴" if v_type in ["NO_HELMET", "FORKLIFT_RISK"] else "🟠" if v_type in ["NO_VEST", "BAD_LIFT"] else "🟡"
                st.write(f"{severity} **{v_type}**: {count}")
            
            st.markdown("---")
            st.markdown("**Risk Level:**")
            if max_risk_score < 30:
                st.success(f"🟢 LOW ({max_risk_score})")
            elif max_risk_score < 60:
                st.warning(f"🟡 MEDIUM ({max_risk_score})")
            elif max_risk_score < 100:
                st.error(f"🟠 HIGH ({max_risk_score})")
            else:
                st.error(f"🔴 CRITICAL ({max_risk_score})")
    else:
        st.success("✅ No violations detected across all analyses!")
    
    st.markdown("---")
    
    # Individual session details
    st.markdown("### 📋 Session Details")
    
    for name, report in reports:
        with st.expander(f"📁 {name} Session", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames", report.get('total_frames', 0))
            col2.metric("Duration", f"{report.get('duration', 0):.1f}s")
            col3.metric("Risk Score", report.get('final_risk_score', 0))
            
            if report.get('violation_counts'):
                st.markdown("**Violations:**")
                for v_type, count in report['violation_counts'].items():
                    st.write(f"- {v_type}: {count}")
            else:
                st.write("No violations in this session.")
            
            # Show risk over time if available
            if report.get('risk_log'):
                st.markdown("**Risk Trend:**")
                fig = plot_risk_over_time(report['risk_log'])
                st.pyplot(fig)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📥 Export Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate summary text
        summary_text = f"""SAFETY COPILOT - ANALYSIS REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

SUMMARY STATISTICS
- Sessions Analyzed: {len(reports)}
- Total Frames Processed: {total_frames:,}
- Total Duration: {total_duration:.1f} seconds
- Maximum Risk Score: {max_risk_score}
- Total Violations: {total_violations}

VIOLATIONS BREAKDOWN
"""
        for v_type, count in all_violation_counts.items():
            summary_text += f"- {v_type}: {count}\n"
        
        summary_text += f"""
{'='*50}
Risk Level: {'LOW' if max_risk_score < 30 else 'MEDIUM' if max_risk_score < 60 else 'HIGH' if max_risk_score < 100 else 'CRITICAL'}
"""
        
        st.download_button(
            "📄 Download Summary (TXT)",
            summary_text,
            file_name="safety_report.txt",
            mime="text/plain"
        )
    
    with col2:
        # JSON export
        export_data = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'sessions': len(reports),
                'total_frames': total_frames,
                'total_duration_seconds': total_duration,
                'max_risk_score': max_risk_score,
                'total_violations': total_violations
            },
            'violations': all_violation_counts,
            'sessions': [
                {'name': name, 'data': report} 
                for name, report in reports
            ]
        }
        
        st.download_button(
            "📊 Download Data (JSON)",
            json.dumps(export_data, indent=2, default=str),
            file_name="safety_report.json",
            mime="application/json"
        )
    
    # Clear all data option
    st.markdown("---")
    if st.button("🗑️ Clear All Analysis Data", type="secondary"):
        st.session_state.video_analysis_report = None
        st.session_state.video_llm_report = None
        st.session_state.live_cam_report = None
        st.session_state.live_llm_report = None
        st.rerun()


if __name__ == "__main__":
    main()
