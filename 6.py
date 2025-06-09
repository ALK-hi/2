import streamlit as st
import os
import tempfile
import subprocess
import re
import json
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, AudioFileClip
import cv2
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Whisper support
try:
    import whisper
    whisper_available = True
except ImportError:
    whisper_available = False

# Streamlit config
st.set_page_config(page_title="VideoEditGenius", page_icon="🎬", layout="wide")

# --- Utility Functions ---
def validate_video(video):
    if not video.isOpened():
        raise ValueError("Cannot open video file.")
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS is zero. Possibly unsupported video format.")
    return fps

def cleanup_temp(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        logger.warning(f"Temp cleanup failed: {e}")

def extract_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio_path = tempfile.mktemp(suffix=".wav")
        clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        clip.close()
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        raise

def transcribe_audio(audio_path, use_whisper=False):
    try:
        if use_whisper and whisper_available:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            return result.get("text", "")

        recognizer = sr.Recognizer()
        audio = AudioSegment.from_file(audio_path)
        wav_path = audio_path.replace(os.path.splitext(audio_path)[1], ".wav")
        audio.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)

    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Speech recognition service error: {e}"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "Transcription failed."

def add_background_music(video_path, music_path, output_path, music_volume=0.1):
    try:
        video = VideoFileClip(video_path)
        music = AudioFileClip(music_path).volumex(music_volume)
        music = music.set_duration(video.duration)
        video = video.set_audio(music)
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        logger.error(f"Failed to add background music: {e}")
        raise

def add_auto_captions(video_path, transcript, output_path):
    try:
        video = VideoFileClip(video_path)
        words = transcript.strip().split()
        duration_per_word = video.duration / max(len(words), 1)

        caption_clips = []
        for i, word in enumerate(words):
            txt_clip = TextClip(word, fontsize=40, color='white', method='pillow', bg_color='black', size=video.size)
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(i * duration_per_word).set_duration(duration_per_word)
            caption_clips.append(txt_clip)

        final = CompositeVideoClip([video] + caption_clips)
        final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    except Exception as e:
        logger.error(f"Failed to add timed captions: {e}")
        raise

def analyze_video(video_path, api_key, use_whisper=False):
    if not api_key:
        return {"summary": "Missing API key.", "viral_moments": [], "edit_suggestions": []}

    try:
        cap = cv2.VideoCapture(video_path)
        fps = validate_video(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        frames, times = [], []
        for i in range(5):
            idx = int(total_frames * (i / 4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                times.append(idx / fps)
        cap.release()

        audio_path = extract_audio(video_path)
        transcript = transcribe_audio(audio_path, use_whisper)
        cleanup_temp(audio_path)

        descriptions = [f"Frame at {t:.2f}s: [Image content]" for t in times]
        prompt = f"""
        Analyze the video:
        Duration: {duration:.2f}s
        Key Frames:
        {chr(10).join(descriptions)}

        Transcript:
        {transcript}

        Output JSON:
        {{
            "summary": "...",
            "viral_moments": [...],
            "edit_suggestions": [...],
            "youtube_shorts_potential": "High|Medium|Low",
            "editing_code": "..."
        }}
        """

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text

        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text

        return json.loads(json_str.strip())

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "summary": f"Error during analysis: {e}",
            "viral_moments": [],
            "edit_suggestions": [],
            "youtube_shorts_potential": "Unknown",
            "editing_code": "# Error"
        }

# --- Streamlit UI ---
st.title("🎬 VideoEditGenius")
st.markdown("Upload your video. Let AI suggest edits and extract viral clips.")

api_key = st.sidebar.text_input("Gemini API Key", type="password")
use_whisper = st.sidebar.checkbox("Use Whisper (if installed)", value=whisper_available)
add_music = st.sidebar.checkbox("Add Background Music")
music_file = st.sidebar.file_uploader("Upload Music", type=["mp3", "wav"]) if add_music else None
add_captions = st.sidebar.checkbox("Add Auto-Captions")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

if uploaded:
    tmp_path = tempfile.mktemp(suffix=".mp4")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())
    st.video(tmp_path)

    if st.button("Analyze Video"):
        with st.spinner("Analyzing..."):
            result = analyze_video(tmp_path, api_key, use_whisper)
            st.session_state["analysis"] = result
            st.session_state["video_path"] = tmp_path
            st.session_state["transcript"] = result.get("transcript", "")

if "analysis" in st.session_state:
    data = st.session_state["analysis"]
    st.subheader("📋 Summary")
    st.write(data.get("summary", "No summary."))

    st.subheader("🔥 Viral Moments")
    for i, moment in enumerate(data.get("viral_moments", [])):
        st.markdown(f"**{i+1}. {moment.get('title_suggestion', 'Clip')}**")
        st.text(f"Timestamp: {moment.get('timestamp')}s | Duration: {moment.get('duration')}s")
        st.caption(moment.get("description", "No description."))

    st.subheader("✂️ Edit Suggestions")
    for i, edit in enumerate(data.get("edit_suggestions", [])):
        st.markdown(f"**{edit.get('type', 'Edit')}** at {edit.get('timestamp')}s: {edit.get('description')}")

    st.subheader("💡 Editing Code")
    st.code(data.get("editing_code", "# No code generated."), language="python")

    if add_music and music_file:
        st.subheader("🎵 Add Background Music")
        music_tmp = tempfile.mktemp(suffix=os.path.splitext(music_file.name)[-1])
        with open(music_tmp, "wb") as f:
            f.write(music_file.read())

        output_with_music = tmp_path.replace(".mp4", "_with_music.mp4")
        if st.button("Apply Music"):
            try:
                with st.spinner("Applying music..."):
                    add_background_music(st.session_state["video_path"], music_tmp, output_with_music)
                    st.success("Background music added!")
                    st.video(output_with_music)
            except Exception as e:
                st.error(f"Failed to add music: {e}")

    if add_captions:
        st.subheader("💬 Add Auto-Captions")
        transcript = st.session_state.get("transcript", "") or data.get("summary", "")
        captioned_output = tmp_path.replace(".mp4", "_captioned.mp4")
        if st.button("Apply Captions"):
            try:
                with st.spinner("Adding captions..."):
                    add_auto_captions(st.session_state["video_path"], transcript, captioned_output)
                    st.success("Captions added!")
                    st.video(captioned_output)
            except Exception as e:
                st.error(f"Failed to add captions: {e}")
