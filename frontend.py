import streamlit as st
import requests
import base64
import tempfile
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="Voice Assistant", page_icon="🧠", layout="centered")
st.title("🧠 Voice Assistant")
st.markdown("Press the mic button to speak. Your question will be answered immediately.")

# --- Record audio ---
audio_bytes = audio_recorder(pause_threshold=1.5)

if audio_bytes:
    with st.spinner("Transcribing and processing..."):
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            tmpfile_path = tmpfile.name

        # Send to backend
        with open(tmpfile_path, "rb") as f:
            files = {"audio": ("recording.wav", f, "audio/wav")}
            try:
                response = requests.post("http://localhost:8001/process-audio", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Response received")
                    st.markdown(f"**🧑 You said:** {data['user_text']}")
                    st.markdown(f"**🤖 Assistant:** {data['ai_text']}")

                    # Play response audio
                    audio_data = base64.b64decode(data["audio_base64"])
                    st.audio(audio_data, format="audio/wav")

                else:
                    st.error(f"Backend error: {response.json().get('error')}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
