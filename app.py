# main.py
import os
import io
import base64
import logging
import numpy as np
from scipy.io import wavfile
from dotenv import load_dotenv
from pydub import AudioSegment
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import BarkProcessor, BarkModel, pipeline
import torch
import uvicorn
import httpx
from pymongo import MongoClient
from datetime import datetime, timezone

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Configure logging
logging.basicConfig(level=logging.INFO)

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.chatbot_db
logs_collection = db.logs

# Device setup
DEVICE = "cpu"
print("Device set to use", DEVICE)

# --- Pipelines ---
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", token=HF_API_KEY, device=0 if DEVICE == "cuda" else -1, framework="pt")

# Bark TTS setup
tts_processor = BarkProcessor.from_pretrained("suno/bark")
tts_model = BarkModel.from_pretrained("suno/bark").to(DEVICE)

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an assistant. Answer the question briefly.\nQuestion: {question}\nAnswer:"
)
qa_chain = prompt | llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        input_bytes = await audio.read()
        content_type = audio.content_type or ""

        if content_type.endswith("wav"):
            audio_format = "wav"
        elif content_type.endswith("mpeg") or content_type.endswith("mp3"):
            audio_format = "mp3"
        elif content_type.endswith("webm"):
            audio_format = "webm"
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported audio format: {content_type}"})

        audio_segment = AudioSegment.from_file(io.BytesIO(input_bytes), format=audio_format).set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        sample_rate, data = wavfile.read(wav_io)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        user_text = asr_pipe(data)["text"].strip()

        if not user_text:
            return JSONResponse(content={"user_text": "[Unrecognized speech]", "ai_text": "", "audio_base64": None})

        try:
            ai_message = qa_chain.invoke({"question": user_text})
            ai_text = ai_message.content.strip()
        except Exception as e:
            logging.error("LLM/Groq error:", exc_info=True)
            return JSONResponse(status_code=503, content={"error": "Groq backend temporarily unavailable. Please try again later."})

        inputs = tts_processor(ai_text, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        tts_audio = tts_model.generate(**inputs).cpu().numpy()[0]

        wav_io = io.BytesIO()
        wavfile.write(wav_io, rate=22050, data=np.int16(tts_audio * 32767))
        wav_io.seek(0)
        audio_base64 = base64.b64encode(wav_io.read()).decode("utf-8")

        logs_collection.insert_one({
            "timestamp": datetime.now(timezone.utc),
            "user_text": user_text,
            "ai_text": ai_text
        })

        return JSONResponse(content={
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_base64
        })

    except Exception as e:
        logging.exception("Error during processing")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/logs")
async def get_logs():
    try:
        logs = list(logs_collection.find({}, {"_id": 0}).sort("timestamp", -1))
        return JSONResponse(content={"logs": logs})
    except Exception as e:
        logging.exception("Error fetching logs")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
