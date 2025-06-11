# main.py
import os
import io
import base64
import logging
import torch
import numpy as np
from scipy.io import wavfile
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
DEVICE = 0 if torch.cuda.is_available() else -1

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Pipelines ---
# ASR
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", token=HF_API_KEY, device=DEVICE)

# TTS
tts_pipe = pipeline("text-to-speech", model="suno/bark-small", token=HF_API_KEY, device=DEVICE)

# LLM with LangChain
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=DEVICE)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt + Chain
prompt = PromptTemplate(input_variables=["question"], template="You are an assistant. Answer the question briefly.\nQuestion: {question}\nAnswer:")
qa_chain = prompt | llm  # RunnableSequence

# --- FastAPI Setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        input_bytes = await audio.read()
        content_type = audio.content_type  # e.g., 'audio/webm', 'audio/wav', 'audio/mp3'

        if content_type == "audio/wav":
            audio_format = "wav"
        elif content_type == "audio/mpeg":
            audio_format = "mp3"
        elif content_type == "audio/webm":
            audio_format = "webm"
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported audio format: {content_type}"})

        audio_segment = AudioSegment.from_file(io.BytesIO(input_bytes), format=audio_format).set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # Read WAV as numpy array
        sample_rate, data = wavfile.read(wav_io)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        # Run ASR
        result = asr_pipe(data)
        user_text = result['text'].strip()

        if not user_text:
            return JSONResponse(content={"user_text": "[Unrecognized speech]", "ai_text": "", "audio_base64": None})

        # LLM response via LangChain
        ai_text = qa_chain.invoke({"question": user_text}).strip()

        # TTS output
        tts_output = tts_pipe(ai_text)
        # Convert raw audio to AudioSegment, then encode to mp3
        tts_audio = np.array(tts_output["audio"])  # float32 PCM
        tts_audio_int16 = np.int16(tts_audio * 32767)  # Convert to int16 for saving
        wav_io = io.BytesIO()
        wavfile.write(wav_io, rate=tts_output["sampling_rate"], data=tts_audio_int16)
        wav_io.seek(0)

        # Base64 encode the WAV output
        audio_base64 = base64.b64encode(wav_io.read()).decode("utf-8")

        return JSONResponse(content={
            "user_text": user_text,
            "ai_text": ai_text,
            "audio_base64": audio_base64
        })

    except Exception as e:
        logging.exception("Error during processing")
        return JSONResponse(status_code=500, content={"error": str(e)})
