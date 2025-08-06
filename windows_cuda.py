#!/usr/bin/env python3
import os
import torch
import gc
import re
from TTS.api import TTS
from pydub import AudioSegment
from transformers import AutoTokenizer

# Ensure ffmpeg (required for pydub)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"


# -------------------------------
# Monkey patch to disable weights_only=True
# -------------------------------
_original_torch_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_load

# -------------------------------
# Device check
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# -------------------------------
# Load tokenizer for token-based chunking
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

def smart_chunk_text_token_limit(text, max_tokens=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        temp_chunk = (current_chunk + " " + sentence).strip()
        token_count = len(tokenizer.encode(temp_chunk))

        if token_count <= max_tokens:
            current_chunk = temp_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# -------------------------------
# Load model
# -------------------------------
print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)
print("XTTS model loaded successfully!")

# -------------------------------
# Generate long audio
# -------------------------------
def generate_long_audio(text, speaker_wav_path, output_dir="audio_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    chunks = smart_chunk_text_token_limit(text)
    audio_files = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_file = f"{output_dir}/chunk_{i:04d}.wav"
        try:
            tts.tts_to_file(
                text=chunk,
                file_path=chunk_file,
                speaker_wav=speaker_wav_path,
                language="en"
            )
            audio_files.append(chunk_file)
        except Exception as e:
            print(f"❌ Error processing chunk {i}: {e}")
            continue

        if i % 5 == 0:
            gc.collect()

    return audio_files

# -------------------------------
# Combine chunks
# -------------------------------
def combine_audio_files(audio_files, final_output="futuristic_story.wav"):
    print("🎧 Combining audio files...")
    combined = AudioSegment.empty()
    for audio_file in audio_files:
        try:
            audio = AudioSegment.from_wav(audio_file)
            combined += audio + AudioSegment.silent(duration=400)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            continue
    combined.export(final_output, format="wav")
    print(f"✅ Final audio saved as: {final_output}")
    return final_output

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    story_file_path = "./story.txt"
    speaker_voice_wav = "./Sample.wav"

    if not os.path.exists(story_file_path):
        print(f"❌ Story file not found at: {story_file_path}")
        exit()

    if not os.path.exists(speaker_voice_wav):
        print(f"❌ Speaker WAV file not found at: {speaker_voice_wav}")
        print("Please provide a valid WAV file (16 kHz, mono, 10–30 seconds of clear speech)")
        exit()

    with open(story_file_path, "r", encoding="utf-8") as f:
        story_text = f.read()

    print("🔊 Generating long-form TTS audio...")
    story_chunks = generate_long_audio(
        text=story_text,
        speaker_wav_path=speaker_voice_wav,
        output_dir="story_chunks"
    )

    if story_chunks:
        combine_audio_files(story_chunks, "futuristic_story.wav")
    else:
        print("❌ Audio generation failed.")
