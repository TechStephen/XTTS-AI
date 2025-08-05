import os
import torch
import gc
import re
from TTS.api import TTS
from pydub import AudioSegment

# Detect MPS GPU (Apple Silicon)
def is_mps_available():
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

use_gpu = is_mps_available()
print(f"Apple GPU acceleration (MPS) available: {use_gpu}")

# Fix PyTorch compatibility for older TTS models
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)

# Initialize XTTS model (no 'gpu' arg)
print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
if is_mps_available():
    tts.to("mps")
print("XTTS model loaded successfully!")

# Sentence-aware, performance-optimized chunking
def smart_chunk_text(text, max_chunk_size=2400):  # Increased for fewer TTS calls
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if len(sentence) > max_chunk_size:
                clauses = re.split(r'(,|;|:)', sentence)
                for clause in clauses:
                    if len(current_chunk) + len(clause) + 1 <= max_chunk_size:
                        current_chunk += clause
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = clause
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Generate audio from long text
def generate_long_audio(text, speaker_wav_path, output_dir="audio_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    chunks = smart_chunk_text(text, max_chunk_size=2400)
    audio_files = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({'GPU' if use_gpu else 'CPU'} mode)")
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
            print(f"Error processing chunk {i}: {e}")
            continue

        if i % 5 == 0:
            gc.collect()

    return audio_files

# Combine audio chunks into one file
def combine_audio_files(audio_files, final_output="futuristic_story.wav"):
    print("Combining audio files...")
    combined = AudioSegment.empty()

    for audio_file in audio_files:
        try:
            audio = AudioSegment.from_wav(audio_file)
            combined += audio + AudioSegment.silent(duration=500)
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            continue

    combined.export(final_output, format="wav")
    print(f"✅ Final audio saved as: {final_output}")
    return final_output

# Entry point
if __name__ == "__main__":
    story_file_path = "./story.txt"
    speaker_voice_wav = "./Sample.wav"

    if not os.path.exists(story_file_path):
        print(f"❌ ERROR: Story file not found at: {story_file_path}")
        exit()

    if not os.path.exists(speaker_voice_wav):
        print(f"❌ ERROR: Speaker WAV file not found at: {speaker_voice_wav}")
        print("Please provide a valid WAV file (16 kHz, mono, 10–30 seconds of clean speech)")
        exit()

    with open(story_file_path, "r", encoding="utf-8") as f:
        story_text = f.read()

    print("🚀 Generating long-form TTS audio...")
    story_chunks = generate_long_audio(
        text=story_text,
        speaker_wav_path=speaker_voice_wav,
        output_dir="story_chunks"
    )

    if story_chunks:
        final_audio = combine_audio_files(story_chunks, "futuristic_story.wav")
        print(f"🎧 Story audio successfully created: {final_audio}")
    else:
        print("❌ Audio generation failed.")
