import os
import re
import subprocess
from tqdm import tqdm
from pydub import AudioSegment
import whisper
import numpy as np
from dtw import dtw
import whisperx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_audio_duration(audio_file):
    """Get audio duration in seconds"""
    try:
        audio = AudioSegment.from_file(audio_file)
        return len(audio) / 1000.0  # Convert to seconds
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 60.0  # Default to 1 minute if unable to determine


def format_timestamp(seconds):
    """Format seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_int = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"


def preprocess_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Convert to lowercase for better matching
    text = text.lower()
    # Remove punctuation for better matching
    text = re.sub(r"[^\w\s]", "", text)
    return text


def create_aligned_srt(
    script_text, audio_file, output_file, model_name="distil-large-v3"
):
    """Create an SRT file with accurate timing by aligning script with whisper output"""
    print(f"Loading whisper model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "tiny" if device == "cpu" else model_name

    # Load WhisperX model
    model = whisperx.load_model(model_name, device)

    print(f"Transcribing audio with whisperx...")
    # WhisperX doesn't use word_timestamps parameter directly
    result = model.transcribe(audio_file)

    whisper_text = result["text"]
    segments = result["segments"]

    # Align the segments with the audio
    print("Aligning timestamps...")
    audio = whisperx.load_audio(audio_file)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    segments = result["segments"]

    print(f"Aligning script with whisperx output...")

    # Preprocess both texts
    clean_script = preprocess_text(script_text)
    clean_whisper = preprocess_text(whisper_text)

    # Split script into sentences
    script_sentences = re.split(r"(?<=[.!?])\s+", script_text)
    script_sentences = [s.strip() for s in script_sentences if s.strip()]

    # Group sentences into logical chunks
    chunks = []
    current_chunk = []
    current_length = 0
    target_chunk_size = 100  # Target character length for chunks

    for sentence in script_sentences:
        # If adding this sentence would make the chunk too large or we already have 2 sentences
        if (
            current_length + len(sentence) > target_chunk_size
            or len(current_chunk) >= 2
        ):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Get word-level timestamps from whisperx
    word_timestamps = []
    for segment in segments:
        for word in segment.get("words", []):
            word_timestamps.append(
                {
                    "word": word.get("word", "").strip(),
                    "start": word.get("start", 0),
                    "end": word.get("end", 0),
                }
            )

    # If word timestamps are not available, fall back to segment-level timestamps
    if not word_timestamps:
        for segment in segments:
            word_timestamps.append(
                {
                    "word": segment.get("text", "").strip(),
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                }
            )

    # Create sliding window for better alignment
    def get_window_text(start_idx, window_size):
        end_idx = min(start_idx + window_size, len(word_timestamps))
        return " ".join(
            [wt["word"].lower() for wt in word_timestamps[start_idx:end_idx]]
        )

    # Vectorizer for text similarity
    vectorizer = TfidfVectorizer()

    # Function to calculate similarity between two text strings
    def calculate_similarity(text1, text2):
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        except:
            return 0

    # Process chunks in sequential order to maintain temporal consistency
    chunk_timings = []
    current_position = 0  # Keep track of where we are in the audio

    for i, chunk in enumerate(chunks):
        clean_chunk = preprocess_text(chunk)
        chunk_words = clean_chunk.split()

        best_similarity = -1
        best_start_idx = current_position
        best_end_idx = -1
        search_window_size = 100  # Number of words to search ahead

        # Use a sliding window approach with bounds to prevent out-of-order alignments
        for j in range(
            current_position,
            min(
                current_position + search_window_size,
                len(word_timestamps) - len(chunk_words) + 1,
            ),
        ):
            window_text = get_window_text(j, len(chunk_words) + 10)  # Add padding
            similarity = calculate_similarity(clean_chunk, window_text)

            if similarity > best_similarity:
                best_similarity = similarity
                best_start_idx = j
                best_end_idx = min(j + len(chunk_words) - 1, len(word_timestamps) - 1)

        # If we couldn't find a good match, use the current position as fallback
        if (
            best_similarity < 0.1 and i > 0
        ):  # Threshold for minimum acceptable similarity
            # For subsequent chunks, use the end of the previous chunk as starting point
            best_start_idx = current_position
            best_end_idx = min(
                best_start_idx + len(chunk_words) - 1, len(word_timestamps) - 1
            )

        # Ensure we're making forward progress
        if best_start_idx < current_position:
            best_start_idx = current_position

        chunk_start = word_timestamps[best_start_idx]["start"]
        chunk_end = word_timestamps[best_end_idx]["end"]

        # Update current position for next chunk
        current_position = best_end_idx + 1

        chunk_timings.append({"text": chunk, "start": chunk_start, "end": chunk_end})

    # Post-process to ensure no overlapping timestamps
    for i in range(1, len(chunk_timings)):
        if chunk_timings[i]["start"] < chunk_timings[i - 1]["end"]:
            chunk_timings[i]["start"] = chunk_timings[i - 1]["end"] + 0.01

    # Write SRT file
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunk_timings):
            # Format timestamps
            start_time_srt = format_timestamp(chunk["start"])
            end_time_srt = format_timestamp(chunk["end"])

            # Write subtitle entry
            f.write(f"{i + 1}\n")
            f.write(f"{start_time_srt} --> {end_time_srt}\n")
            f.write(f"{chunk['text']}\n\n")

    print(f"Aligned SRT file created: {output_file}")
    return output_file


def process_file(audio_file, script_file, output_file):
    """Process a single audio file and script to create subtitles"""
    print(f"Processing {os.path.basename(audio_file)}...")

    # Read script
    with open(script_file, "r", encoding="utf-8") as f:
        script_text = f.read()

    # Clean text
    script_text = re.sub(r"\s+", " ", script_text).strip()

    # Create aligned SRT file
    return create_aligned_srt(script_text, audio_file, output_file)


#########################Example Usage#########################

f_name = f"main_1.wav"

audio_path = f"./audio/Iliad/{f_name}"
script_path = "./audio/Iliad/Part 1.txt"
srt_path = f"./audio/Iliad/Part 1.srt"
process_file(audio_path, script_path, srt_path)
