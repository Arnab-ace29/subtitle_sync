import os
import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
import time
import subprocess
import json
from pydub import AudioSegment

def optimize_whisper_transcription(input_file, output_dir,optimized_aligned):
    """
    Optimized Whisper transcription with memory and speed improvements
    """
    # Use tiny model for speed
    model_name = "distil-whisper/distil-large-v3"
    
    print(f"Loading optimized model: {model_name}")
    
    # Load with low precision for speed
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Transcribing with optimized settings on {device}...")
    
    # Load audio with optimized settings
    audio, sr = librosa.load(input_file, sr=16000, mono=True, res_type='soxr_hq')
    
    # Process in chunks with overlap for better continuity
    chunk_length_s = 30  # seconds
    chunk_overlap_s = 2  # seconds overlap
    
    chunk_length = chunk_length_s * sr
    chunk_overlap = chunk_overlap_s * sr
    
    chunks = []
    timestamps = []
    for i in range(0, len(audio), chunk_length - chunk_overlap):
        chunk_end = min(i + chunk_length, len(audio))
        chunks.append(audio[i:chunk_end])
        timestamps.append(i / sr)
    
    all_transcriptions = []
    
    for i, (chunk, timestamp) in enumerate(zip(chunks, timestamps)):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Process with optimized batch settings
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt").to(device)
        inputs["input_features"] = inputs["input_features"].to(torch.float16)

        # Generate with optimized settings
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                max_length=256,
                num_beams=1,  # Greedy decoding for speed
                do_sample=False,
                temperature=1.0
            )
            
        # Convert to text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Create segment
        segment = {
            "start": timestamp,
            "end": timestamp + (len(chunk) / sr),
            "text": transcription.strip()
        }
        all_transcriptions.append(segment)
    
    # Write to SRT file
    srt_file = os.path.join(output_dir, f"{optimized_aligned}.srt")
    with open(srt_file, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(all_transcriptions):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            # Format for SRT
            start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
            end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
            
            f.write(f"{idx + 1}\n")
            f.write(f"{start_time_srt} --> {end_time_srt}\n")
            f.write(f"{text}\n\n")
    
    print(f"Optimized transcription saved to: {srt_file}")
    return srt_file

def align_with_gentle(audio_file, script_file, output_dir):
    """
    Align script with audio using Gentle forced aligner
    Note: Requires Gentle to be installed (https://github.com/lowerquality/gentle)
    """
    print("Aligning with Gentle forced aligner...")
    
    # Prepare paths
    output_json = os.path.join(output_dir, "gentle_aligned.json")
    output_srt = os.path.join(output_dir, "gentle_aligned.srt")
    
    # Read script
    with open(script_file, 'r', encoding='utf-8') as f:
        script_text = f.read()
    
    try:
        # Run Gentle aligner (make sure Gentle is installed and in PATH)
        cmd = ["gentle", "--nthreads", "4", "--disfluency", "false", 
               audio_file, script_file, "--output", output_json]
        
        subprocess.run(cmd, check=True)
        
        # Convert JSON to SRT
        with open(output_json, 'r', encoding='utf-8') as f:
            alignment = json.load(f)
        
        # Process words to create subtitle segments
        words = alignment.get('words', [])
        segments = []
        current_segment = {"words": [], "start": None, "end": None}
        
        for word in words:
            if word.get('case') == 'success':
                if current_segment["start"] is None:
                    current_segment["start"] = word["start"]
                
                current_segment["words"].append(word["word"])
                current_segment["end"] = word["end"]
                
                # Break segments at punctuation or after ~10 words
                if word["word"].endswith(('.', '!', '?')) or len(current_segment["words"]) >= 10:
                    segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "text": " ".join(current_segment["words"])
                    })
                    current_segment = {"words": [], "start": None, "end": None}
        
        # Add any remaining words
        if current_segment["words"]:
            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "text": " ".join(current_segment["words"])
            })
        
        # Write SRT file
        with open(output_srt, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(segments):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Format for SRT
                start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
                end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
                
                f.write(f"{idx + 1}\n")
                f.write(f"{start_time_srt} --> {end_time_srt}\n")
                f.write(f"{text}\n\n")
        
        print(f"Gentle alignment saved to: {output_srt}")
        return output_srt
    
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Error running Gentle: {e}")
        print("Falling back to alternative method...")
        return align_with_aeneas_simplified(audio_file, script_file, output_dir)

def align_with_aeneas_simplified(audio_file, script_file, output_dir,optimized_aligned):
    """
    Simplified version of Aeneas alignment that avoids common errors
    """
    print("Aligning with simplified Aeneas method...")
    
    try:
        # Import only when needed to avoid startup errors
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        
        # Convert audio to WAV if it's not already
        audio_wav = audio_file
        if not audio_file.lower().endswith('.wav'):
            audio_wav = os.path.join(output_dir, "temp_audio.wav")
            audio = AudioSegment.from_file(audio_file)
            audio.export(audio_wav, format="wav")
        
        # Prepare script in the format Aeneas expects
        prepared_script = os.path.join(output_dir, "prepared_script.txt")
        with open(script_file, 'r', encoding='utf-8') as f:
            script_text = f.read()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', script_text)
        
        with open(prepared_script, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                if sentence.strip():  # Skip empty lines
                    f.write(f"{i+1}|{sentence.strip()}\n")
        
        # Create a simpler task configuration
        config_string = (
            "task_language=eng|"
            "is_text_type=plain|"
            "is_text_file_format=txt|"
            "is_text_file_has_id=True|"
            "os_task_file_format=srt|"
            "is_audio_file_detect_head_max=0|"
            "is_audio_file_detect_tail_max=0|"
            "task_adjust_boundary_algorithm=percent|"
            "task_adjust_boundary_percent_value=50|"
            "task_adjust_boundary_nonspeech_min=0.1|"
            "task_adjust_boundary_nonspeech_string=|"
            "is_text_file_ignore_regex=\\[.*?\\]"
        )
        
        # Create a task
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio_wav
        task.text_file_path_absolute = prepared_script
        
        # Output the result
        output_file = os.path.join(output_dir, "aeneas_aligned.srt")
        task.sync_map_file_path_absolute = output_file
        
        # Execute with smaller segments
        ExecuteTask(task).execute()
        
        print(f"Aeneas alignment saved to: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error with Aeneas alignment: {e}")
        print("Falling back to Whisper transcription...")
        return optimize_whisper_transcription(audio_file, output_dir,optimized_aligned)

def align_with_whisperx(audio_file, script_file, output_dir,whisper_srt):
    """
    Align script with audio using WhisperX
    Note: Requires WhisperX to be installed (pip install whisperx)
    """
    try:
        import whisperx
        
        print("Aligning with WhisperX...")
        
        # Load audio
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "tiny" if device == "cpu" else "base"
        
        # Get transcription
        model = whisperx.load_model(model_name, device)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, language="en")
        
        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        
        # Align utterances
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        
        # Write to SRT
        output_srt = os.path.join(output_dir, whisper_srt)
        with open(output_srt, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Format for SRT
                start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
                end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"
                
                f.write(f"{idx + 1}\n")
                f.write(f"{start_time_srt} --> {end_time_srt}\n")
                f.write(f"{text}\n\n")
        
        print(f"WhisperX alignment saved to: {output_srt}")
        return output_srt
    
    except ImportError:
        print("WhisperX not installed. Please install with: pip install whisperx")
        print("Falling back to alternative method...")
        return align_with_gentle(audio_file, script_file, output_dir)
    except Exception as e:
        print(f"Error with WhisperX alignment: {e}")
        print("Falling back to alternative method...")
        return align_with_gentle(audio_file, script_file, output_dir)

def sync_script_with_audio(audio_file, script_file, output_dir,whisper_srt,optimized_aligned):
    """
    Master function to synchronize your existing script with audio
    Tries multiple methods in order of preference
    """
    # Try modern alignment methods first
    try:
        return align_with_whisperx(audio_file, script_file, output_dir,whisper_srt)
    except Exception as e:
        print(f"WhisperX method failed: {e}")
    
    try:
        return align_with_gentle(audio_file, script_file, output_dir)
    except Exception as e:
        print(f"Gentle method failed: {e}")
    
    try:
        return align_with_aeneas_simplified(audio_file, script_file, output_dir,optimized_aligned)
    except Exception as e:
        print(f"Aeneas method failed: {e}")
    
    # Last resort: Whisper transcription
    print("All alignment methods failed. Falling back to Whisper transcription")
    return optimize_whisper_transcription(audio_file, output_dir,optimized_aligned)

def benchmark_and_sync(audio_file, script_file, output_dir,whisper_srt,optimized_aligned):
    """
    Benchmark different approaches and choose the fastest
    """
    print("Benchmarking different approaches...")
    
    # Method 1: Quick Whisper transcription
    start_time = time.time()
    whisper_srt = optimize_whisper_transcription(audio_file, output_dir,optimized_aligned)
    whisper_time = time.time() - start_time
    print(f"Whisper transcription completed in {whisper_time:.2f} seconds")
    
    # Method 2: Sync existing script with audio
    start_time = time.time()
    try:
        aligned_srt = sync_script_with_audio(audio_file, script_file, output_dir,whisper_srt,optimized_aligned)
        sync_time = time.time() - start_time
        print(f"Script syncing completed in {sync_time:.2f} seconds")
        
        # Compare results
        print("\nBenchmark Results:")
        print(f"Whisper transcription: {whisper_time:.2f}s")
        print(f"Script synchronization: {sync_time:.2f}s")
        print(f"\nRecommended approach: {'Script synchronization' if sync_time < whisper_time * 1.5 else 'Whisper transcription'}")
        
        return aligned_srt if sync_time < whisper_time * 1.5 else whisper_srt
    except Exception as e:
        print(f"Script syncing failed: {e}")
        print("Falling back to Whisper transcription")
        return whisper_srt

def create_merged_subtitle(whisper_srt, aligned_srt, output_dir,srt_name):
    """
    Create a merged subtitle file that combines the best of both approaches
    This can help improve accuracy
    """
    from pysrt import open as srt_open
    
    print("Creating merged subtitle file...")
    
    # Read both SRT files
    try:
        whisper_subs = srt_open(whisper_srt)
        aligned_subs = srt_open(aligned_srt)
        
        # Create a merged SRT file
        output_srt = os.path.join(output_dir,srt_name )
        
        # Use timing from aligned_subs and add whisper text as alternative
        with open(output_srt, "w", encoding="utf-8") as f:
            for i, aligned_sub in enumerate(aligned_subs):
                # Find closest whisper sub by time
                closest_whisper = min(whisper_subs, 
                                     key=lambda x: abs((x.start.ordinal - aligned_sub.start.ordinal) / 1000))
                
                # Write to file
                f.write(f"{i+1}\n")
                f.write(f"{aligned_sub.start} --> {aligned_sub.end}\n")
                f.write(f"{aligned_sub.text}\n")
                
                # Add whisper text as comment if significantly different
                if aligned_sub.text.lower() != closest_whisper.text.lower():
                    f.write(f"[Whisper: {closest_whisper.text}]\n")
                
                f.write("\n")
        
        print(f"Merged subtitle file saved to: {output_srt}")
        return output_srt
    except Exception as e:
        print(f"Error creating merged subtitle: {e}")
        return aligned_srt  # Return aligned_srt as fallback

def convert_audio_format(input_file, output_dir):
    """
    Convert audio to WAV format if needed
    """
    if input_file.lower().endswith('.wav'):
        return input_file
    
    # Convert to WAV
    output_file = os.path.join(output_dir, "temp_audio.wav")
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        print(f"Converted audio to WAV: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_file

def ensure_directory_exists(directory):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main(input_audio_file,script_file,output_directory,srt_name,whisper_srt,optimized_aligned):
    """
    Main function to process audio and script files
    """
    # Example usage
    # input_audio_file = "./audiobook/The Brothers Karamazov/1.wav"
    # script_file = "./audiobook/The Brothers Karamazov/Chapter I.txt"
    # output_directory = "./audiobook/The Brothers Karamazov"

    method = 'auto'  # Choices: 'whisper', 'align', 'auto'
    
    # Ensure output directory exists
    ensure_directory_exists(output_directory)
    
    # Convert audio format if needed
    input_audio_file = convert_audio_format(input_audio_file, output_directory)
    
    # Process based on method
    if method == 'whisper':
        srt_file = optimize_whisper_transcription(input_audio_file, output_directory,optimized_aligned)
    elif method == 'align':
        srt_file = sync_script_with_audio(input_audio_file, script_file, output_directory,whisper_srt,optimized_aligned)
    else:  # auto
        # Try syncing first, fallback to whisper if it fails
        try:
            aligned_srt = sync_script_with_audio(input_audio_file, script_file, output_directory,whisper_srt,optimized_aligned)
            whisper_srt = optimize_whisper_transcription(input_audio_file, output_directory,optimized_aligned)
            
            # Create merged subtitle file
            srt_file = create_merged_subtitle(whisper_srt, aligned_srt, output_directory,srt_name)
        except Exception as e:
            print(f"Error during processing: {e}")
            srt_file = benchmark_and_sync(input_audio_file, script_file, output_directory,whisper_srt,optimized_aligned)
    
    print(f"Final SRT file: {srt_file}")
    return srt_file


############ Example Usage ################



srt_name = f"merged.srt"    
whisper_srt = f"whisper.srt" 
optimized_aligned = f"optimized.srt" 
input_audio_file = f"./audiobook/audio.wav"  
script_file = f"./audiobook/Script.txt"
output_directory = "./audiobook/Subtitles"
main(input_audio_file,script_file,output_directory,srt_name,whisper_srt,optimized_aligned)
