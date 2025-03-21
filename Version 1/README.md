# Audio-to-Text Synchronization Tool

This tool provides a robust solution for synchronizing audio files with text scripts, generating accurate subtitle files in SRT format. It employs multiple alignment methods with built-in fallback mechanisms to ensure reliable results across different scenarios.

## 🔍 Overview

This Python-based tool can:

1. Transcribe audio using Whisper models
2. Align existing scripts with audio timestamps
3. Compare and merge results from different approaches
4. Handle various audio formats
5. Provide performance benchmarking

The system is designed with robust error handling, making it suitable for production environments where reliability is crucial.

## ⚙️ Features

### Multiple Alignment Methods
- **WhisperX**: State-of-the-art audio transcription and alignment
- **Gentle Forced Aligner**: High-precision alignment for existing scripts
- **Aeneas**: Simplified alternative alignment method
- **Optimized Whisper**: High-performance transcription with memory efficiency

### Smart Fallback System
The tool automatically tries multiple methods in order of accuracy, falling back to alternatives if any method fails. This ensures you always get results, even in challenging scenarios.

### Optimization Techniques
- Chunk-based processing for long audio files
- GPU acceleration (when available)
- Low-precision computation for speed
- Memory optimization for large files

### Output Options
- Standard SRT subtitle format
- Merged subtitles with alternative transcriptions
- Performance benchmarking data

## 🔧 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/audio-sync-tool.git
cd audio-sync-tool

# Install required packages
pip install torch numpy librosa tqdm transformers pydub pysrt

# Optional dependencies for specific alignment methods
pip install whisperx  # For WhisperX alignment
# For Gentle, follow installation guide at: https://github.com/lowerquality/gentle
# For Aeneas: pip install aeneas
```

## 🚀 Usage

### Basic Usage

```python
from audio_sync import main

main(
    input_audio_file="path/to/audio.mp3",
    script_file="path/to/script.txt",
    output_directory="path/to/output",
    srt_name="final_subtitles.srt",
    whisper_srt="whisper_aligned.srt",
    optimized_aligned="whisper_optimized.srt"
)
```

### Function Details

#### Main Function
```python
main(input_audio_file, script_file, output_directory, srt_name, whisper_srt, optimized_aligned)
```
- `input_audio_file`: Path to the audio file (supports multiple formats)
- `script_file`: Path to the text script file
- `output_directory`: Where to save the output files
- `srt_name`: Name of the final SRT file
- `whisper_srt`: Name for the Whisper transcription SRT file
- `optimized_aligned`: Name for the optimized alignment SRT file

## 🔄 Process Flow

1. **Preparation**:
   - Ensure output directory exists
   - Convert audio to compatible format if needed

2. **Alignment Attempts** (in order):
   - Try WhisperX alignment
   - If that fails, try Gentle forced alignment
   - If that fails, try Aeneas simplified alignment
   - As a last resort, use optimized Whisper transcription

3. **Result Processing**:
   - Generate SRT files from alignment data
   - Optionally merge results from different methods
   - Return path to the final SRT file

## 💡 Key Components

### Whisper Transcription

The `optimize_whisper_transcription()` function uses the Whisper model with the following optimizations:
- Processes audio in overlapping chunks for better continuity
- Uses half-precision (float16) for faster processing
- Employs GPU acceleration when available
- Uses simpler decoding strategies for speed

### Gentle Alignment

The `align_with_gentle()` function:
- Aligns script text with audio using forced alignment
- Creates segments at natural break points (punctuation)
- Formats output as standard SRT format

### WhisperX Alignment

The `align_with_whisperx()` function:
- Leverages the WhisperX library for state-of-the-art alignment
- Uses appropriate model size based on available hardware
- Generates precisely aligned timestamps for each segment

### Aeneas Simplified Alignment

The `align_with_aeneas_simplified()` function:
- Provides a simplified interface to Aeneas alignment
- Handles script preparation automatically
- Uses appropriate configuration for reliable alignment

### Benchmark and Comparison

The `benchmark_and_sync()` function:
- Times different approaches
- Recommends the most efficient method
- Returns results from the best-performing approach

## 📋 Output Examples

The tool generates SRT files in standard format:

```
1
00:00:01,500 --> 00:00:05,000
This is the first subtitle segment.

2
00:00:05,200 --> 00:00:08,500
This is the second subtitle segment.
```

When using the merged output option, alternative transcriptions are included:

```
1
00:00:01,500 --> 00:00:05,000
This is the first subtitle segment.
[Whisper: This is the first subtitle segment with slight variation.]

2
00:00:05,200 --> 00:00:08,500
This is the second subtitle segment.
```

## ⚠️ Requirements

- Python 3.7+
- PyTorch
- transformers
- librosa
- pydub
- pysrt (for merged subtitles)
- GPU recommended for larger files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
