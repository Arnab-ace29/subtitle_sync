# 🎬 Audio-Subtitle Synchronization Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://github.com/openai/whisper)

A powerful tool for synchronizing audio files with subtitle text, creating precisely timed SRT files by aligning your script with OpenAI Whisper transcriptions.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Advanced Configuration](#-advanced-configuration)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

This tool addresses a common problem in content creation: synchronizing a pre-written script with recorded audio to generate accurate subtitles. Using OpenAI's Whisper speech recognition model, it automatically aligns your script to the audio and produces properly timed SRT subtitle files.

**Perfect for:**
- YouTubers with scripted content
- Podcast producers
- Educational video creators
- Content localization teams
- Documentary filmmakers

## ✨ Features

- **Accurate Alignment** - Precisely match your script to spoken audio
- **Sequential Processing** - Ensures subtitles appear in the correct order
- **Multiple Whisper Models** - Compatible with all Whisper models (base, small, medium, large, distilled variants)
- **Word-Level Timing** - Utilizes Whisper's word timestamps for precision
- **Robust Text Matching** - Uses TF-IDF similarity for better text alignment
- **Automatic Subtitle Formatting** - Creates properly formatted SRT files
- **Flexible Chunking** - Intelligent sentence grouping for readable subtitles

## 📥 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/subtitle-sync.git
cd subtitle-sync

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```
openai-whisper>=20231117
pydub>=0.25.1
tqdm>=4.65.0
numpy>=1.22.0
scikit-learn>=1.0.2
dtw-python>=1.3.0
```

## 🚀 Usage

### Basic Usage

```python
from subtitle_sync import process_file

# Process a single file
process_file(
    audio_file="path/to/your/audio.mp3",
    script_file="path/to/your/script.txt",
    output_file="path/to/output/subtitles.srt"
)
```

### Command Line Interface

```bash
python subtitle_sync.py --audio path/to/audio.mp3 --script path/to/script.txt --output subtitles.srt --model large-v3
```

### Batch Processing

```python
import os
from subtitle_sync import process_file

# Define directories
audio_dir = "audio_files"
script_dir = "scripts"
output_dir = "subtitles"

# Process all matching files
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(('.mp3', '.wav', '.m4a')):
        base_name = os.path.splitext(audio_file)[0]
        script_file = os.path.join(script_dir, f"{base_name}.txt")
        output_file = os.path.join(output_dir, f"{base_name}.srt")
        
        if os.path.exists(script_file):
            process_file(
                audio_file=os.path.join(audio_dir, audio_file),
                script_file=script_file,
                output_file=output_file
            )
```

## 🔧 How It Works

The synchronization process involves several sophisticated steps:

1. **Audio Transcription**
   - Your audio is transcribed using OpenAI's Whisper model
   - The model produces a transcript with word-level timestamps

2. **Text Preprocessing**
   - Both your script and the Whisper transcript are normalized
   - Text is converted to lowercase, punctuation is removed, and whitespace is standardized

3. **Script Chunking**
   - Your script is divided into sentences
   - Sentences are grouped into logical chunks based on length and natural breaks

4. **Alignment Algorithm**
   - The tool uses a sliding window approach to find the best match for each chunk
   - Text similarity is calculated using TF-IDF vectorization
   - A sequential process ensures temporal consistency:
     ```
     For each chunk in your script:
     1. Look for best matching section in Whisper transcript
     2. Only search forward from current position (prevents out-of-order subtitles)
     3. Calculate start and end timestamps for the chunk
     4. Update current position to continue sequential processing
     ```

5. **Timestamp Refinement**
   - Overlapping timestamps are detected and fixed
   - Ensures subtitles appear and disappear smoothly

6. **SRT Generation**
   - Timestamps are formatted according to SRT specifications
   - Final subtitle file is written with proper numbering and formatting

## ⚙️ Advanced Configuration

### Model Selection

The tool supports all Whisper model variants. Larger models provide better accuracy but require more computational resources:

```python
# Available model options:
models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", 
          "distil-medium.en", "distil-large-v2", "distil-large-v3"]

# Usage example with large-v3 model
create_aligned_srt(script_text, audio_file, output_file, model_name="large-v3")
```

### Chunk Size Tuning

You can adjust how the script is divided into subtitle chunks by modifying the `target_chunk_size` variable:

```python
# For shorter subtitles (easier to read quickly)
target_chunk_size = 70  # Characters per chunk

# For longer subtitles (fewer subtitle changes)
target_chunk_size = 120  # Characters per chunk
```

### Similarity Threshold

The minimum acceptable similarity score can be adjusted:

```python
# More permissive matching (may help with heavily paraphrased content)
if best_similarity < 0.05:  # Lower threshold
    # fallback logic

# Stricter matching (better for verbatim scripts)
if best_similarity < 0.2:  # Higher threshold
    # fallback logic
```

## 🔍 Troubleshooting

### Common Issues and Solutions

| Issue | Possible Solution |
|-------|------------------|
| Random subtitles appear out of order | Increase the search window size to find better matches |
| Subtitles appear too early/late | Check if your script matches what's actually spoken |
| Memory errors with large files | Use a smaller Whisper model or process audio in chunks |
| Slow processing times | Use a distilled model variant or a GPU for acceleration |
| Poor matching with ad-libbed content | Lower the similarity threshold or consider transcription-only mode |

### Debugging Tips

Add debug logging to see the alignment process in detail:

```python
# Add this parameter
create_aligned_srt(script_text, audio_file, output_file, debug=True)
```

## 🚀 Performance Tips

1. **GPU Acceleration**
   - Whisper models run significantly faster with CUDA-enabled GPUs
   - Install the GPU version of PyTorch before installing whisper

2. **Model Selection**
   - For quick testing: use "tiny" or "base" models
   - For production: "distil-large-v3" offers the best speed/accuracy tradeoff
   - For maximum accuracy: "large-v3" is the most accurate but slowest

3. **Preprocess Audio**
   - Converting to mono 16kHz WAV format can speed up processing
   - Removing background noise can improve transcription quality

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Made with ❤️ for content creators everywhere
