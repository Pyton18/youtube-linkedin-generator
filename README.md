# YouTube Finance Show to LinkedIn Post Generator

Automatically convert 30-60 minute YouTube finance show videos into engaging LinkedIn posts using AI summarization and text generation.

## Features

- 🎥 **YouTube Transcript Extraction** - Automatically fetch video transcripts
- 🤖 **AI Summarization** - Use Hugging Face BART model to summarize long content
- 📝 **LinkedIn Post Generation** - Generate professional, engaging posts
- ⚙️ **Configurable** - Customize models, persona, and output length
- 🔄 **Fallback Support** - Whisper audio transcription for videos without captions

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd youtube-linkedin-generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Settings

```bash
# Copy environment template
copy env.example .env

# Edit .env with your preferences (optional)
# HF_TOKEN=your_huggingface_token_here
# SUMMARIZER_MODEL=facebook/bart-large-cnn
# GENERATOR_MODEL=distilgpt2
# PERSONA=industrial engineer
```

### 3. Run the Generator

```bash
python youtube_summarizer.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Usage

### Basic Usage
```bash
python youtube_summarizer.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Advanced Usage
```bash
python youtube_summarizer.py \
  --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --max-words 250 \
  --persona "data analyst" \
  --output-file "my_post.txt"
```

## Requirements

- **Python 3.8+**
- **RAM**: 4GB+ recommended (models will download ~2GB)
- **Internet**: For YouTube API and model downloads
- **Storage**: 2GB free space for models

## Models Used

- **Summarization**: `facebook/bart-large-cnn` (1.6GB)
- **Text Generation**: `distilgpt2` (500MB)
- **Audio Fallback**: `openai-whisper` (if no captions available)

## Configuration

Edit `.env` file to customize:

```env
# Hugging Face API (optional - for faster inference)
HF_TOKEN=your_token_here

# Model selection
SUMMARIZER_MODEL=facebook/bart-large-cnn
GENERATOR_MODEL=distilgpt2

# Output settings
MAX_POST_WORDS=300
PERSONA=industrial engineer
```

## Troubleshooting

### Out of Memory Error
If you get memory errors:
1. Try using Hugging Face Inference API (add `HF_TOKEN` to `.env`)
2. Use smaller models (e.g., `distilbart-cnn-12-6`)
3. Process shorter video segments

### No Transcript Available
- The script will automatically try Whisper audio transcription
- Ensure the video has clear audio
- Processing time will be longer

### Model Download Issues
- Check internet connection
- Ensure sufficient disk space (2GB+)
- Try running with `--verbose` flag for detailed logs

## Project Structure

```
youtube-linkedin-generator/
├── youtube_summarizer.py    # Main script
├── requirements.txt         # Python dependencies
├── env.example             # Environment template
├── .env                    # Your configuration (not tracked)
├── .gitignore              # Git ignore rules
├── outputs/                # Generated posts
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Hugging Face for the transformer models
- YouTube Transcript API for video captions
- OpenAI Whisper for audio transcription
