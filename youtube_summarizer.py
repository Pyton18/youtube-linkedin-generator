#!/usr/bin/env python3
"""
YouTube Finance Show to LinkedIn Post Generator

Converts 30-60 minute YouTube finance videos into engaging LinkedIn posts
using AI summarization and text generation.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Third-party imports
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import nltk
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt_tab', quiet=True)


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/v/([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript from YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Full transcript text
        
    Raises:
        Exception: If transcript cannot be fetched
    """
    try:
        print(f"Fetching transcript for video ID: {video_id}")
        
        # Create API instance
        api = YouTubeTranscriptApi()
        
        # Get list of available transcripts
        transcript_list = api.list(video_id)
        
        # Try to find transcript in any available language (prefer Spanish, then English, then any)
        transcript = None
        try:
            # First try Spanish transcripts (manually created, then auto-generated)
            transcript = transcript_list.find_manually_created_transcript(['es', 'es-ES', 'es-MX'])
            print("📝 Using manually created Spanish transcript")
        except:
            try:
                transcript = transcript_list.find_generated_transcript(['es', 'es-ES', 'es-MX'])
                print("🤖 Using auto-generated Spanish transcript")
            except:
                try:
                    # Then try English transcripts
                    transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
                    print("📝 Using manually created English transcript")
                except:
                    try:
                        transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                        print("🤖 Using auto-generated English transcript")
                    except:
                        # Finally, try any available transcript
                        available_transcripts = list(transcript_list)
                        if available_transcripts:
                            transcript = available_transcripts[0]
                            print(f"🌐 Using available transcript in {transcript.language_code}")
                        else:
                            raise Exception("No transcripts available")
        
        # Fetch the actual transcript data
        transcript_data = transcript.fetch()
        
        # Convert to text - handle both dict and object formats
        if isinstance(transcript_data[0], dict):
            transcript_text = " ".join([entry['text'] for entry in transcript_data])
        else:
            # Handle FetchedTranscriptSnippet objects
            transcript_text = " ".join([entry.text for entry in transcript_data])
        
        print(f"✅ Successfully fetched transcript ({len(transcript_text)} characters)")
        return transcript_text
        
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        print("💡 Tip: Make sure the video has captions/subtitles enabled")
        print("💡 This video might not have English captions available")
        raise


def clean_transcript(text: str) -> str:
    """
    Clean and preprocess transcript text.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text
    """
    print("Cleaning transcript...")
    
    # Basic cleaning
    cleaned = text.replace('\n', ' ').strip()
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Split into sentences for better processing
    sentences = nltk.sent_tokenize(cleaned)
    
    # Filter out very short sentences (likely artifacts)
    sentences = [s for s in sentences if len(s.strip()) > 10]
    
    cleaned_text = ' '.join(sentences)
    
    print(f"✅ Cleaned transcript: {len(cleaned_text)} characters, {len(sentences)} sentences")
    return cleaned_text


def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """
    Split text into chunks for processing by language models.
    
    Args:
        text: Text to chunk
        max_length: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences first
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + " " + sentence) <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"✅ Split text into {len(chunks)} chunks")
    return chunks


def setup_models() -> Dict[str, Any]:
    """
    Initialize and return the required models.
    
    Returns:
        Dictionary containing initialized models
    """
    print("Setting up AI models...")
    
    # Get configuration from environment
    hf_token = os.getenv("HF_TOKEN")
    summarizer_model = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    generator_model = os.getenv("GENERATOR_MODEL", "gpt2")
    
    models = {}
    
    try:
        # Initialize summarizer
        print(f"Loading summarization model: {summarizer_model}")
        # Note: This script always uses local transformers pipelines.
        # HF_TOKEN (if present) is ignored here and does not switch to cloud inference.
        print("Using local transformers pipeline (models may download and use local resources)")
        models['summarizer'] = pipeline("summarization", model=summarizer_model)
        
        # Initialize text generator
        print(f"Loading text generation model: {generator_model}")
        # Using local transformers pipeline for generation as well
        print("Using local transformers pipeline (models may download and use local resources)")
        models['generator'] = pipeline("text-generation", model=generator_model)
        
        print("✅ Models loaded successfully")
        return models
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


def summarize_text(text: str, models: Dict[str, Any]) -> str:
    """
    Summarize text using the BART model.
    
    Args:
        text: Text to summarize
        models: Dictionary containing loaded models
        
    Returns:
        Summarized text
    """
    print("Summarizing content...")
    
    # Chunk the text if it's too long
    chunks = chunk_text(text, max_length=1000)
    
    if len(chunks) == 1:
        # Single chunk - summarize directly
        summary = models['summarizer'](
            chunks[0],
            max_length=300,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
    else:
        # Multiple chunks - summarize each then combine
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            chunk_summary = models['summarizer'](
                chunk,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            chunk_summaries.append(chunk_summary)
        
        # Combine chunk summaries
        combined_text = " ".join(chunk_summaries)
        
        # Final summary of combined chunks
        summary = models['summarizer'](
            combined_text,
            max_length=300,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
    
    print(f"✅ Summary generated ({len(summary)} characters)")
    return summary


def generate_linkedin_post(summary: str, models: Dict[str, Any], persona: str = "industrial engineer") -> str:
    """
    Generate a LinkedIn post from the summary.
    
    Args:
        summary: Summarized content
        models: Dictionary containing loaded models
        persona: Persona for the post
        
    Returns:
        Generated LinkedIn post
    """
    print("Generating LinkedIn post...")
    
    # Create a prompt for the post generation
    prompt = f"""Create a professional LinkedIn post based on this finance show summary: '{summary}'

As a {persona}, write an engaging post that:
- Starts with a compelling hook
- Highlights key insights from the show
- Includes your professional perspective
- Ends with a question to encourage discussion
- Keeps it under 300 words
- Uses professional but accessible language

Post:"""
    
    # Generate the post
    generated = models['generator'](
        prompt,
        max_length=400,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=models['generator'].tokenizer.eos_token_id
    )[0]['generated_text']
    
    # Extract just the post part (after "Post:")
    post = generated.split("Post:")[-1].strip()
    
    # Clean up the post
    post = re.sub(r'\n+', '\n', post)  # Remove extra newlines
    post = post.strip()
    
    print(f"✅ LinkedIn post generated ({len(post)} characters)")
    return post


def main():
    """Main function to run the YouTube to LinkedIn post generator."""
    parser = argparse.ArgumentParser(
        description="Convert YouTube finance videos to LinkedIn posts"
    )
    parser.add_argument(
        "--url", 
        required=True, 
        help="YouTube video URL"
    )
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=300,
        help="Maximum words for LinkedIn post (default: 300)"
    )
    parser.add_argument(
        "--persona", 
        default="industrial engineer",
        help="Professional persona for the post (default: industrial engineer)"
    )
    parser.add_argument(
        "--output-file", 
        default="outputs/linkedin_post.txt",
        help="Output file path (default: outputs/linkedin_post.txt)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract video ID
        video_id = extract_video_id(args.url)
        print(f"Video ID: {video_id}")
        
        # Fetch transcript
        transcript = fetch_transcript(video_id)
        
        # Clean transcript
        cleaned_transcript = clean_transcript(transcript)
        
        # Setup models
        models = setup_models()
        
        # Summarize content
        summary = summarize_text(cleaned_transcript, models)
        
        # Generate LinkedIn post
        post = generate_linkedin_post(summary, models, args.persona)
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(post)
        
        print(f"\n🎉 Success! LinkedIn post saved to: {output_path}")
        print(f"\n📝 Generated Post:\n{'-'*50}")
        print(post)
        print(f"{'-'*50}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
