#!/usr/bin/env python3
"""
YouTube Finance Show to LinkedIn Post Generator - API Version

Uses Hugging Face Inference API to avoid local model downloads.
"""

import os
import sys
import argparse
import re
import requests
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Third-party imports
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from dotenv import load_dotenv
from tqdm import tqdm

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
    """Fetch transcript from YouTube video."""
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
    """Clean and preprocess transcript text."""
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
    """Split text into chunks for processing."""
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
    
    # Merge short chunks with previous chunk
    merged_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 50 and merged_chunks:  # If chunk is too short and we have previous chunks
            # Merge with previous chunk
            merged_chunks[-1] += " " + chunk
        else:
            merged_chunks.append(chunk)
    
    print(f"✅ Split text into {len(chunks)} chunks, merged to {len(merged_chunks)} chunks")
    return merged_chunks


def call_hf_api(text: str, task: str, model: str, hf_token: str, progress_bar=None) -> str:
    """Call Hugging Face Inference API with progress tracking."""
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Truncate text to safe length for API
    max_input_length = 1000 if task == "summarization" else 2000
    if len(text) > max_input_length:
        text = text[:max_input_length]
        if progress_bar:
            progress_bar.set_description(f"⚠️ Truncated text to {max_input_length} chars")
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 200 if task == "summarization" else 300,
            "min_length": 50 if task == "summarization" else 30,
            "do_sample": True,
            "temperature": 0.7
        }
    }
    
    if progress_bar:
        progress_bar.set_description(f"🤖 Calling {model} API...")
    
    # Make the API call with timeout
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            if progress_bar:
                progress_bar.set_description("⏳ Model is loading, please wait...")
            time.sleep(10)
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if task == "summarization":
                return result[0].get('summary_text', '')
            else:
                return result[0].get('generated_text', '')
        
        return ""
        
    except requests.exceptions.Timeout:
        raise Exception("API call timed out. The model might be overloaded. Please try again later.")
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")


def summarize_text(text: str, hf_token: str, debug_chunks: bool = False) -> str:
    """Summarize text using Hugging Face API."""
    print("\n🔄 Summarizing content using Hugging Face API...")
    
    # Chunk the text if it's too long (much smaller chunks for API)
    chunks = chunk_text(text, max_length=300)
    
    # Save chunks for debugging (if enabled)
    if debug_chunks:
        chunks_output_path = Path("outputs/chunks_debug.txt")
        chunks_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            f.write(f"TOTAL CHUNKS: {len(chunks)}\n")
            f.write("=" * 50 + "\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1}/{len(chunks)} (Length: {len(chunk)} chars)\n")
                f.write("-" * 30 + "\n")
                f.write(chunk)
                f.write("\n\n" + "=" * 50 + "\n\n")
        
        print(f"📄 Chunks saved to: {chunks_output_path}")
    
    print(f"📊 Chunk stats: {len(chunks)} chunks, avg length: {sum(len(c) for c in chunks) // len(chunks)} chars")
    
    if len(chunks) == 1:
        # Single chunk - summarize directly
        with tqdm(total=1, desc="📝 Summarizing", unit="chunk") as pbar:
            summary = call_hf_api(
                chunks[0], 
                "summarization", 
                "facebook/bart-large-cnn", 
                hf_token,
                pbar
            )
            pbar.update(1)
    else:
        # Multiple chunks - summarize each then combine
        chunk_summaries = []
        with tqdm(total=len(chunks) + 1, desc="📝 Summarizing", unit="chunk") as pbar:
            for i, chunk in enumerate(chunks):
                pbar.set_description(f"📝 Summarizing chunk {i+1}/{len(chunks)}")
                chunk_summary = call_hf_api(
                    chunk, 
                    "summarization", 
                    "facebook/bart-large-cnn", 
                    hf_token,
                    pbar
                )
                chunk_summaries.append(chunk_summary)
                pbar.update(1)
            
            # Combine chunk summaries
            combined_text = " ".join(chunk_summaries)
            
            # Final summary of combined chunks
            pbar.set_description("📝 Creating final summary")
            summary = call_hf_api(
                combined_text, 
                "summarization", 
                "facebook/bart-large-cnn", 
                hf_token,
                pbar
            )
            pbar.update(1)
    
    print(f"✅ Summary generated ({len(summary)} characters)")
    return summary


def generate_linkedin_post(summary: str, hf_token: str, persona: str = "startup founder") -> str:
    """Generate a LinkedIn post from the summary using Hugging Face API."""
    print("\n🔄 Generating LinkedIn post using Hugging Face API...")
    
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
    
    # Generate the post using HF API with progress bar
    with tqdm(total=1, desc="✍️ Generating LinkedIn post", unit="post") as pbar:
        generated = call_hf_api(
            prompt, 
            "text-generation", 
            "distilgpt2", 
            hf_token,
            pbar
        )
        pbar.update(1)
    
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
        description="Convert YouTube finance videos to LinkedIn posts using HF API"
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
        default="startup founder",
        help="Professional persona for the post (default: startup founder)"
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
    parser.add_argument(
        "--debug-chunks", 
        action="store_true",
        help="Save chunks to file for debugging"
    )
    
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in .env file")
        print("💡 Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    try:
        # Create output directory
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Overall progress tracking
        print("🚀 Starting YouTube to LinkedIn Post Generator")
        print("=" * 60)
        
        with tqdm(total=5, desc="🎯 Overall Progress", unit="step") as main_pbar:
            # Extract video ID
            main_pbar.set_description("🔍 Extracting video ID")
            video_id = extract_video_id(args.url)
            print(f"Video ID: {video_id}")
            main_pbar.update(1)
            
            # Fetch transcript
            main_pbar.set_description("📺 Fetching transcript")
            transcript = fetch_transcript(video_id)
            main_pbar.update(1)
            
            # Clean transcript
            main_pbar.set_description("🧹 Cleaning transcript")
            cleaned_transcript = clean_transcript(transcript)
            main_pbar.update(1)
            
            # Summarize content
            main_pbar.set_description("📝 Summarizing content")
            summary = summarize_text(cleaned_transcript, hf_token, args.debug_chunks)
            main_pbar.update(1)
            
            # Generate LinkedIn post
            main_pbar.set_description("✍️ Generating LinkedIn post")
            post = generate_linkedin_post(summary, hf_token, args.persona)
            main_pbar.update(1)
        
        # Save output
        print("\n💾 Saving output...")
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
