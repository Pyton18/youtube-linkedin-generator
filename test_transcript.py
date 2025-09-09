#!/usr/bin/env python3
"""
Test script to fetch and save YouTube transcript for manual verification
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Third-party imports
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
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
        return transcript_text, transcript.language_code
        
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


def main():
    """Test script to fetch and save transcript."""
    parser = argparse.ArgumentParser(description="Test YouTube transcript fetching")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output", default="outputs/transcript_test.txt", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("🧪 Testing YouTube Transcript Fetching")
        print("=" * 50)
        
        # Extract video ID
        video_id = extract_video_id(args.url)
        print(f"Video ID: {video_id}")
        
        # Fetch transcript
        transcript, language = fetch_transcript(video_id)
        
        # Clean transcript
        cleaned_transcript = clean_transcript(transcript)
        
        # Save raw transcript
        raw_output = output_path.parent / "raw_transcript.txt"
        with open(raw_output, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Save cleaned transcript
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
        
        print(f"\n✅ Success! Transcripts saved:")
        print(f"📄 Raw transcript: {raw_output}")
        print(f"📄 Cleaned transcript: {output_path}")
        print(f"🌐 Language: {language}")
        print(f"📊 Stats: {len(cleaned_transcript)} characters, {len(cleaned_transcript.split())} words")
        
        # Show first 500 characters as preview
        print(f"\n📖 Preview (first 500 characters):")
        print("-" * 50)
        print(cleaned_transcript[:500] + "..." if len(cleaned_transcript) > 500 else cleaned_transcript)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
