#!/usr/bin/env python3
import argparse
import re
import sys
from collections import Counter
from html import unescape
from urllib.parse import urlparse

import pyarrow.parquet as pq
import pandas as pd
from bs4 import BeautifulSoup

def extract_domain(url):
    """Extract domain name from a URL."""
    if not url:
        return None

    try:
        # Handle special cases like relative URLs or malformed URLs
        if not url.startswith(('http://', 'https://')):
            if url.startswith('//'):
                url = 'http:' + url
            else:
                url = 'http://' + url

        # Parse URL
        parsed_url = urlparse(url)

        # Get domain (remove www. if present)
        domain = parsed_url.netloc
        domain = re.sub(r'^www\.', '', domain)

        return domain
    except Exception:
        # If URL parsing fails, return the original string as fallback
        return url

def clean_html(html_text):
    """Clean HTML from text, preserving paragraph structure and link information."""
    if pd.isna(html_text):
        return None

    # If the text is not a string (somehow), convert it to string
    if not isinstance(html_text, str):
        return str(html_text)

    # Unescape HTML entities
    unescaped = unescape(html_text)

    # Only use BeautifulSoup if the text actually contains HTML-like patterns
    if re.search(r'<[a-zA-Z]+[^>]*>|&[a-zA-Z]+;', unescaped):
        # Use BeautifulSoup to parse and extract text
        soup = BeautifulSoup(unescaped, 'html.parser')

        # Replace <p> tags with newlines
        for p in soup.find_all('p'):
            p.replace_with('\n\n' + p.get_text() + '\n\n')

        # Extract text from links while preserving the URL
        for a in soup.find_all('a'):
            href = a.get('href', '')
            a.replace_with(f"{a.get_text()} [{href}]")

        cleaned_text = soup.get_text().strip()
    else:
        # If no HTML detected, just use the text as is
        cleaned_text = unescaped

    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    return cleaned_text

def extract_words(text):
    """Extract words and URL domains from text."""
    if pd.isna(text) or not text:
        return []

    words = []

    # Find URLs and extract domains
    url_pattern = r'\[(https?://[^\s\]]+)\]|\b(https?://[^\s]+)\b'

    # Function to extract domains from URLs
    def extract_url_domain(match):
        url = match.group(1) or match.group(2)
        domain = extract_domain(url)
        if domain:
            words.append(f"linkto:{domain}")
        return " "  # Replace URL with space

    # Replace URLs with their domain tokens and extract domains
    processed_text = re.sub(url_pattern, extract_url_domain, text)

    # Tokenize the text - split by whitespace and filter out punctuation
    # Keep apostrophes and hyphens within words
    extracted_words = re.findall(r'\b[a-zA-Z][a-zA-Z\'-]*[a-zA-Z]\b|\b[a-zA-Z]\b', processed_text.lower())

    # Remove common stop words
    # stop_words = {'the', 'and', 'a', 'to', 'of', 'in', 'that', 'is', 'for', 'it', 'as', 'with', 'on', 'by', 'at', 'this', 'an', 'are', 'be', 'or', 'was', 'but', 'not', 'have', 'from', 'has', 'had', 'will', 'they', 'what', 'which', 'who', 'when', 'where', 'how', 'why', 'their', 'there', 'these', 'those', 'been', 'being', 'would', 'could', 'should', 'you', 'your', 'i', 'my', 'me', 'we', 'us', 'our'}
    stop_words = {}

    # Add words that pass the filter
    for word in extracted_words:
        if word not in stop_words and len(word) > 1:
            words.append(word)

    return words

def process_parquet_file(file_path, word_counter):
    """Process a single parquet file and update global word counter."""
    try:
        print(f"Processing file: {file_path}", file=sys.stderr)

        # Open the parquet file
        parquet_file = pq.ParquetFile(file_path)

        # Process the file in batches (row groups)
        for batch in parquet_file.iter_batches(batch_size=1000):
            # Convert batch to pandas DataFrame
            batch_df = batch.to_pandas()

            # Filter to only include comments
            comments_df = batch_df[batch_df['type'] == 'comment'].copy()

            # Skip if no comments in this batch
            if comments_df.empty:
                continue

            # Process each comment
            for index, row in comments_df.iterrows():
                # Skip if no text
                if pd.isna(row['text']):
                    continue

                # Clean the text
                cleaned_text = clean_html(row['text'])

                # Extract words
                words = extract_words(cleaned_text)

                # Update word counter
                word_counter.update(words)

        print(f"Finished processing file: {file_path}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}", file=sys.stderr)
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a list of the most frequent words from parquet files.')
    parser.add_argument('files', nargs='+', help='Parquet files to process')
    parser.add_argument('--count', type=int, default=10000, help='Number of top words to output')
    parser.add_argument('--output-file', required=True, help='File to write the top words')

    # Parse arguments
    args = parser.parse_args()

    # Initialize counter for words
    word_counter = Counter()

    # Count of successfully processed files
    processed_count = 0

    # Process each parquet file
    for file_path in args.files:
        try:
            if process_parquet_file(file_path, word_counter):
                processed_count += 1
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"An error occurred processing file '{file_path}': {e}. Skipping.", file=sys.stderr)

    # Report processing summary
    print(f"Successfully processed {processed_count} of {len(args.files)} files.", file=sys.stderr)
    print(f"Found {len(word_counter)} unique words.", file=sys.stderr)

    # Get the top N words
    top_words = [word for word, count in word_counter.most_common(args.count)]

    # Write to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for word in top_words:
            f.write(f"{word}\n")

    print(f"Wrote top {len(top_words)} words to {args.output_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
