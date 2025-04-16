import pandas as pd
import json
import sys
import re
import urllib.parse
import argparse
import multiprocessing as mp
from functools import partial
from html import unescape
from bs4 import BeautifulSoup
import pyarrow.parquet as pq
import os
import time
from itertools import islice

# Pre-compile regex patterns for better performance
URL_PATTERN = re.compile(r'\[(https?://[^\s\]]+)\]|\b(https?://[^\s]+)\b')
HTML_PATTERN = re.compile(r'<[a-zA-Z]+[^>]*>|&[a-zA-Z]+;')
WORD_PATTERN = re.compile(r'\b[a-zA-Z][a-zA-Z\'-]*[a-zA-Z]\b|\b[a-zA-Z]\b')
MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')

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
        parsed_url = urllib.parse.urlparse(url)

        # Get domain (remove www. if present)
        domain = parsed_url.netloc
        domain = re.sub(r'^www\.', '', domain)

        return domain
    except Exception:
        # If URL parsing fails, return the original string as fallback
        return url

def create_frequency_table(text):
    """Create a frequency table of ALL words in the given text."""
    if pd.isna(text) or not text:
        return {}

    # Initialize frequency table
    freq_table = {}

    # Function to replace URLs with their domain representation
    def replace_url(match):
        url = match.group(1) or match.group(2)
        domain = extract_domain(url)
        token = f"linkto:{domain}"

        # Add to frequency table
        freq_table[token] = freq_table.get(token, 0) + 1

        return f" {token} "

    # Replace URLs with tokens
    text = URL_PATTERN.sub(replace_url, text)

    # Tokenize the text - split by whitespace and filter out punctuation
    words = WORD_PATTERN.findall(text.lower())

    # Count word frequencies - no filtering by stopwords or vocabulary
    for word in words:
        if len(word) > 0:  # Keep all words, even single characters
            freq_table[word] = freq_table.get(word, 0) + 1

    return freq_table

def clean_html(html_text):
    """Clean HTML from text, preserving paragraph structure and link information."""
    if pd.isna(html_text):
        return None

    # If the text is not a string (somehow), convert it to string
    if not isinstance(html_text, str):
        return str(html_text)

    # Unescape HTML entities
    unescaped = unescape(html_text)

    # Only use BeautifulSoup if the text actually contains HTML-like patterns.
    # This speeds up computation since BS can be slow.
    if HTML_PATTERN.search(unescaped):
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
    cleaned_text = MULTI_NEWLINE_PATTERN.sub('\n\n', cleaned_text)

    return cleaned_text

def merge_frequency_tables(table1, table2):
    """Merge two frequency tables, summing counts for common words."""
    merged = table1.copy()
    for word, count in table2.items():
        merged[word] = merged.get(word, 0) + count
    return merged

def process_parquet_file(file_path):
    """Process a single parquet file and return user frequency tables."""
    try:
        worker_id = mp.current_process().name
        print(f"[{worker_id}] Processing file: {file_path}", file=sys.stderr)
        start_time = time.time()

        # Local frequency table for this file
        local_user_freqtab = {}

        try:
            # Open the parquet file and read only the columns we need
            parquet_file = pq.ParquetFile(file_path)

            # Check if we can use predicate pushdown
            schema = parquet_file.schema
            has_type_column = 'type' in schema.names

            # Process the file in batches (row groups)
            try:
                # First try with filters (newer PyArrow versions)
                if has_type_column:
                    batches = parquet_file.iter_batches(
                        batch_size=1000,
                        columns=['by', 'text', 'type'],
                        filters=[('type', '=', 'comment')]
                    )
                else:
                    batches = parquet_file.iter_batches(
                        batch_size=1000,
                        columns=['by', 'text']
                    )
            except TypeError:
                # Fallback for older PyArrow versions without filter support
                print(f"[{worker_id}] Filter not supported, falling back to manual filtering", file=sys.stderr)
                if has_type_column:
                    batches = parquet_file.iter_batches(
                        batch_size=1000,
                        columns=['by', 'text', 'type']
                    )
                else:
                    batches = parquet_file.iter_batches(
                        batch_size=1000,
                        columns=['by', 'text']
                    )

            for batch in batches:
                # Convert batch to pandas DataFrame
                batch_df = batch.to_pandas()

                # Filter to only include comments if we couldn't use predicate pushdown
                if not has_type_column or 'type' not in batch_df.columns:
                    # If 'type' column doesn't exist, assume all are comments
                    comments_df = batch_df
                else:
                    comments_df = batch_df[batch_df['type'] == 'comment'].copy()

                # Skip if no comments in this batch
                if comments_df.empty:
                    continue

                # Process each comment
                for _, row in comments_df.iterrows():
                    # Skip if no user or text
                    if pd.isna(row['by']) or pd.isna(row['text']):
                        continue

                    username = row['by']

                    # Clean the text
                    cleaned_text = clean_html(row['text'])

                    # Generate frequency table for this comment - no vocabulary or stopwords filtering
                    comment_freqtab = create_frequency_table(cleaned_text)

                    # If this is the first comment from this user, initialize their frequency table
                    if username not in local_user_freqtab:
                        local_user_freqtab[username] = comment_freqtab
                    else:
                        # Merge this comment's frequency table with the user's existing one
                        local_user_freqtab[username] = merge_frequency_tables(
                            local_user_freqtab[username], comment_freqtab
                        )

        except Exception as e:
            print(f"[{worker_id}] Error reading file '{file_path}': {e}", file=sys.stderr)
            return {}

        elapsed = time.time() - start_time
        print(f"[{worker_id}] Finished processing file: {file_path} in {elapsed:.2f} seconds", file=sys.stderr)
        print(f"[{worker_id}] Found {len(local_user_freqtab)} users in this file", file=sys.stderr)

        return local_user_freqtab

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}", file=sys.stderr)
        return {}

def worker_init():
    """Initialize worker process."""
    # Silence excessive logging in worker processes
    import logging
    logging.basicConfig(level=logging.WARNING)

def merge_worker_results(results):
    """Merge frequency tables from multiple workers."""
    merged_freqtab = {}

    # Merge user frequency tables from all files
    for user_freqtab in results:
        for username, freqtab in user_freqtab.items():
            if username not in merged_freqtab:
                merged_freqtab[username] = freqtab
            else:
                merged_freqtab[username] = merge_frequency_tables(
                    merged_freqtab[username], freqtab
                )

    return merged_freqtab

def chunks(data, size):
    """Split data into chunks of specified size."""
    it = iter(data)
    for i in range(0, len(data), size):
        yield list(islice(it, size))

def write_results_to_file(user_freqtab, output_file=None):
    """Write results to file or stdout, sorting frequency tables by most frequent words."""
    output = output_file if output_file else sys.stdout

    try:
        with open(output, 'w', encoding='utf-8') if isinstance(output, str) else output as f:
            for username, freqtab in user_freqtab.items():
                # Sort the frequency table by count (most frequent first)
                sorted_freqtab = dict(sorted(freqtab.items(), key=lambda x: x[1], reverse=True))

                # Create a record with just username and sorted frequency table
                record = {
                    "by": username,
                    "freqtab": sorted_freqtab
                }
                # Convert to JSON and write
                json_string = json.dumps(record)
                f.write(json_string + '\n')
    except Exception as e:
        print(f"Error writing results: {e}", file=sys.stderr)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate frequency tables from parquet files in parallel.')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                        help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--chunk-size', type=int, default=1,
                        help='Number of files per worker (default: 1)')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('files', nargs='+', help='Parquet files to process')

    # Parse arguments
    args = parser.parse_args()

    # Calculate the actual number of workers to use (min of files and requested workers)
    num_chunks = (len(args.files) + args.chunk_size - 1) // args.chunk_size
    num_workers = min(args.workers, num_chunks)

    print(f"Starting with {num_workers} worker processes", file=sys.stderr)
    print(f"Processing {len(args.files)} files in chunks of {args.chunk_size}", file=sys.stderr)

    # Start the timer
    start_time = time.time()

    # Group files into chunks for better load balancing
    file_chunks = list(chunks(args.files, args.chunk_size))

    # Create a pool of worker processes
    with mp.Pool(processes=num_workers, initializer=worker_init) as pool:
        # Create a partial function with fixed parameters
        process_func = partial(process_chunk)

        # Process chunks in parallel
        results = pool.map(process_func, file_chunks)

    # Merge results from all workers
    merged_freqtab = merge_worker_results(results)

    # Calculate elapsed time
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds", file=sys.stderr)
    print(f"Processed {len(args.files)} files", file=sys.stderr)
    print(f"Found data for {len(merged_freqtab)} unique users", file=sys.stderr)

    # Write the results
    write_results_to_file(merged_freqtab, args.output)

def process_chunk(file_paths):
    """Process a chunk of files and return combined frequency tables."""
    chunk_results = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}", file=sys.stderr)
            continue

        file_results = process_parquet_file(file_path)

        # Merge this file's results into the chunk results
        for username, freqtab in file_results.items():
            if username not in chunk_results:
                chunk_results[username] = freqtab
            else:
                chunk_results[username] = merge_frequency_tables(
                    chunk_results[username], freqtab
                )

    return chunk_results

if __name__ == "__main__":
    main()
