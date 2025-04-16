#!/usr/bin/env python3
import json
import redis
import numpy as np
import sys
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Global dictionary to map words to vector indices
word_to_index = {}

# Global dictionaries for word statistics
word_counts = defaultdict(int)     # How many users use this word
word_freq_sum = defaultdict(float) # Sum of frequencies for calculating mean
word_freq_sq_sum = defaultdict(float) # Sum of squared frequencies for variance
total_users = 0                    # Total number of users processed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process user frequency tables and store them as vectors in Redis')

    # Required arguments
    parser.add_argument('filename', help='JSONL file containing user frequency tables')
    parser.add_argument('--top-words-file', required=True, help='File containing top words, one per line')
    parser.add_argument('--top-words-count', type=int, required=True, help='Number of top words to use')

    # Optional arguments with defaults
    parser.add_argument('--host', default='localhost', help='Redis host (default: localhost)')
    parser.add_argument('--port', type=int, default=6379, help='Redis port (default: 6379)')
    parser.add_argument('--keyname', default='hn_fingerprints', help='Redis key to store vectors (default: hn_fingerprints)')
    parser.add_argument('--suffix', default='', help='String to append to usernames, useful to add the same users with different names and check if the style detection works (default: empty string, nothing added)')

    return parser.parse_args()

def load_top_words(filename: str, count: int) -> List[str]:
    """
    Load the top words from a file, limited by count
    Returns a list of top words (ordered to maintain consistent vector indices)
    """
    global word_to_index

    top_words = []
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i >= count:
                    break
                word = line.strip()
                if word:
                    top_words.append(word)
                    # Create mapping from word to its index in the vector
                    word_to_index[word] = i

        print(f"Loaded {len(top_words)} top words from {filename}")
        return top_words
    except Exception as e:
        print(f"Error loading top words file: {e}")
        sys.exit(1)

def parse_frequency_table(json_data: str) -> Tuple[str, Dict[str, int]]:
    """
    Parse the frequency table from JSON data
    Returns the username and a dictionary of word frequencies
    """
    try:
        data = json.loads(json_data)
        username = data.get("by", "unknown")
        freq_table = data.get("freqtab", {})
        return username, freq_table
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Problematic data: {json_data[:100]}...")
        raise

def calculate_global_statistics(filename: str, top_words: List[str]):
    """
    First pass: Calculate global statistics (mean, stddev) for each word
    """
    global word_counts, word_freq_sum, word_freq_sq_sum, total_users

    print("First pass: Calculating global word statistics...")

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    _, freq_table = parse_frequency_table(line)

                    # Calculate total words in this document for relative frequencies
                    total_words = sum(freq_table.values())
                    if total_words == 0:
                        continue

                    # Update statistics for words in top_words
                    for word, count in freq_table.items():
                        if word in top_words:
                            # Convert to relative frequency (percentage)
                            rel_freq = count / total_words

                            # Update statistics
                            word_counts[word] += 1
                            word_freq_sum[word] += rel_freq
                            word_freq_sq_sum[word] += rel_freq * rel_freq

                    total_users += 1

                    if line_num % 1000 == 0:
                        print(f"Processed {line_num} users for statistics...")

                except Exception as e:
                    print(f"Error processing line {line_num} for statistics: {e}")
                    continue

        print(f"Completed statistics calculation for {total_users} users")

        # Calculate means and standard deviations
        word_means = {}
        word_stddevs = {}

        for word in top_words:
            if word_counts[word] > 1:  # Need at least 2 occurrences for stddev
                mean = word_freq_sum[word] / total_users
                # Calculate variance using the computational formula: Var(X) = E(X²) - E(X)²
                variance = (word_freq_sq_sum[word] / total_users) - (mean * mean)
                # Use Bessel's correction for sample standard deviation.
                # You may wonder why to use Bessel's correction if we have all
                # the HN users. Well, the code assumes that this is still a
                # sample of a much larger theoretical population.
                corrected_variance = variance * total_users / (total_users - 1) if total_users > 1 else variance
                stddev = max(np.sqrt(corrected_variance), 1e-6)  # Avoid division by zero

                word_means[word] = mean
                word_stddevs[word] = stddev
            else:
                # If a word doesn't occur enough, use fallback values
                word_means[word] = 0.0
                word_stddevs[word] = 1.0  # Avoid division by zero

        print(f"Calculated statistics for {len(word_means)} words")
        return word_means, word_stddevs

    except Exception as e:
        print(f"Error calculating global statistics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_user_vector(freq_table: Dict[str, int], top_words: List[str],
                      word_means: Dict[str, float], word_stddevs: Dict[str, float]) -> Tuple[np.ndarray, int]:
    """
    Create a user vector based on the frequency table using Burrows' Delta
    standardization. Returns the vector and the total word count.
    """
    # Initialize a zero vector with dimension equal to the number of top words
    vector = np.zeros(len(top_words), dtype=np.float32)

    # Calculate total words for relative frequencies and for metadata
    total_words = sum(freq_table.values())
    if total_words == 0:
        return vector, 0

    # Count words that are in our top words list (for metadata)
    top_words_count = 0

    # Process each word in the frequency table, but only if it's in top_words
    for word, frequency in freq_table.items():
        # Skip invalid words or frequency values
        if not isinstance(word, str) or not word:
            continue
        if not isinstance(frequency, (int, float)) or frequency <= 0:
            continue

        # Skip words not in our top words list
        if word not in word_to_index:
            continue

        # Add to the count of words in our list
        top_words_count += frequency

        # Convert to relative frequency
        rel_freq = frequency / total_words

        # Standardize using z-score: z = (freq - mean) / stddev
        mean = word_means.get(word, 0.0)
        stddev = word_stddevs.get(word, 1.0)  # Default to 1.0 to avoid division by zero

        z_score = (rel_freq - mean) / stddev

        # Set the z-score directly in the vector at the word's index
        vector[word_to_index[word]] = z_score

    return vector, top_words_count

def add_user_to_redis(r: redis.Redis, username: str, vector: np.ndarray, word_count: int, redis_key: str, suffix: str) -> bool:
    """
    Add a user vector to Redis using VADD and set attributes with word count
    """
    try:
        # Convert the numpy array to bytes in FP32 format
        vector_bytes = vector.astype(np.float32).tobytes()

        # Use VADD with FP32 to add the vector
        metadata = json.dumps({"wordcount": word_count})
        result = r.execute_command(
            'VADD',
            redis_key,
            'FP32',
            vector_bytes,
            username+suffix,
            "SETATTR",
            metadata
        )
        return result == 1
    except Exception as e:
        print(f"Error adding user {username} to Redis: {e}")
        return False

def process_file(filename: str, redis_conn: redis.Redis, redis_key: str, top_words: List[str], word_means: Dict[str, float], word_stddevs: Dict[str, float], suffix: str) -> int:
    """
    Process a file containing user frequency tables in JSONL format
    Returns the number of successful additions to Redis
    """
    success_count = 0

    try:
        print(f"Second pass: Processing JSONL file to create vectors: {filename}")
        # Read the file line by line to handle JSONL format
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse the JSON object in this line
                    username, freq_table = parse_frequency_table(line)

                    # Count how many words are in the top words list
                    common_words = set(freq_table.keys()).intersection(word_to_index.keys())

                    # Print some stats about the user
                    word_count = len(freq_table)
                    common_count = len(common_words)

                    if line_num % 100 == 0:
                        print(f"Line {line_num}: User '{username}' with {word_count} unique words ({common_count} in top words list)")

                    # Create the vector fingerprint using Burrows' Delta standardization
                    vector, top_words_count = create_user_vector(freq_table, top_words, word_means, word_stddevs)

                    # Add to Redis with word count metadata
                    if add_user_to_redis(redis_conn, username, vector, top_words_count, redis_key, suffix):
                        success_count += 1
                        if line_num % 100 == 0:
                            print(f"Successfully added user: {username} (word count: {top_words_count})")
                    else:
                        print(f"Failed to add user: {username}")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing user at line {line_num}: {e}")
                    continue

                # Print progress periodically
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines, added {success_count} users so far...")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        import traceback
        traceback.print_exc()

    return success_count

def main():
    # Parse command line arguments
    args = parse_args()

    # Load top words from file
    top_words = load_top_words(args.top_words_file, args.top_words_count)

    if not top_words:
        print("No top words loaded, exiting.")
        sys.exit(1)

    print("=" * 50)
    print(f"User Fingerprinting with Redis Vector Sets (Burrows' Delta)")
    print("=" * 50)
    print(f"File: {args.filename}")
    print(f"Redis: {args.host}:{args.port}")
    print(f"Redis key: {args.keyname}")
    print(f"Top words file: {args.top_words_file}")
    print(f"Top words count: {args.top_words_count} (loaded {len(top_words)})")
    print(f"Vector dimension: {len(top_words)}")
    print("=" * 50)

    # Connect to Redis
    try:
        r = redis.Redis(host=args.host, port=args.port, decode_responses=False)
        r.ping()  # Check connection

        # Verify Vector Sets are available
        try:
            r.execute_command('VCARD', args.keyname)
        except redis.exceptions.ResponseError as e:
            # If key doesn't exist, that's fine
            if 'key does not exist' not in str(e).lower():
                raise

        print("Connected to Redis successfully")

    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        print("Make sure Redis is running and the Vector Sets module is loaded")
        sys.exit(1)

    # FIRST PASS: Calculate global statistics
    word_means, word_stddevs = calculate_global_statistics(args.filename, top_words)

    # SECOND PASS: Process the file and create vectors
    success_count = process_file(args.filename, r, args.keyname, top_words, word_means, word_stddevs, args.suffix)

    if success_count > 0:
        print("\nSummary:")
        print(f"Successfully added {success_count} users to Redis Vector Set")
        print(f"Total words in the vector space: {len(word_to_index)}")

        # Print the total number of vectors in the set
        try:
            total = r.execute_command('VCARD', args.keyname)
            print(f"Total vectors in {args.keyname}: {total}")

            # Example of how to use VSIM with FILTER
            print("\nExample search with word count filter:")
            print("To find similar users with at least 1000 words:")
            print(f"VSIM {args.keyname} <username> 20 FILTER \".wordcount < 5000\"")
        except Exception as e:
            print(f"Unable to get total vector count: {e}")
    else:
        print("\nNo users were added to the Redis Vector Set")

    print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
