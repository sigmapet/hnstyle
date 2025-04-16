This repository contains the code used in the following blog post and YouTube videos:

* https://antirez.com/news/150
* https://www.youtube.com/@antirez

## Where to get the HN archive Parquet files:

I downloaded the files from this HF repo. Warning: 10GB of data.

* https://huggingface.co/datasets/OpenPipe/hacker-news

## Original post that inspired this work:

The initial analysis was performed by Christopher Tarry in 2022, then
posted on Hacker News. Unfortunately the web site is now offline, but luckily the Internet Archive have a copy of the site about section and the output for the Paul Graham account:

* https://news.ycombinator.com/item?id=33755016
* https://web.archive.org/web/20221126204005/https://stylometry.net/about
* https://web.archive.org/web/20221126235433/https://stylometry.net/user?username=pg

## Executing the scripts to populate Redis

You need Redis 8 RC or greater in order to have the new data type, the Vector Sets. Once you have an instance running on your computer, do the following.

Generate the top words list. We generate 10k, but we use a lot less later.

    python3 gen-top-words.py dataset/train*.parquet --count 10000 --output-file top.txt

Turn the dataset into a JSONL file with just the username and the frequency talbe. This script will use quite some memory, since it needs to aggregate data by user, and will produce the output only at the end. Note also that this will generate the full frequency table for the user, all the words, not just the top words or the 350 words we use later. This way, the generated file can be used later for different goals and parameters (with 100 or 500 top words for instance).

    python3 gen-freqtab.py dataset/train*.parquet > freqtab.jsonl

Finally we are ready to insert the data into a Redis vector set.

    python3 insert.py --top-words-file top.txt --top-words-count 350 freqtab.jsonl

Now, start the `redis-cli`, and run something like:

    127.0.0.1:6379> vsim hn_fingerprint ele pg
     1) "pg"
     2) "karaterobot"
     3) "Natsu"
     4) "mattmaroon"
     5) "chc"
     6) "montrose"
     7) "jfengel"
     8) "emodendroket"
     9) "vintermann"
    10) "c3534l"

Use the `WITHSCORES` option if you want to see the similarity score.

## Using the vector visualization tool.

This tool will display the vector as graphics in your terminal. You need to use Ghostty, Kitty or any other terminal supporting the Kitty terminal graphics protocol. There are other graphics protocols available, feel free to port this program to other terminals and send a PR, if you wish.

Compile with `make`, then:

    redis-cli vemb hn_fingerprint | ./vshow

