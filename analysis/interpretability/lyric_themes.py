"""
Lyric Theme Analysis Module

Extract keywords, themes, sentiment, and complexity metrics from lyrics using:
- TF-IDF for keyword extraction
- VADER for sentiment analysis
- textstat for readability metrics
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)


def load_lyrics_for_cluster(
    df: pd.DataFrame,
    cluster_id: int,
    lyrics_dir: str = 'lyrics/temp/',
) -> List[Tuple[str, str]]:
    """
    Load all lyric text files for songs in cluster.

    Args:
        df: DataFrame with cluster assignments and filenames
        cluster_id: ID of the cluster
        lyrics_dir: Directory containing lyric text files

    Returns:
        List of tuples: (track_name, lyrics_text)
        Only includes songs where lyrics were successfully loaded.
    """
    lyrics_path = Path(lyrics_dir)
    cluster_df = df[df['cluster'] == cluster_id]

    lyrics_data = []
    missing_count = 0

    for _, row in cluster_df.iterrows():
        track_name = row.get('track_name', 'Unknown')
        artist = row.get('artist', 'Unknown')
        filename = row.get('filename', '')

        if not filename:
            missing_count += 1
            continue

        # Convert .mp3 filename to .txt
        lyric_filename = filename.replace('.mp3', '.txt')
        lyric_file = lyrics_path / lyric_filename

        if not lyric_file.exists():
            missing_count += 1
            continue

        try:
            with open(lyric_file, 'r', encoding='utf-8') as f:
                lyrics_text = f.read().strip()

            if lyrics_text:
                lyrics_data.append((f"{artist} - {track_name}", lyrics_text))

        except UnicodeDecodeError:
            try:
                with open(lyric_file, 'r', encoding='latin-1') as f:
                    lyrics_text = f.read().strip()

                if lyrics_text:
                    lyrics_data.append((f"{artist} - {track_name}", lyrics_text))
            except Exception as e:
                logger.warning(f"Failed to read lyrics for {track_name}: {e}")
                missing_count += 1

        except Exception as e:
            logger.warning(f"Failed to read lyrics for {track_name}: {e}")
            missing_count += 1

    if missing_count > 0:
        logger.info(f"Cluster {cluster_id}: Loaded {len(lyrics_data)} lyrics, {missing_count} missing")

    return lyrics_data


def extract_tfidf_keywords(
    all_lyrics: List[str],
    cluster_lyrics: List[str],
    top_n: int = 30,
    ngram_range: Tuple[int, int] = (1, 3),
) -> Dict:
    """
    Extract top keywords for cluster using TF-IDF.

    TF-IDF identifies words that are frequent in this cluster but rare globally.

    Args:
        all_lyrics: All lyrics in dataset (for IDF calculation)
        cluster_lyrics: Lyrics in this cluster only
        top_n: Number of keywords to return
        ngram_range: (min_n, max_n) for n-grams, e.g., (1,3) for unigrams, bigrams, trigrams

    Returns:
        dict with:
        - unigrams: [(word, score), ...]
        - bigrams: [(phrase, score), ...]
        - trigrams: [(phrase, score), ...]
        - coverage_pct: % of cluster with lyrics
    """
    if not cluster_lyrics:
        return {
            'unigrams': [],
            'bigrams': [],
            'trigrams': [],
            'coverage_pct': 0.0
        }

    # Fit TF-IDF on all lyrics
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        min_df=2,  # Must appear in at least 2 documents
    )

    try:
        vectorizer.fit(all_lyrics)

        # Transform cluster lyrics
        cluster_tfidf = vectorizer.transform(cluster_lyrics)

        # Average TF-IDF scores across all songs in cluster
        avg_scores = np.array(cluster_tfidf.mean(axis=0)).flatten()

        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        word_scores = list(zip(feature_names, avg_scores))

        # Sort by score descending
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)

        # Separate into unigrams, bigrams, trigrams
        unigrams = [(word, score) for word, score in word_scores if len(word.split()) == 1][:top_n]
        bigrams = [(word, score) for word, score in word_scores if len(word.split()) == 2][:top_n]
        trigrams = [(word, score) for word, score in word_scores if len(word.split()) == 3][:top_n]

        return {
            'unigrams': unigrams,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'coverage_pct': 100.0,  # Assume all input lyrics are valid
        }

    except Exception as e:
        logger.error(f"TF-IDF extraction failed: {e}")
        return {
            'unigrams': [],
            'bigrams': [],
            'trigrams': [],
            'coverage_pct': 0.0
        }


def analyze_sentiment(lyrics_text: str) -> Dict:
    """
    Analyze sentiment of lyrics using VADER sentiment analyzer.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    designed for social media and short texts, works well for song lyrics.

    Args:
        lyrics_text: String of lyrics

    Returns:
        dict with:
        - compound_score: Overall sentiment (-1 to 1)
        - positive: Positive sentiment score (0 to 1)
        - negative: Negative sentiment score (0 to 1)
        - neutral: Neutral sentiment score (0 to 1)
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(lyrics_text)

        return {
            'compound_score': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
        }

    except ImportError:
        logger.warning("vaderSentiment not installed, using fallback TextBlob")
        try:
            from textblob import TextBlob
            blob = TextBlob(lyrics_text)
            polarity = blob.sentiment.polarity  # -1 to 1

            # Convert to VADER-like format
            return {
                'compound_score': polarity,
                'positive': max(0, polarity),
                'negative': max(0, -polarity),
                'neutral': 1 - abs(polarity),
            }

        except ImportError:
            logger.error("Neither vaderSentiment nor TextBlob installed")
            return {
                'compound_score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
            }


def compute_lyric_complexity(lyrics_text: str) -> Dict:
    """
    Compute complexity metrics for lyrics.

    Args:
        lyrics_text: String of lyrics

    Returns:
        dict with:
        - word_count: Total number of words
        - unique_words: Number of unique words
        - vocabulary_richness: unique_words / word_count (0-1)
        - avg_word_length: Average characters per word
        - flesch_reading_ease: Readability score (0-100, higher = easier)
        - compression_ratio: Measure of repetitiveness (lower = more repetitive)
    """
    # Clean and tokenize
    words = re.findall(r'\b\w+\b', lyrics_text.lower())

    if not words:
        return {
            'word_count': 0,
            'unique_words': 0,
            'vocabulary_richness': 0.0,
            'avg_word_length': 0.0,
            'flesch_reading_ease': 0.0,
            'compression_ratio': 0.0,
        }

    word_count = len(words)
    unique_words = len(set(words))
    vocabulary_richness = unique_words / word_count if word_count > 0 else 0.0
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    # Flesch Reading Ease using textstat
    try:
        import textstat
        flesch_score = textstat.flesch_reading_ease(lyrics_text)
    except ImportError:
        logger.warning("textstat not installed, skipping readability score")
        flesch_score = 0.0

    # Compression ratio: unique_chars / total_chars (measure of repetitiveness)
    chars = lyrics_text.replace(' ', '').replace('\n', '')
    unique_chars = len(set(chars))
    total_chars = len(chars)
    compression_ratio = unique_chars / total_chars if total_chars > 0 else 0.0

    return {
        'word_count': word_count,
        'unique_words': unique_words,
        'vocabulary_richness': vocabulary_richness,
        'avg_word_length': avg_word_length,
        'flesch_reading_ease': flesch_score,
        'compression_ratio': compression_ratio,
    }


def extract_common_phrases(cluster_lyrics: List[str], top_n: int = 20) -> List[Tuple[str, int]]:
    """
    Extract most common phrases using frequency analysis.

    This is different from TF-IDF: it finds phrases that are repeated often
    within this cluster, useful for identifying choruses and repeated lines.

    Args:
        cluster_lyrics: List of lyric strings for cluster
        top_n: Number of top phrases to return

    Returns:
        List of (phrase, count) tuples
    """
    if not cluster_lyrics:
        return []

    # Combine all lyrics
    all_text = ' '.join(cluster_lyrics).lower()

    # Extract 3-5 word phrases
    words = re.findall(r'\b\w+\b', all_text)

    phrase_counts = Counter()

    # Generate 3-grams, 4-grams, 5-grams
    for n in [3, 4, 5]:
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            phrase_counts[phrase] += 1

    # Return top phrases (that appear more than once)
    common_phrases = [(phrase, count) for phrase, count in phrase_counts.items() if count > 1]
    common_phrases = sorted(common_phrases, key=lambda x: x[1], reverse=True)

    return common_phrases[:top_n]


def compare_cluster_keywords(
    cluster_a_keywords: Dict,
    cluster_b_keywords: Dict,
) -> Dict:
    """
    Find unique and shared keywords between two clusters.

    Args:
        cluster_a_keywords: Keywords dict from extract_tfidf_keywords for cluster A
        cluster_b_keywords: Keywords dict from extract_tfidf_keywords for cluster B

    Returns:
        dict with 'unique_to_a', 'unique_to_b', 'shared' lists
    """
    # Extract unigrams only for comparison
    words_a = set(word for word, score in cluster_a_keywords.get('unigrams', []))
    words_b = set(word for word, score in cluster_b_keywords.get('unigrams', []))

    unique_to_a = list(words_a - words_b)
    unique_to_b = list(words_b - words_a)
    shared = list(words_a & words_b)

    return {
        'unique_to_a': unique_to_a,
        'unique_to_b': unique_to_b,
        'shared': shared,
    }


def topic_modeling_lda(
    cluster_lyrics: List[str],
    n_topics: int = 3,
    top_words: int = 10,
) -> List[Dict]:
    """
    Discover sub-themes within cluster using Latent Dirichlet Allocation (LDA).

    Only recommended for large clusters (100+ songs with lyrics).

    Args:
        cluster_lyrics: List of lyric strings for cluster
        n_topics: Number of topics to discover
        top_words: Number of top words per topic

    Returns:
        List of topic dicts, each with 'topic_id' and 'top_words'
    """
    if len(cluster_lyrics) < 20:
        logger.warning(f"Too few lyrics ({len(cluster_lyrics)}) for LDA, need at least 20")
        return []

    try:
        # TF-IDF vectorization (LDA works better with TF or TF-IDF than raw counts)
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            min_df=2,
        )

        tfidf_matrix = vectorizer.fit_transform(cluster_lyrics)
        feature_names = vectorizer.get_feature_names_out()

        # LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
        )

        lda.fit(tfidf_matrix)

        # Extract top words for each topic
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-top_words:][::-1]
            top_words_list = [feature_names[i] for i in top_indices]

            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words_list,
            })

        return topics

    except Exception as e:
        logger.error(f"LDA topic modeling failed: {e}")
        return []
