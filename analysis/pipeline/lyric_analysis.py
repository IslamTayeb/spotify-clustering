#!/usr/bin/env python3
"""
Lyric Embedding Extraction

Supports multiple embedding backends:
- bge-m3: BAAI/bge-m3 (1024-dim, default, multilingual, max 8192 tokens)
- e5: intfloat/multilingual-e5-large (1024-dim, higher quality, max 512 tokens)

Design rationale:
- BGE-M3 is default for backward compatibility (longer context, 8192 tokens)
- E5 is optional upgrade for higher quality embeddings (requires instruction prefix)
- Both are multilingual and L2-normalized for cosine distance
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from langdetect import LangDetectException, detect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language of text using langdetect."""
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def get_lyric_model(backend: str = 'bge-m3') -> Tuple[SentenceTransformer, int]:
    """
    Get lyric embedding model based on backend.

    Args:
        backend: 'bge-m3' or 'e5'

    Returns:
        Tuple of (model, max_seq_length)
    """
    if backend == 'bge-m3':
        logger.info("Loading BGE-M3 model (default, 8192 max tokens)...")
        model = SentenceTransformer('BAAI/bge-m3')
        max_seq_length = 8192
    elif backend == 'e5':
        logger.info("Loading E5 model (higher quality, 512 max tokens)...")
        model = SentenceTransformer('intfloat/multilingual-e5-large')
        max_seq_length = 512
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'bge-m3' or 'e5'.")

    model.max_seq_length = max_seq_length
    return model, max_seq_length


def format_text_for_model(text: str, backend: str) -> str:
    """
    Format text for model input.

    E5 models require "passage: " prefix for passage encoding.
    BGE-M3 doesn't need any prefix.

    Args:
        text: Input text
        backend: 'bge-m3' or 'e5'

    Returns:
        Formatted text string
    """
    if backend == 'e5':
        return f"passage: {text}"
    return text


def embed_lyrics(
    text: str,
    model: SentenceTransformer,
    backend: str = 'bge-m3',
    max_length: int = 8192,
    normalize: bool = True
) -> np.ndarray:
    """
    Embed lyrics text with chunking support for long texts.

    Args:
        text: Lyrics text
        model: SentenceTransformer model
        backend: 'bge-m3' or 'e5' (for instruction formatting)
        max_length: Maximum sequence length in tokens
        normalize: L2 normalize embeddings (recommended for cosine distance)

    Returns:
        Embedding vector (L2-normalized if normalize=True)
        Zero vector if text is empty/very short
    """
    embedding_dim = model.get_sentence_embedding_dimension()
    if not text or len(text.strip()) < 10:
        return np.zeros(embedding_dim)

    # Format text for model (E5 needs "passage: " prefix)
    formatted_text = format_text_for_model(text, backend)

    # Check if text needs chunking
    tokens = model.tokenizer.tokenize(text)
    if len(tokens) <= max_length:
        # Single chunk
        embedding = model.encode(formatted_text, convert_to_numpy=True)
    else:
        # Multiple chunks (rare for E5 with 512 max, common for BGE-M3 with 8192)
        chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
        chunk_embeddings = []

        for chunk in chunks:
            chunk_text = model.tokenizer.convert_tokens_to_string(chunk)
            chunk_formatted = format_text_for_model(chunk_text, backend)
            emb = model.encode(chunk_formatted, convert_to_numpy=True)
            chunk_embeddings.append(emb)

        # Average chunk embeddings
        embedding = np.mean(chunk_embeddings, axis=0)

    # L2 normalization for cosine distance
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

    return embedding


def extract_lyric_features_from_file(filepath: str, model: SentenceTransformer) -> Dict:
    try:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()

        embedding = embed_lyrics(text, model)
        language = detect_language(text)
        word_count = len(text.split())
        has_lyrics = len(text.strip()) >= 10

        return {
            'filename': Path(filepath).name,
            'filepath': filepath,
            'embedding': embedding,
            'language': language,
            'word_count': word_count,
            'has_lyrics': has_lyrics
        }

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None


def extract_lyric_features(
    master_index_path: str = 'spotify/master_index.json',
    cache_path: Optional[str] = None,
    backend: str = 'bge-m3',
    batch_size: int = 2,
    use_cache: bool = True
) -> List[Dict]:
    """
    Extract lyric embeddings for all tracks.

    Args:
        master_index_path: Path to master index JSON
        cache_path: Path to save cache file (auto-generated if None)
        backend: 'bge-m3' (default) or 'e5'
        batch_size: Batch size for encoding
        use_cache: Load from cache if available

    Returns:
        List of dicts with track metadata and lyric embeddings
    """
    # Auto-generate cache path based on backend
    if cache_path is None:
        if backend == 'bge-m3':
            cache_path = 'cache/lyric_features.pkl'  # Preserve existing name
        elif backend == 'e5':
            cache_path = 'cache/lyric_features_e5.pkl'
        else:
            cache_path = f'cache/lyric_features_{backend}.pkl'

    # Check cache
    cache_file = Path(cache_path)
    if use_cache and cache_file.exists():
        logger.info(f"Loading lyric features from cache: {cache_path}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        logger.info(f"Loaded {len(cached_data)} lyric features from cache")
        return cached_data

    # Load model
    model, max_seq_length = get_lyric_model(backend)
    embedding_dim = model.get_sentence_embedding_dimension()

    with open(master_index_path, 'r') as f:
        master_index = json.load(f)

    tracks = [track for track in master_index['tracks'] if track.get('mp3_file')]
    logger.info(f"Processing lyrics for {len(tracks)} tracks (with MP3 files)")

    # Separate tracks with and without lyrics for batch processing
    tracks_with_lyrics = []
    tracks_without_lyrics = []
    lyrics_texts = []
    lyrics_metadata = []

    logger.info("Loading lyrics files...")
    for track in tqdm(tracks, desc="Reading lyrics files"):
        lyrics_file = track.get('lyrics_file')

        if not lyrics_file or not Path(lyrics_file).exists():
            tracks_without_lyrics.append(track)
            continue

        try:
            try:
                with open(lyrics_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(lyrics_file, 'r', encoding='latin-1') as f:
                    text = f.read()

            if len(text.strip()) < 10:
                tracks_without_lyrics.append(track)
                continue

            tracks_with_lyrics.append(track)
            lyrics_texts.append(text)
            lyrics_metadata.append({
                'filepath': lyrics_file,
                'filename': Path(lyrics_file).name,
                'word_count': len(text.split()),
                'language': detect_language(text)
            })
        except Exception as e:
            logger.error(f"Error reading {lyrics_file}: {e}")
            tracks_without_lyrics.append(track)

    logger.info(f"Tracks with lyrics: {len(tracks_with_lyrics)}")
    logger.info(f"Tracks without lyrics: {len(tracks_without_lyrics)}")

    features = []

    # Process tracks without lyrics (fast - just add zero vectors)
    for track in tracks_without_lyrics:
        features.append({
            'track_id': track['track_id'],
            'track_name': track['track_name'],
            'artist': track['artist'],
            'filename': track['track_name'],
            'filepath': track.get('lyrics_file'),
            'embedding': np.zeros(embedding_dim, dtype=np.float32),
            'language': 'unknown',
            'word_count': 0,
            'has_lyrics': False,
            'model_backend': backend
        })

    # Batch process tracks with lyrics using the model
    logger.info(f"Encoding {len(lyrics_texts)} lyrics using batch_size={batch_size} ({backend})...")

    # Format texts for model (E5 needs "passage: " prefix)
    formatted_texts = [format_text_for_model(text, backend) for text in lyrics_texts]

    # Encode batch
    embeddings = model.encode(
        formatted_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization for cosine distance
    )

    # Combine embeddings with metadata
    for track, embedding, metadata in zip(tracks_with_lyrics, embeddings, lyrics_metadata):
        features.append({
            'track_id': track['track_id'],
            'track_name': track['track_name'],
            'artist': track['artist'],
            'filename': metadata['filename'],
            'filepath': metadata['filepath'],
            'embedding': embedding,
            'language': metadata['language'],
            'word_count': metadata['word_count'],
            'has_lyrics': True,
            'model_backend': backend  # Track which model was used
        })

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)

    logger.info(f"Successfully extracted lyric features from {len(features)} tracks using {backend}")
    logger.info(f"Cache saved to: {cache_path}")
    with_lyrics_count = sum(1 for f in features if f['has_lyrics'])
    without_lyrics_count = len(features) - with_lyrics_count
    logger.info(f"Tracks with lyrics: {with_lyrics_count}/{len(features)}")
    logger.info(f"Tracks without lyrics: {without_lyrics_count}/{len(features)}")

    if features:
        logger.info(f"Embedding shape: {features[0]['embedding'].shape}")
        logger.info(f"Embedding dtype: {features[0]['embedding'].dtype}")

    return features


if __name__ == '__main__':
    features = extract_lyric_features()
    print(f"\nExtracted {len(features)} lyric features")
    print(f"\nSample feature keys: {list(features[0].keys())}")
