#!/usr/bin/env python3
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from langdetect import LangDetectException, detect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def embed_lyrics(text: str, model: SentenceTransformer, max_length: int = 8192) -> np.ndarray:
    """
    Embed lyrics text. Returns zero vector if text is empty/very short.
    Note: Zero vector does NOT mean instrumental - use audio voice/instrumental classification.
    """
    embedding_dim = model.get_sentence_embedding_dimension()
    if not text or len(text.strip()) < 10:
        return np.zeros(embedding_dim)

    tokens = model.tokenizer.tokenize(text)
    if len(tokens) <= max_length:
        return model.encode(text, convert_to_numpy=True)

    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    chunk_embeddings = []

    for chunk in chunks:
        chunk_text = model.tokenizer.convert_tokens_to_string(chunk)
        emb = model.encode(chunk_text, convert_to_numpy=True)
        chunk_embeddings.append(emb)

    return np.mean(chunk_embeddings, axis=0)


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


def extract_lyric_features(master_index_path: str = 'spotify/master_index.json',
                          cache_path: str = 'cache/lyric_features.pkl',
                          batch_size: int = 2) -> List[Dict]:

    logger.info("Loading sentence-transformers model...")
    model = SentenceTransformer('BAAI/bge-m3')
    model.max_seq_length = 8192
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
            'embedding': np.zeros(embedding_dim),
            'language': 'unknown',
            'word_count': 0,
            'has_lyrics': False
        })

    # Batch process tracks with lyrics using the model
    logger.info(f"Encoding {len(lyrics_texts)} lyrics using batch_size={batch_size}...")
    embeddings = model.encode(
        lyrics_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
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
            'has_lyrics': True
        })

    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)

    logger.info(f"Successfully extracted lyric features from {len(features)} tracks")
    with_lyrics_count = sum(1 for f in features if f['has_lyrics'])
    without_lyrics_count = len(features) - with_lyrics_count
    logger.info(f"Tracks with lyrics: {with_lyrics_count}/{len(features)}")
    logger.info(f"Tracks without lyrics: {without_lyrics_count}/{len(features)}")

    return features


if __name__ == '__main__':
    features = extract_lyric_features()
    print(f"\nExtracted {len(features)} lyric features")
    print(f"\nSample feature keys: {list(features[0].keys())}")
