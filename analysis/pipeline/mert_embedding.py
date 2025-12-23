#!/usr/bin/env python3
"""
MERT Audio Embedding Extraction

Uses m-a-p/MERT-v1-95M (HuggingFace) to extract semantic audio embeddings for clustering.
MERT is a state-of-the-art audio representation model trained on music understanding tasks.

Design rationale:
- MERT is used for clustering because it provides strong general-purpose audio representations
- Essentia is kept for interpretation (genre/mood/BPM tags) because it already outputs
  human-readable musical attributes and is fast
- 30s excerpts from center of track (standard practice for music tagging)
- CLS token pooling (recommended by MERT paper)
- L2 normalization for cosine distance in clustering
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MERTEmbeddingExtractor:
    """
    Extracts MERT embeddings from audio files.

    MERT (Music-Understanding Evaluation and Representation Transformer) is a
    self-supervised audio model specifically designed for music understanding tasks.
    """

    def __init__(
        self,
        model_name: str = 'm-a-p/MERT-v1-95M',
        device: Optional[str] = None,
        sample_rate: int = 24000,
        excerpt_seconds: int = 30
    ):
        """
        Initialize MERT model.

        Args:
            model_name: HuggingFace model ID
            device: 'cuda', 'cpu', or None (auto-detect)
            sample_rate: Target sample rate (MERT uses 24kHz)
            excerpt_seconds: Length of audio excerpt to extract
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.excerpt_seconds = excerpt_seconds

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Loading MERT model: {model_name}")
        logger.info(f"Device: {self.device}")

        # Load model and feature extractor
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Use torch.compile for ~30% speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == 'cuda':
            try:
                self.model = torch.compile(self.model)
                logger.info("Enabled torch.compile() for faster inference")
            except Exception as e:
                logger.warning(f"Could not enable torch.compile(): {e}")

        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"MERT embedding dimension: {self.embedding_dim}")

    def preprocess_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform (numpy array) or None if loading fails
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Extract center excerpt
            target_length = self.excerpt_seconds * self.sample_rate

            if len(y) > target_length:
                # Crop from center
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
            elif len(y) < target_length:
                # Pad with zeros
                padding = target_length - len(y)
                y = np.pad(y, (0, padding), mode='constant')

            return y

        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None

    def extract_embedding(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Extract MERT embedding from audio waveform.

        Args:
            audio_waveform: Audio waveform (1D numpy array)

        Returns:
            L2-normalized embedding vector (768-dim for MERT-v1-95M)
        """
        # Process audio
        inputs = self.processor(
            audio_waveform,
            sampling_rate=self.sample_rate,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token (first token of last hidden state)
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]

        # L2 normalization for cosine distance
        embedding = cls_embedding / np.linalg.norm(cls_embedding)

        return embedding.astype(np.float32)

    def extract_batch(self, audio_waveforms: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of audio waveforms.

        Args:
            audio_waveforms: List of audio waveforms

        Returns:
            Batch of L2-normalized embeddings
        """
        # Process batch
        inputs = self.processor(
            audio_waveforms,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token for each item in batch
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()

        # L2 normalization
        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        embeddings = cls_embeddings / norms

        return embeddings.astype(np.float32)


def extract_mert_embeddings(
    master_index_path: str = 'spotify/master_index.json',
    cache_path: str = 'cache/mert_embeddings_24khz_30s_cls.pkl',
    batch_size: int = 8,
    save_interval: int = 50,
    use_cache: bool = True,
    force: bool = False
) -> List[Dict]:
    """
    Extract MERT embeddings for all tracks in master index.

    Args:
        master_index_path: Path to master index JSON
        cache_path: Path to save cache file
        batch_size: Number of files to process per batch (reduce if GPU OOM)
        save_interval: Save cache every N songs (for crash recovery)
        use_cache: Load from cache if available
        force: Force re-extraction even if cache exists

    Returns:
        List of dicts with track metadata and MERT embeddings
    """
    cache_file = Path(cache_path)

    # Load from cache if available
    if use_cache and not force and cache_file.exists():
        logger.info(f"Loading MERT embeddings from cache: {cache_path}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        logger.info(f"Loaded {len(cached_data)} MERT embeddings from cache")
        return cached_data

    # Load master index
    with open(master_index_path, 'r') as f:
        data = json.load(f)
        master_index = data.get('tracks', [])

    logger.info(f"Loaded master index with {len(master_index)} tracks")

    # Initialize extractor
    extractor = MERTEmbeddingExtractor()

    # Load existing cache if available (for incremental updates)
    existing_embeddings = {}
    if cache_file.exists() and not force:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            existing_embeddings = {item['track_id']: item for item in cached_data}
            logger.info(f"Loaded {len(existing_embeddings)} existing embeddings")
        except Exception as e:
            logger.warning(f"Could not load existing cache: {e}")

    # Extract embeddings
    all_embeddings = []
    errors_log = Path('analysis/errors.log')
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    # Prepare batches
    tracks_to_process = []
    for track in master_index:
        track_id = track['track_id']

        # Skip if already cached
        if track_id in existing_embeddings and not force:
            all_embeddings.append(existing_embeddings[track_id])
            continue

        # Check if MP3 file exists
        mp3_file = track.get('mp3_file')
        if not mp3_file or not Path(mp3_file).exists():
            logger.warning(f"MP3 file not found: {mp3_file}")
            with open(errors_log, 'a') as f:
                f.write(f"MERT extraction - MP3 not found: {mp3_file}\n")
            continue

        tracks_to_process.append(track)

    logger.info(f"Processing {len(tracks_to_process)} new tracks")

    # Process in batches
    for i in tqdm(range(0, len(tracks_to_process), batch_size), desc="Extracting MERT embeddings"):
        batch_tracks = tracks_to_process[i:i + batch_size]
        batch_waveforms = []
        batch_metadata = []

        # Load audio for batch
        for track in batch_tracks:
            try:
                audio_waveform = extractor.preprocess_audio(track['mp3_file'])
                if audio_waveform is not None:
                    batch_waveforms.append(audio_waveform)
                    batch_metadata.append(track)
            except Exception as e:
                logger.error(f"Error preprocessing {track['track_name']}: {e}")
                with open(errors_log, 'a') as f:
                    f.write(f"MERT preprocessing error - {track['track_name']}: {e}\n")

        # Extract embeddings for batch
        if batch_waveforms:
            try:
                embeddings = extractor.extract_batch(batch_waveforms)

                # Create result dicts
                for track, embedding in zip(batch_metadata, embeddings):
                    # Get duration
                    try:
                        y, sr = librosa.load(track['mp3_file'], sr=None, duration=0.1)
                        duration = librosa.get_duration(path=track['mp3_file'])
                    except:
                        duration = 0.0

                    result = {
                        'track_id': track['track_id'],
                        'track_name': track['track_name'],
                        'artist': track['artist'],
                        'filename': Path(track['mp3_file']).name,
                        'filepath': track['mp3_file'],
                        'embedding': embedding,
                        'duration_seconds': duration,
                        'model_name': extractor.model_name,
                        'sample_rate': extractor.sample_rate,
                        'excerpt_seconds': extractor.excerpt_seconds
                    }
                    all_embeddings.append(result)

            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory! Reducing batch size...")
                # Retry with batch_size = 1
                for track in batch_metadata:
                    try:
                        audio_waveform = extractor.preprocess_audio(track['mp3_file'])
                        if audio_waveform is not None:
                            embedding = extractor.extract_embedding(audio_waveform)

                            try:
                                duration = librosa.get_duration(path=track['mp3_file'])
                            except:
                                duration = 0.0

                            result = {
                                'track_id': track['track_id'],
                                'track_name': track['track_name'],
                                'artist': track['artist'],
                                'filename': Path(track['mp3_file']).name,
                                'filepath': track['mp3_file'],
                                'embedding': embedding,
                                'duration_seconds': duration,
                                'model_name': extractor.model_name,
                                'sample_rate': extractor.sample_rate,
                                'excerpt_seconds': extractor.excerpt_seconds
                            }
                            all_embeddings.append(result)
                    except Exception as e:
                        logger.error(f"Error extracting {track['track_name']}: {e}")
                        with open(errors_log, 'a') as f:
                            f.write(f"MERT extraction error - {track['track_name']}: {e}\n")

            except Exception as e:
                logger.error(f"Error extracting batch: {e}")
                with open(errors_log, 'a') as f:
                    f.write(f"MERT batch extraction error: {e}\n")

        # Incremental save
        if (i + batch_size) % save_interval == 0:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(all_embeddings, f)
            logger.info(f"Incremental save: {len(all_embeddings)} embeddings")

    # Final save
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(all_embeddings, f)

    logger.info(f"Saved {len(all_embeddings)} MERT embeddings to {cache_path}")

    if len(all_embeddings) > 0:
        logger.info(f"MERT embedding shape: {all_embeddings[0]['embedding'].shape}")
        logger.info(f"MERT embedding dtype: {all_embeddings[0]['embedding'].dtype}")

    return all_embeddings


if __name__ == '__main__':
    """Test MERT extraction on a small subset"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract MERT embeddings')
    parser.add_argument('--force', action='store_true', help='Force re-extraction')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    embeddings = extract_mert_embeddings(
        force=args.force,
        batch_size=args.batch_size
    )

    print(f"\nExtracted {len(embeddings)} embeddings")
    if embeddings:
        print(f"Embedding shape: {embeddings[0]['embedding'].shape}")
        print(f"Sample track: {embeddings[0]['artist']} - {embeddings[0]['track_name']}")
