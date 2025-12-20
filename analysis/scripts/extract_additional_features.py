#!/usr/bin/env python3
"""
Extract additional features from cached embeddings.

This script loads cached audio features (which contain embeddings),
runs new Essentia models on those embeddings, and merges the results
back into the cache. Much faster than re-processing MP3 files.
"""

import pickle
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from essentia.standard import TensorflowPredict2D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdditionalFeatureExtractor:
    """Extracts additional features from cached discogs-effnet embeddings"""

    def __init__(self):
        logger.info("Initializing new Essentia models...")
        models_dir = Path.home() / '.essentia' / 'models'

        # Approachability models
        self.approachability_2c = TensorflowPredict2D(
            graphFilename=str(models_dir / 'approachability_2c-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )
        self.approachability_3c = TensorflowPredict2D(
            graphFilename=str(models_dir / 'approachability_3c-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )
        self.approachability_regression = TensorflowPredict2D(
            graphFilename=str(models_dir / 'approachability_regression-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Identity"
        )

        # Engagement models
        self.engagement_2c = TensorflowPredict2D(
            graphFilename=str(models_dir / 'engagement_2c-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )
        self.engagement_3c = TensorflowPredict2D(
            graphFilename=str(models_dir / 'engagement_3c-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )
        self.engagement_regression = TensorflowPredict2D(
            graphFilename=str(models_dir / 'engagement_regression-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Identity"
        )

        # MTG-Jamendo multi-label
        self.mtg_jamendo = TensorflowPredict2D(
            graphFilename=str(models_dir / 'mtg_jamendo_moodtheme-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Sigmoid"
        )

        logger.info("Models initialized successfully")

    def extract_from_embedding(self, embedding: np.ndarray) -> dict:
        """
        Extract features from a cached embedding vector.

        Args:
            embedding: 1D array of shape (512,) - cached discogs-effnet embedding

        Returns:
            dict with new features
        """
        # Reshape embedding to (1, 512) for model input
        # Models expect (time_steps, features), so we add a time dimension
        embeddings_2d = embedding.reshape(1, -1)

        # Approachability extraction
        approachability_2c_probs = self.approachability_2c(embeddings_2d).mean(axis=0)
        approachability_3c_probs = self.approachability_3c(embeddings_2d).mean(axis=0)
        approachability_regression_score = self.approachability_regression(embeddings_2d).mean()

        # Engagement extraction
        engagement_2c_probs = self.engagement_2c(embeddings_2d).mean(axis=0)
        engagement_3c_probs = self.engagement_3c(embeddings_2d).mean(axis=0)
        engagement_regression_score = self.engagement_regression(embeddings_2d).mean()

        # MTG-Jamendo multi-label extraction (56 classes)
        mtg_jamendo_probs = self.mtg_jamendo(embeddings_2d).mean(axis=0)

        return {
            # Approachability features
            'approachability_2c_accessible': float(approachability_2c_probs[0]),
            'approachability_2c_niche': float(approachability_2c_probs[1]),
            'approachability_3c_probs': approachability_3c_probs.tolist(),
            'approachability_score': float(approachability_regression_score),

            # Engagement features
            'engagement_2c_low': float(engagement_2c_probs[0]),
            'engagement_2c_high': float(engagement_2c_probs[1]),
            'engagement_3c_probs': engagement_3c_probs.tolist(),
            'engagement_score': float(engagement_regression_score),

            # MTG-Jamendo mood/theme (56-dimensional probability vector)
            'mtg_jamendo_probs': mtg_jamendo_probs.tolist(),
        }


def main():
    cache_path = 'cache/audio_features.pkl'

    print("=" * 70)
    print("EXTRACT ADDITIONAL FEATURES FROM CACHED EMBEDDINGS")
    print("=" * 70)

    # Load cached features
    logger.info(f"Loading cached features from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cached_features = pickle.load(f)

    logger.info(f"Loaded {len(cached_features)} cached feature sets")

    # Initialize extractor
    extractor = AdditionalFeatureExtractor()

    # Extract new features from cached embeddings
    logger.info("Extracting additional features from cached embeddings...")
    updated_features = []

    for feature_dict in tqdm(cached_features, desc="Processing embeddings"):
        # Get cached embedding
        embedding = feature_dict['embedding']

        # Extract new features
        new_features = extractor.extract_from_embedding(embedding)

        # Merge new features into existing dict
        updated_dict = {**feature_dict, **new_features}
        updated_features.append(updated_dict)

    # Save updated cache
    logger.info(f"Saving updated features to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(updated_features, f)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\nUpdated {len(updated_features)} feature sets")
    print(f"Cache saved to: {cache_path}")
    print("\nNew features added:")
    print("  - Approachability (4 features)")
    print("  - Engagement (4 features)")
    print("  - MTG-Jamendo mood/theme (56-dim vector)")
    print(f"\nTotal new features: ~64 per track")

    logger.info("Additional feature extraction complete!")


if __name__ == '__main__':
    main()
