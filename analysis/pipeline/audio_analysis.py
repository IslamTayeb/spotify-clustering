#!/usr/bin/env python3
import json
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import essentia.standard as es
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ESSENTIA_MODELS = {
    'embeddings': 'discogs-effnet-bs64-1',
    'genre': 'genre_discogs400-discogs-effnet-1',
    'mood_happy': 'mood_happy-discogs-effnet-1',
    'mood_sad': 'mood_sad-discogs-effnet-1',
    'mood_aggressive': 'mood_aggressive-discogs-effnet-1',
    'mood_relaxed': 'mood_relaxed-discogs-effnet-1',
    'mood_party': 'mood_party-discogs-effnet-1',
    'arousal': 'deam-msd-musicnn-2',
    'valence': 'deam-msd-musicnn-2',
    'danceability': 'danceability-discogs-effnet-1',
    'voice_instrumental': 'voice_instrumental-discogs-effnet-1',
}

GENRE_LABELS = {}

def _load_labels_from_json(filepath: Path) -> List[str]:
    """Helper to load labels from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_genre_labels(model_name: str = 'discogs400') -> List[str]:
    """Load genre labels for a specific model, with caching."""
    global GENRE_LABELS
    if model_name not in GENRE_LABELS:
        if model_name == 'discogs400':
            filepath = Path(__file__).parent.parent / 'data' / 'genre_discogs400_labels.json'
            if not filepath.exists():
                logger.error(f"Discogs 400 genre labels file not found: {filepath}. Falling back to generic labels.")
                GENRE_LABELS[model_name] = [f"genre_{i}" for i in range(400)]
            else:
                GENRE_LABELS[model_name] = _load_labels_from_json(filepath)
        elif model_name == 'mtg_jamendo_genre':
            filepath = Path(__file__).parent.parent / 'data' / 'mtg_jamendo_genre_labels.json'
            if not filepath.exists():
                logger.error(f"MTG-Jamendo genre labels file not found: {filepath}. Falling back to generic labels.")
                GENRE_LABELS[model_name] = [f"mtg_jamendo_genre_{i}" for i in range(87)]
            else:
                GENRE_LABELS[model_name] = _load_labels_from_json(filepath)
        else:
            logger.warning(f"Unknown genre model: {model_name}. Returning empty list.")
            GENRE_LABELS[model_name] = []
    return GENRE_LABELS[model_name]


class AudioFeatureExtractor:
    def __init__(self):
        logger.info("Initializing Essentia models...")
        from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
        from essentia.standard import TensorflowPredictMusiCNN
        from essentia.standard import RhythmExtractor2013, KeyExtractor

        models_dir = Path.home() / '.essentia' / 'models'

        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(models_dir / 'discogs-effnet-bs64-1.pb'),
            output="PartitionedCall:1"
        )

        self.genre_model = TensorflowPredict2D(
            graphFilename=str(models_dir / 'genre_discogs400-discogs-effnet-1.pb'),
            input="serving_default_model_Placeholder",
            output="PartitionedCall"
        )

        self.mood_models = {
            'mood_happy': TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_happy-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            ),
            'mood_sad': TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_sad-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            ),
            'mood_aggressive': TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_aggressive-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            ),
            'mood_relaxed': TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_relaxed-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            ),
            'mood_party': TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_party-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            ),
        }

        self.danceability_model = TensorflowPredict2D(
            graphFilename=str(models_dir / 'danceability-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )
        self.voice_model = TensorflowPredict2D(
            graphFilename=str(models_dir / 'voice_instrumental-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Softmax"
        )

        # NEW: Approachability models (mainstream vs niche)
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

        # NEW: Engagement models (active vs background listening)
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

        # NEW: MTG-Jamendo multi-label mood/theme (56 classes)
        self.mtg_jamendo = TensorflowPredict2D(
            graphFilename=str(models_dir / 'mtg_jamendo_moodtheme-discogs-effnet-1.pb'),
            input="model/Placeholder", output="model/Sigmoid"
        )

        # NEW: MusiCNN embedding extractor (needed for MusiCNN-based models)
        try:
            self.musicnn_embedding_model = TensorflowPredictMusiCNN(
                graphFilename=str(models_dir / 'msd-musicnn-1.pb'),
                output="model/dense/BiasAdd"
            )
        except Exception as e:
            logger.warning(f"Could not load MusiCNN embedding model: {e}")
            self.musicnn_embedding_model = None

        # NEW: Valence/Arousal model (uses MusiCNN embeddings)
        try:
            self.valence_arousal_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'deam-msd-musicnn-2.pb'),
                input="model/Placeholder",
                output="model/Identity"
            )
        except Exception as e:
            logger.warning(f"Could not load valence/arousal model: {e}")
            self.valence_arousal_model = None

        # NEW (2024): Voice Gender model (male vs female vocals)
        try:
            self.voice_gender_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'gender-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            )
        except Exception as e:
            logger.warning(f"Could not load voice gender model: {e}")
            self.voice_gender_model = None

        # NEW (2024): Timbre model (bright vs dark)
        try:
            self.timbre_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'timbre-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            )
        except Exception as e:
            logger.warning(f"Could not load timbre model: {e}")
            self.timbre_model = None

        # NEW (2024): Acoustic vs Electronic model
        try:
            self.mood_acoustic_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'mood_acoustic-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Softmax"
            )
        except Exception as e:
            logger.warning(f"Could not load acoustic/electronic model: {e}")
            self.mood_acoustic_model = None

        # NEW (2024): MTG-Jamendo Instrument (40 classes)
        try:
            self.mtg_jamendo_instrument = TensorflowPredict2D(
                graphFilename=str(models_dir / 'mtg_jamendo_instrument-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Sigmoid"
            )
        except Exception as e:
            logger.warning(f"Could not load instrument model: {e}")
            self.mtg_jamendo_instrument = None

        # NEW (2024): Alternative Arousal/Valence (emoMusic dataset, uses MusiCNN embeddings)
        try:
            self.emomusic_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'emomusic-msd-musicnn-2.pb'),
                input="model/Placeholder",
                output="model/Identity"
            )
        except Exception as e:
            logger.warning(f"Could not load emoMusic model: {e}")
            self.emomusic_model = None

        # NEW (2024): MTG-Jamendo Genre (87 classes)
        try:
            self.mtg_jamendo_genre = TensorflowPredict2D(
                graphFilename=str(models_dir / 'mtg_jamendo_genre-discogs-effnet-1.pb'),
                input="model/Placeholder", output="model/Sigmoid"
            )
        except Exception as e:
            logger.warning(f"Could not load MTG-Jamendo genre model: {e}")
            self.mtg_jamendo_genre = None

        # NEW (2024): MIREX Moods (5 mood clusters, uses MusiCNN embeddings)
        try:
            self.moods_mirex_model = TensorflowPredict2D(
                graphFilename=str(models_dir / 'moods_mirex-msd-musicnn-1.pb'),
                input="serving_default_model_Placeholder",
                output="PartitionedCall"
            )
        except Exception as e:
            logger.warning(f"Could not load MIREX moods model: {e}")
            self.moods_mirex_model = None

        self.rhythm_extractor = RhythmExtractor2013(method="multifeature")
        self.key_extractor = KeyExtractor()

        logger.info("Models initialized successfully")

    def extract(self, filepath: str) -> Dict:
        try:
            loader = es.MonoLoader(filename=filepath, sampleRate=16000)
            audio = loader()

            if len(audio) < 16000:
                logger.warning(f"Audio too short: {filepath}")
                return None

            embeddings = self.embedding_model(audio)
            genre_probs = self.genre_model(embeddings).mean(axis=0)

            moods = {name: model(embeddings).mean(axis=0)[1] for name, model in self.mood_models.items()}
            danceability = self.danceability_model(embeddings).mean(axis=0)[1]

            voice_probs = self.voice_model(embeddings).mean(axis=0)
            instrumentalness = voice_probs[0]

            # NEW: Approachability extraction
            approachability_2c_probs = self.approachability_2c(embeddings).mean(axis=0)
            approachability_3c_probs = self.approachability_3c(embeddings).mean(axis=0)
            approachability_regression_score = self.approachability_regression(embeddings).mean()

            # NEW: Engagement extraction
            engagement_2c_probs = self.engagement_2c(embeddings).mean(axis=0)
            engagement_3c_probs = self.engagement_3c(embeddings).mean(axis=0)
            engagement_regression_score = self.engagement_regression(embeddings).mean()

            # NEW: MTG-Jamendo multi-label extraction (56 classes)
            mtg_jamendo_probs = self.mtg_jamendo(embeddings).mean(axis=0)

            # NEW: MusiCNN embeddings extraction (for MusiCNN-based models)
            musicnn_embeddings = None
            if self.musicnn_embedding_model:
                musicnn_embeddings = self.musicnn_embedding_model(audio)

            # NEW: Valence/Arousal extraction (uses MusiCNN embeddings)
            if not self.valence_arousal_model:
                raise RuntimeError("Valence/arousal model not loaded - cannot extract features")
            if musicnn_embeddings is None:
                raise RuntimeError("MusiCNN embeddings not available - cannot extract valence/arousal")
            va_preds = self.valence_arousal_model(musicnn_embeddings)
            mean_preds = va_preds.mean(axis=0)
            valence = float(mean_preds[0])
            arousal = float(mean_preds[1])

            # NEW (2024): Voice Gender extraction
            if not self.voice_gender_model:
                raise RuntimeError("Voice gender model not loaded - cannot extract features")
            gender_probs = self.voice_gender_model(embeddings).mean(axis=0)
            # For instrumental songs, set both to 0.0 (no voice present)
            if instrumentalness >= 0.5:
                voice_gender_female = 0.0
                voice_gender_male = 0.0
            else:
                voice_gender_female = float(gender_probs[0])
                voice_gender_male = float(gender_probs[1])

            # NEW (2024): Timbre extraction (bright vs dark)
            if not self.timbre_model:
                raise RuntimeError("Timbre model not loaded - cannot extract features")
            timbre_probs = self.timbre_model(embeddings).mean(axis=0)
            timbre_bright = float(timbre_probs[0])
            timbre_dark = float(timbre_probs[1])

            # NEW (2024): Acoustic vs Electronic
            if not self.mood_acoustic_model:
                raise RuntimeError("Acoustic/electronic model not loaded - cannot extract features")
            acoustic_probs = self.mood_acoustic_model(embeddings).mean(axis=0)
            mood_acoustic = float(acoustic_probs[0])
            mood_electronic = float(acoustic_probs[1])

            # NEW (2024): MTG-Jamendo Instruments (40 classes)
            if not self.mtg_jamendo_instrument:
                raise RuntimeError("MTG-Jamendo instrument model not loaded - cannot extract features")
            mtg_jamendo_instrument_probs = self.mtg_jamendo_instrument(embeddings).mean(axis=0).tolist()

            # NEW (2024): Alternative Arousal/Valence (emoMusic, uses MusiCNN embeddings)
            if not self.emomusic_model:
                raise RuntimeError("EmoMusic model not loaded - cannot extract features")
            if musicnn_embeddings is None:
                raise RuntimeError("MusiCNN embeddings not available - cannot extract emoMusic features")
            emo_preds = self.emomusic_model(musicnn_embeddings)
            mean_emo = emo_preds.mean(axis=0)
            arousal_emomusic = float(mean_emo[0])
            valence_emomusic = float(mean_emo[1])

            # NEW (2024): MTG-Jamendo Genre (87 classes)
            if not self.mtg_jamendo_genre:
                raise RuntimeError("MTG-Jamendo genre model not loaded - cannot extract features")
            mtg_jamendo_genre_probs = self.mtg_jamendo_genre(embeddings).mean(axis=0).tolist()

            # NEW (2024): MIREX Moods (5 clusters, uses MusiCNN embeddings)
            if not self.moods_mirex_model:
                raise RuntimeError("MIREX moods model not loaded - cannot extract features")
            if musicnn_embeddings is None:
                raise RuntimeError("MusiCNN embeddings not available - cannot extract MIREX moods")
            moods_mirex_probs = self.moods_mirex_model(musicnn_embeddings).mean(axis=0).tolist()

            bpm, _, _, _, _ = self.rhythm_extractor(audio)
            key, scale, _ = self.key_extractor(audio)

            genre_labels = load_genre_labels('discogs400')
            top_3_indices = np.argsort(genre_probs)[-3:][::-1]
            top_3_genres = [(genre_labels[i], float(genre_probs[i])) for i in top_3_indices]

            # Process MTG-Jamendo Genre probabilities
            mtg_jamendo_genre_labels = load_genre_labels('mtg_jamendo_genre')
            top_mtg_jamendo_genre_indices = np.argsort(mtg_jamendo_genre_probs)[-3:][::-1]
            top_mtg_jamendo_genres = [(mtg_jamendo_genre_labels[i], float(mtg_jamendo_genre_probs[i])) for i in top_mtg_jamendo_genre_indices]

            return {
                'filename': Path(filepath).name,
                'filepath': filepath,
                'embedding': embeddings.mean(axis=0),
                'genre_probs': genre_probs,
                'top_3_genres': top_3_genres,
                'mood_happy': float(moods['mood_happy']),
                'mood_sad': float(moods['mood_sad']),
                'mood_aggressive': float(moods['mood_aggressive']),
                'mood_relaxed': float(moods['mood_relaxed']),
                'mood_party': float(moods['mood_party']),
                'danceability': float(danceability),
                'instrumentalness': float(instrumentalness),
                'bpm': float(bpm),
                'key': f"{key} {scale}",
                'duration_seconds': float(len(audio) / 16000),

                'valence': valence,
                'arousal': arousal,

                # NEW: Approachability features
                'approachability_2c_accessible': float(approachability_2c_probs[0]),
                'approachability_2c_niche': float(approachability_2c_probs[1]),
                'approachability_3c_probs': approachability_3c_probs.tolist(),
                'approachability_score': float(approachability_regression_score),

                # NEW: Engagement features
                'engagement_2c_low': float(engagement_2c_probs[0]),
                'engagement_2c_high': float(engagement_2c_probs[1]),
                'engagement_3c_probs': engagement_3c_probs.tolist(),
                'engagement_score': float(engagement_regression_score),

                # NEW: MTG-Jamendo mood/theme (56-dimensional probability vector)
                'mtg_jamendo_probs': mtg_jamendo_probs.tolist(),

                # NEW (2024): Voice Gender
                'voice_gender_female': voice_gender_female,
                'voice_gender_male': voice_gender_male,

                # NEW (2024): Timbre
                'timbre_bright': timbre_bright,
                'timbre_dark': timbre_dark,

                # NEW (2024): Acoustic vs Electronic
                'mood_acoustic': mood_acoustic,
                'mood_electronic': mood_electronic,

                # NEW (2024): MTG-Jamendo Instruments
                'mtg_jamendo_instrument_probs': mtg_jamendo_instrument_probs,

                # NEW (2024): Alternative Arousal/Valence
                'arousal_emomusic': arousal_emomusic,
                'valence_emomusic': valence_emomusic,

                # NEW (2024): MTG-Jamendo Genre
                'mtg_jamendo_genre_probs': mtg_jamendo_genre_probs,
                'top_mtg_jamendo_genres': top_mtg_jamendo_genres,

                # NEW (2024): MIREX Moods
                'moods_mirex_probs': moods_mirex_probs,
            }

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None


def update_cached_features(cache_path: str = 'cache/audio_features.pkl') -> List[Dict]:
    """
    Update existing cached features with missing fields (valence, arousal, new models).
    Loads audio but skips re-calculating the main embedding if possible.
    """
    if not Path(cache_path).exists():
        logger.error(f"Cache file not found: {cache_path}")
        return []

    with open(cache_path, 'rb') as f:
        features = pickle.load(f)

    logger.info(f"Updating {len(features)} cached features with missing classifiers...")

    extractor = AudioFeatureExtractor()
    updated_features = []

    for feature in tqdm(features, desc="Updating audio features"):
        # Check if we need to update this track - check ALL possible fields
        needs_update = (
            # Basic rhythm/key features
            'bpm' not in feature or
            'key' not in feature or
            'instrumentalness' not in feature or

            # Genre features
            'genre_probs' not in feature or
            'top_3_genres' not in feature or

            # Mood features
            'mood_happy' not in feature or
            'mood_sad' not in feature or
            'mood_aggressive' not in feature or
            'mood_relaxed' not in feature or
            'mood_party' not in feature or

            # Basic audio features
            'danceability' not in feature or

            # Advanced emotional features
            'valence' not in feature or
            'arousal' not in feature or

            # Model features (2023)
            'approachability_score' not in feature or
            'engagement_score' not in feature or
            'mtg_jamendo_probs' not in feature or

            # NEW (2024): Voice & Production features
            'voice_gender_female' not in feature or
            'voice_gender_male' not in feature or
            'timbre_bright' not in feature or
            'timbre_dark' not in feature or
            'mood_acoustic' not in feature or
            'mood_electronic' not in feature or

            # NEW (2024): Advanced classifiers
            'mtg_jamendo_instrument_probs' not in feature or
            'arousal_emomusic' not in feature or
            'valence_emomusic' not in feature or
            'mtg_jamendo_genre_probs' not in feature or
            'top_mtg_jamendo_genres' not in feature or
            'moods_mirex_probs' not in feature
        )

        if not needs_update:
            updated_features.append(feature)
            continue

        try:
            filepath = feature['filepath']
            if not Path(filepath).exists():
                logger.warning(f"File not found during update: {filepath}")
                updated_features.append(feature)
                continue

            # Load audio
            loader = es.MonoLoader(filename=filepath, sampleRate=16000)
            audio = loader()

            # Recalculate embeddings since we need them for the new classifiers
            embeddings = extractor.embedding_model(audio)

            # Extract MusiCNN embeddings if needed for MusiCNN-based models
            musicnn_embeddings = None
            if extractor.musicnn_embedding_model and (
                'valence' not in feature or 'arousal_emomusic' not in feature or 'moods_mirex_probs' not in feature
            ):
                musicnn_embeddings = extractor.musicnn_embedding_model(audio)

            # 0. Basic rhythm/key features (should always be present, but check anyway)
            if 'bpm' not in feature or 'key' not in feature:
                bpm, _, _, _, _ = extractor.rhythm_extractor(audio)
                key, scale, _ = extractor.key_extractor(audio)
                feature['bpm'] = float(bpm)
                feature['key'] = f"{key} {scale}"

            # 1. Instrumentalness (basic feature)
            if 'instrumentalness' not in feature:
                voice_probs = extractor.voice_model(embeddings).mean(axis=0)
                feature['instrumentalness'] = float(voice_probs[0])

            # 2. Genre features (should always be present)
            if 'genre_probs' not in feature or 'top_3_genres' not in feature:
                genre_probs = extractor.genre_model(embeddings).mean(axis=0)
                feature['genre_probs'] = genre_probs

                genre_labels = load_genre_labels()
                top_3_indices = np.argsort(genre_probs)[-3:][::-1]
                feature['top_3_genres'] = [(genre_labels[i], float(genre_probs[i])) for i in top_3_indices]

            # 3. Danceability
            if 'danceability' not in feature:
                feature['danceability'] = float(extractor.danceability_model(embeddings).mean(axis=0)[1])

            # 4. Mood features
            if 'mood_happy' not in feature:
                mood_results = {name: model(embeddings).mean(axis=0)[1] for name, model in extractor.mood_models.items()}
                feature['mood_happy'] = float(mood_results['mood_happy'])
                feature['mood_sad'] = float(mood_results['mood_sad'])
                feature['mood_aggressive'] = float(mood_results['mood_aggressive'])
                feature['mood_relaxed'] = float(mood_results['mood_relaxed'])
                feature['mood_party'] = float(mood_results['mood_party'])

            # 5. Valence/Arousal (Requires MusiCNN embeddings)
            if 'valence' not in feature or 'arousal' not in feature:
                if not extractor.valence_arousal_model:
                    raise RuntimeError("Valence/arousal model not loaded - cannot extract features")
                if musicnn_embeddings is None:
                    raise RuntimeError("MusiCNN embeddings not available - cannot extract valence/arousal")
                va_preds = extractor.valence_arousal_model(musicnn_embeddings)
                mean_preds = va_preds.mean(axis=0)
                feature['valence'] = float(mean_preds[0])
                feature['arousal'] = float(mean_preds[1])

            # 6. Approachability
            if 'approachability_score' not in feature:
                feature['approachability_2c_accessible'] = float(extractor.approachability_2c(embeddings).mean(axis=0)[0])
                feature['approachability_2c_niche'] = float(extractor.approachability_2c(embeddings).mean(axis=0)[1])
                feature['approachability_3c_probs'] = extractor.approachability_3c(embeddings).mean(axis=0).tolist()
                feature['approachability_score'] = float(extractor.approachability_regression(embeddings).mean())

            # 7. Engagement
            if 'engagement_score' not in feature:
                feature['engagement_2c_low'] = float(extractor.engagement_2c(embeddings).mean(axis=0)[0])
                feature['engagement_2c_high'] = float(extractor.engagement_2c(embeddings).mean(axis=0)[1])
                feature['engagement_3c_probs'] = extractor.engagement_3c(embeddings).mean(axis=0).tolist()
                feature['engagement_score'] = float(extractor.engagement_regression(embeddings).mean())

            # 8. MTG-Jamendo
            if 'mtg_jamendo_probs' not in feature:
                feature['mtg_jamendo_probs'] = extractor.mtg_jamendo(embeddings).mean(axis=0).tolist()

            # 9. NEW (2024): Voice Gender
            if 'voice_gender_female' not in feature or 'voice_gender_male' not in feature:
                if not extractor.voice_gender_model:
                    raise RuntimeError("Voice gender model not loaded - cannot extract features")
                gender_probs = extractor.voice_gender_model(embeddings).mean(axis=0)
                # For instrumental songs, set both to 0.0 (no voice present)
                instrumentalness = feature.get('instrumentalness', 0.5)
                if instrumentalness >= 0.5:
                    feature['voice_gender_female'] = 0.0
                    feature['voice_gender_male'] = 0.0
                else:
                    feature['voice_gender_female'] = float(gender_probs[0])
                    feature['voice_gender_male'] = float(gender_probs[1])

            # 10. NEW (2024): Timbre
            if 'timbre_bright' not in feature or 'timbre_dark' not in feature:
                if not extractor.timbre_model:
                    raise RuntimeError("Timbre model not loaded - cannot extract features")
                timbre_probs = extractor.timbre_model(embeddings).mean(axis=0)
                feature['timbre_bright'] = float(timbre_probs[0])
                feature['timbre_dark'] = float(timbre_probs[1])

            # 11. NEW (2024): Acoustic vs Electronic
            if 'mood_acoustic' not in feature or 'mood_electronic' not in feature:
                if not extractor.mood_acoustic_model:
                    raise RuntimeError("Acoustic/electronic model not loaded - cannot extract features")
                acoustic_probs = extractor.mood_acoustic_model(embeddings).mean(axis=0)
                feature['mood_acoustic'] = float(acoustic_probs[0])
                feature['mood_electronic'] = float(acoustic_probs[1])

            # 12. NEW (2024): MTG-Jamendo Instruments
            if 'mtg_jamendo_instrument_probs' not in feature:
                if not extractor.mtg_jamendo_instrument:
                    raise RuntimeError("MTG-Jamendo instrument model not loaded - cannot extract features")
                feature['mtg_jamendo_instrument_probs'] = extractor.mtg_jamendo_instrument(embeddings).mean(axis=0).tolist()

            # 13. NEW (2024): Alternative Arousal/Valence (emoMusic, requires MusiCNN embeddings)
            if 'arousal_emomusic' not in feature or 'valence_emomusic' not in feature:
                if not extractor.emomusic_model:
                    raise RuntimeError("EmoMusic model not loaded - cannot extract features")
                if musicnn_embeddings is None:
                    raise RuntimeError("MusiCNN embeddings not available - cannot extract emoMusic features")
                emo_preds = extractor.emomusic_model(musicnn_embeddings)
                mean_emo = emo_preds.mean(axis=0)
                feature['arousal_emomusic'] = float(mean_emo[0])
                feature['valence_emomusic'] = float(mean_emo[1])

            # 14. NEW (2024): MTG-Jamendo Genre
            if 'mtg_jamendo_genre_probs' not in feature or 'top_mtg_jamendo_genres' not in feature:
                if not extractor.mtg_jamendo_genre:
                    raise RuntimeError("MTG-Jamendo genre model not loaded - cannot extract features")
                mtg_jamendo_genre_probs = extractor.mtg_jamendo_genre(embeddings).mean(axis=0).tolist()
                feature['mtg_jamendo_genre_probs'] = mtg_jamendo_genre_probs

                mtg_jamendo_genre_labels = load_genre_labels('mtg_jamendo_genre')
                top_mtg_jamendo_genre_indices = np.argsort(mtg_jamendo_genre_probs)[-3:][::-1]
                feature['top_mtg_jamendo_genres'] = [(mtg_jamendo_genre_labels[i], float(mtg_jamendo_genre_probs[i])) for i in top_mtg_jamendo_genre_indices]

            # 15. NEW (2024): MIREX Moods (requires MusiCNN embeddings)
            if 'moods_mirex_probs' not in feature:
                if not extractor.moods_mirex_model:
                    raise RuntimeError("MIREX moods model not loaded - cannot extract features")
                if musicnn_embeddings is None:
                    raise RuntimeError("MusiCNN embeddings not available - cannot extract MIREX moods")
                feature['moods_mirex_probs'] = extractor.moods_mirex_model(musicnn_embeddings).mean(axis=0).tolist()

            # 16. Genre Ladder (computed from top_3_genres, added after all features are collected)
            # Note: This is handled in add_genre_ladder_to_features() call after the loop

            updated_features.append(feature)

        except Exception as e:
            logger.error(f"Error updating {feature.get('filename')}: {e}")
            updated_features.append(feature)

    # Add genre ladder feature if missing
    from analysis.pipeline.genre_ladder import add_genre_ladder_to_features
    updated_features = add_genre_ladder_to_features(updated_features)

    # Save back to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(updated_features, f)

    logger.info(f"Successfully updated {len(updated_features)} tracks")
    return updated_features


def extract_audio_features(master_index_path: str = 'spotify/master_index.json',
                          cache_path: str = 'cache/audio_features.pkl') -> List[Dict]:

    with open(master_index_path, 'r') as f:
        master_index = json.load(f)

    tracks_with_mp3 = [track for track in master_index['tracks'] if track.get('mp3_file')]

    logger.info(f"Found {len(tracks_with_mp3)} tracks with MP3 files to process")

    extractor = AudioFeatureExtractor()

    features = []
    errors = []

    for i, track in enumerate(tqdm(tracks_with_mp3, desc="Extracting audio features")):
        filepath = track['mp3_file']
        result = extractor.extract(filepath)

        if result is not None:
            result['track_id'] = track['track_id']
            result['track_name'] = track['track_name']
            result['artist'] = track['artist']
            features.append(result)
        else:
            errors.append(filepath)

        if (i + 1) % 100 == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            logger.info(f"Cached {len(features)} features")

    # Add genre ladder feature (0=acoustic, 1=electronic)
    from analysis.pipeline.genre_ladder import add_genre_ladder_to_features
    features = add_genre_ladder_to_features(features)

    with open(cache_path, 'wb') as f:
        pickle.dump(features, f)

    if errors:
        with open('errors.log', 'w') as f:
            f.write('\n'.join(errors))
        logger.warning(f"Failed to process {len(errors)} files. See errors.log")

    logger.info(f"Successfully extracted features from {len(features)}/{len(tracks_with_mp3)} files")
    return features


if __name__ == '__main__':
    features = extract_audio_features()
    print(f"\nExtracted {len(features)} audio features")
    print(f"\nSample feature keys: {list(features[0].keys())}")
