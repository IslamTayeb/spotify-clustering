#!/usr/bin/env python3
"""
Interpretable Lyric Feature Extraction

Extracts interpretable features from song lyrics using:
- gpt-5-mini-2025-08-07 for semantic features (emotions, themes, explicit content, narrative)
- Local computation for linguistic metrics (vocabulary richness, repetition)

Features follow the three-tier architecture:
- Tier 1: Parallel emotional dimensions (1:1 with audio: valence, arousal, happy, sad, aggressive, relaxed)
- Tier 3: Lyric-unique features (explicit, narrative, theme, language, vocabulary, repetition)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load .env from project root
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# GPT prompt for lyric analysis
LYRIC_ANALYSIS_PROMPT = """You are analyzing song lyrics to extract emotional and thematic features for music clustering.
Your goal is to capture what the lyrics EXPRESS, not what genre the song might be.

## Scoring Guidelines

### Emotional Dimensions (0-1 scale)
These are NOT mutually exclusive. A song can be both happy (0.7) and sad (0.4) if it has bittersweet content.
Use the full range - don't cluster everything around 0.5.

- **valence**: Overall emotional tone
  - 0.0-0.2: Deeply negative (despair, hopelessness, nihilism)
  - 0.3-0.4: Negative (sadness, frustration, anxiety)
  - 0.5: Neutral or mixed
  - 0.6-0.7: Positive (contentment, hope, warmth)
  - 0.8-1.0: Very positive (joy, triumph, elation)

- **arousal**: Energy level of the emotional content
  - 0.0-0.2: Very low energy (lullaby, meditation, quiet reflection)
  - 0.3-0.4: Calm (gentle, peaceful, mellow)
  - 0.5: Moderate
  - 0.6-0.7: Energetic (upbeat, driving, passionate)
  - 0.8-1.0: Very high energy (intense, explosive, manic)

### Mood Scores (0-1 each, independent)
Rate how strongly each mood is present. Most songs will have 1-2 dominant moods (0.6+) and others near 0.

- **happy**: Joy, celebration, elation, fun, playfulness
- **sad**: Grief, melancholy, loss, depression, longing, heartache
- **aggressive**: Anger, frustration, rage, confrontation, defiance, intensity
- **relaxed**: Peace, calm, tranquility, laid-back vibes, unwinding

### Explicit Content (0-1, holistic score)
Consider ALL explicit elements combined:
- Profanity frequency and intensity
- Sexual content (references, explicitness)
- Violence (descriptions, glorification)
- Drug/alcohol references
- Hate speech or slurs

Scale: 0.0 = completely clean, 0.3 = mild (occasional damn/hell), 0.5 = moderate (regular profanity), 0.7 = heavy (explicit sex/violence), 1.0 = extremely explicit

### Theme (pick ONE primary theme)
Choose the single most dominant theme. If truly split, pick what the song is ABOUT at its core.

- "love": Romantic affection, falling in love, devotion, desire for someone
- "heartbreak": Lost love, betrayal, loneliness, breakups, missing someone
- "party": Celebration, clubbing, turning up, having fun, nightlife
- "struggle": Hardship, adversity, overcoming obstacles, perseverance (general, not street-specific)
- "flex": Braggadocio, confidence, success, wealth, status, being the best, showing off
- "street": Trap life, hustle, block, survival, gang life, street credibility, hood stories
- "introspection": Mental health, self-reflection, existential thoughts, vulnerability, inner demons
- "spirituality": God, prayer, faith, blessings, soul, religious themes, higher power
- "social": Politics, social commentary, systemic issues, activism, world problems
- "other": Abstract vibes, nostalgia, nature, fictional narratives, anime themes, misc

### Narrative (0-1)
How much does this song tell a specific story vs. express abstract feelings?
- 0.0-0.2: Pure vibes, abstract, repetitive hooks, no story
- 0.3-0.4: Some context but mostly emotional expression
- 0.5: Mix of story and mood
- 0.6-0.7: Clear narrative with beginning/middle/end
- 0.8-1.0: Very specific story, like a short film

### Language
Return the PRIMARY language. If 80%+ is one language with some words from another, return the dominant one.
Common values: "english", "arabic", "japanese", "spanish", "french", "korean", "portuguese", "german"

## Edge Cases
- Very short/repetitive lyrics: Still analyze what's there, but narrative should be low
- Multilingual: Return the dominant language
- Slang/dialect: Still counts as that language (AAVE = english, etc.)
- Instrumental sections described: Ignore, focus on actual lyrics

## Output Format
Return ONLY valid JSON, no explanation:

{{
  "valence": <float 0-1>,
  "arousal": <float 0-1>,
  "happy": <float 0-1>,
  "sad": <float 0-1>,
  "aggressive": <float 0-1>,
  "relaxed": <float 0-1>,
  "explicit": <float 0-1>,
  "theme": "<string>",
  "narrative": <float 0-1>,
  "language": "<string>"
}}

## Lyrics to Analyze:
{lyrics}
"""


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client, checking for API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not found in environment. GPT features will be skipped."
        )
        return None
    return OpenAI(api_key=api_key)


def extract_gpt_features(lyrics_text: str, client: OpenAI) -> Dict:
    """
    Extract features from lyrics using gpt-5-mini-2025-08-07.

    Args:
        lyrics_text: Raw lyrics text
        client: OpenAI client instance

    Returns:
        Dictionary with extracted features
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[
                {
                    "role": "user",
                    "content": LYRIC_ANALYSIS_PROMPT.format(lyrics=lyrics_text),
                }
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error extracting GPT features: {e}")
        # Return defaults on error
        return {
            "valence": 0.5,
            "arousal": 0.5,
            "happy": 0.0,
            "sad": 0.0,
            "aggressive": 0.0,
            "relaxed": 0.0,
            "explicit": 0.0,
            "theme": "other",
            "narrative": 0.0,
            "language": "unknown",
        }


def compute_local_features(lyrics_text: str) -> Dict:
    """
    Compute linguistic features locally (free, deterministic).

    Args:
        lyrics_text: Raw lyrics text

    Returns:
        Dictionary with vocabulary_richness and repetition_score
    """
    if not lyrics_text or len(lyrics_text.strip()) < 10:
        return {"vocabulary_richness": 0.0, "repetition_score": 0.0}

    words = lyrics_text.lower().split()
    lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]

    # Vocabulary richness (TTR: Type-Token Ratio)
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / len(words) if words else 0.0

    # Repetition score: 1 - (unique lines / total lines)
    unique_lines = set(lines)
    repetition_score = 1 - (len(unique_lines) / len(lines)) if lines else 0.0

    return {
        "vocabulary_richness": vocabulary_richness,
        "repetition_score": repetition_score,
    }


def normalize_features(raw: dict) -> Dict:
    """
    Normalize all features to 0-1 range (except categorical).

    Args:
        raw: Raw features from GPT + local computation

    Returns:
        Normalized features with proper naming
    """
    normalized = {}

    # Tier 1: Parallel emotional dimensions (1:1 with audio)
    normalized["lyric_valence"] = max(
        0.0, min(1.0, float(raw.get("valence", 0.5)))
    )  # ↔ valence
    normalized["lyric_arousal"] = max(
        0.0, min(1.0, float(raw.get("arousal", 0.5)))
    )  # ↔ arousal
    normalized["lyric_mood_happy"] = max(
        0.0, min(1.0, float(raw.get("happy", 0)))
    )  # ↔ mood_happy
    normalized["lyric_mood_sad"] = max(
        0.0, min(1.0, float(raw.get("sad", 0)))
    )  # ↔ mood_sad
    normalized["lyric_mood_aggressive"] = max(
        0.0, min(1.0, float(raw.get("aggressive", 0)))
    )  # ↔ mood_aggressive
    normalized["lyric_mood_relaxed"] = max(
        0.0, min(1.0, float(raw.get("relaxed", 0)))
    )  # ↔ mood_relaxed

    # Tier 3: Lyric-unique features
    normalized["lyric_explicit"] = max(0.0, min(1.0, float(raw.get("explicit", 0))))
    normalized["lyric_narrative"] = max(0.0, min(1.0, float(raw.get("narrative", 0))))

    # Categorical (store as-is)
    normalized["lyric_theme"] = raw.get("theme", "other")
    normalized["lyric_language"] = raw.get("language", "unknown")

    # Local features
    normalized["lyric_vocabulary_richness"] = raw.get("vocabulary_richness", 0.0)
    normalized["lyric_repetition"] = raw.get("repetition_score", 0.0)

    return normalized


def extract_interpretable_lyric_features(
    lyrics_text: str, client: Optional[OpenAI] = None, use_gpt: bool = True
) -> Dict:
    """
    Extract all interpretable features from lyrics.

    Args:
        lyrics_text: Raw lyrics text
        client: OpenAI client (if None, will try to create one)
        use_gpt: Whether to use GPT (if False, only local features)

    Returns:
        Dictionary with all normalized interpretable features
    """
    if not lyrics_text or len(lyrics_text.strip()) < 10:
        # Return zeros for instrumental/no lyrics
        return {
            "lyric_valence": 0.0,
            "lyric_arousal": 0.0,
            "lyric_mood_happy": 0.0,
            "lyric_mood_sad": 0.0,
            "lyric_mood_aggressive": 0.0,
            "lyric_mood_relaxed": 0.0,
            "lyric_explicit": 0.0,
            "lyric_narrative": 0.0,
            "lyric_theme": "none",
            "lyric_language": "none",
            "lyric_vocabulary_richness": 0.0,
            "lyric_repetition": 0.0,
        }

    # Get OpenAI client if needed
    if use_gpt and client is None:
        client = get_openai_client()

    # Extract GPT features
    gpt_features = {}
    if use_gpt and client:
        gpt_features = extract_gpt_features(lyrics_text, client)
    else:
        # Defaults if GPT not available
        gpt_features = {
            "valence": 0.5,
            "arousal": 0.5,
            "happy": 0.0,
            "sad": 0.0,
            "aggressive": 0.0,
            "relaxed": 0.0,
            "explicit": 0.0,
            "theme": "other",
            "narrative": 0.0,
            "language": "unknown",
        }

    # Compute local features
    local_features = compute_local_features(lyrics_text)

    # Combine and normalize
    combined = {**gpt_features, **local_features}
    return normalize_features(combined)


def batch_extract_interpretable_features(
    tracks_with_lyrics: List[Dict],
    lyrics_texts: List[str],
    cache_path: Optional[str] = None,
    use_cache: bool = True,
    batch_delay: float = 0.1,
    save_interval: int = 25,
) -> List[Dict]:
    """
    Extract interpretable features for multiple tracks with incremental saving.

    Args:
        tracks_with_lyrics: List of track dicts with lyrics
        lyrics_texts: List of lyrics text strings (aligned with tracks)
        cache_path: Path to cache file
        use_cache: Load from cache if available (also enables resume from partial)
        batch_delay: Delay between API calls (seconds)
        save_interval: Save progress every N tracks (default: 25)

    Returns:
        List of feature dicts (one per track)
    """
    import pickle
    import time

    cache_file = (
        Path(cache_path)
        if cache_path
        else Path("cache/lyric_interpretable_features.pkl")
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache for resume support
    existing_features = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            existing_features = {f["track_id"]: f for f in cached_data}
            logger.info(f"Found {len(existing_features)} existing cached features")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")

    # If use_cache and ALL tracks are already cached, return early
    all_track_ids = {t["track_id"] for t in tracks_with_lyrics}
    if use_cache and all_track_ids <= set(existing_features.keys()):
        logger.info("All tracks already cached, returning cached data")
        return [
            existing_features[tid]
            for tid in [t["track_id"] for t in tracks_with_lyrics]
        ]

    # Get OpenAI client
    client = get_openai_client()
    if not client:
        logger.warning("OpenAI client not available. Using defaults for all tracks.")

    features = []
    processed_count = 0
    skipped_count = 0

    # Process each track
    for idx, (track, lyrics_text) in enumerate(
        tqdm(
            zip(tracks_with_lyrics, lyrics_texts),
            desc="Extracting interpretable lyric features",
            total=len(tracks_with_lyrics),
        )
    ):
        track_id = track["track_id"]

        # Skip if already cached (resume support)
        if use_cache and track_id in existing_features:
            features.append(existing_features[track_id])
            skipped_count += 1
            continue

        try:
            interpretable_features = extract_interpretable_lyric_features(
                lyrics_text, client=client, use_gpt=(client is not None)
            )

            # Add track metadata
            feature_dict = {
                "track_id": track_id,
                "track_name": track["track_name"],
                "artist": track["artist"],
                **interpretable_features,
            }
            features.append(feature_dict)
            existing_features[track_id] = feature_dict
            processed_count += 1

            # Small delay to respect rate limits
            if client:
                time.sleep(batch_delay)

        except Exception as e:
            logger.error(f"Error processing {track.get('track_name', 'unknown')}: {e}")
            # Add defaults on error
            default_features = {
                "track_id": track_id,
                "track_name": track["track_name"],
                "artist": track["artist"],
                "lyric_valence": 0.5,
                "lyric_arousal": 0.5,
                "lyric_mood_happy": 0.0,
                "lyric_mood_sad": 0.0,
                "lyric_mood_aggressive": 0.0,
                "lyric_mood_relaxed": 0.0,
                "lyric_explicit": 0.0,
                "lyric_narrative": 0.0,
                "lyric_theme": "other",
                "lyric_language": "unknown",
                "lyric_vocabulary_richness": 0.0,
                "lyric_repetition": 0.0,
            }
            features.append(default_features)
            existing_features[track_id] = default_features

        # Incremental save every N tracks
        if processed_count > 0 and processed_count % save_interval == 0:
            all_features = list(existing_features.values())
            with open(cache_file, "wb") as f:
                pickle.dump(all_features, f)
            logger.info(f"Incremental save: {len(all_features)} tracks cached")

    # Final save
    all_features = list(existing_features.values())
    with open(cache_file, "wb") as f:
        pickle.dump(all_features, f)

    logger.info(
        f"Complete: {processed_count} new, {skipped_count} cached, "
        f"{len(all_features)} total saved to {cache_file}"
    )

    return features
