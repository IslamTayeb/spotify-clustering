"""Constants for clustering pipeline."""

# Names for all 33 embedding dimensions (used for clustering)
EMBEDDING_DIM_NAMES = [
    # Audio features (16 dims: indices 0-15)
    "emb_bpm",                    # 0: BPM (normalized to [0,1])
    "emb_danceability",           # 1: Danceability
    "emb_instrumentalness",       # 2: Instrumentalness
    "emb_valence",                # 3: Valence (normalized to [0,1])
    "emb_arousal",                # 4: Arousal (normalized to [0,1])
    "emb_engagement",             # 5: Engagement score
    "emb_approachability",        # 6: Approachability score
    "emb_mood_happy",             # 7: Mood - Happy
    "emb_mood_sad",               # 8: Mood - Sad
    "emb_mood_aggressive",        # 9: Mood - Aggressive
    "emb_mood_relaxed",           # 10: Mood - Relaxed
    "emb_mood_party",             # 11: Mood - Party
    "emb_voice_gender",           # 12: Voice Gender (0=female, 1=male)
    "emb_genre_ladder",           # 13: Genre Ladder (0=pure genre, 1=genre fusion)
    "emb_acoustic_electronic",    # 14: Acoustic/Electronic (0=electronic, 1=acoustic)
    "emb_timbre_brightness",      # 15: Timbre Brightness (0=dark, 1=bright)
    # Key features (3 dims: indices 16-18)
    "emb_key_sin",                # 16: Key pitch (sin component)
    "emb_key_cos",                # 17: Key pitch (cos component)
    "emb_key_scale",              # 18: Key scale (0=minor, 0.33=major)
    # Lyric features (10 dims: indices 19-28)
    "emb_lyric_valence",          # 19: Lyric valence
    "emb_lyric_arousal",          # 20: Lyric arousal
    "emb_lyric_mood_happy",       # 21: Lyric mood - Happy
    "emb_lyric_mood_sad",         # 22: Lyric mood - Sad
    "emb_lyric_mood_aggressive",  # 23: Lyric mood - Aggressive
    "emb_lyric_mood_relaxed",     # 24: Lyric mood - Relaxed
    "emb_lyric_explicit",         # 25: Explicit content
    "emb_lyric_narrative",        # 26: Narrative style
    "emb_lyric_vocabulary",       # 27: Vocabulary richness
    "emb_lyric_repetition",       # 28: Repetition score
    # Theme, Language, Metadata (4 dims: indices 29-32)
    "emb_theme",                  # 29: Theme (ordinal scale)
    "emb_language",               # 30: Language (ordinal scale)
    "emb_popularity",             # 31: Popularity (normalized to [0,1])
    "emb_release_year",           # 32: Release Year (decade buckets)
]

# Index ranges for feature weighting
FEATURE_WEIGHT_INDICES = {
    'core_audio': (0, 7),      # BPM through approachability (indices 0-6)
    'mood': (7, 12),           # 5 mood dimensions (indices 7-11)
    'genre': (12, 16),         # Voice gender, genre ladder, acoustic/electronic, timbre (indices 12-15)
    'key': (16, 19),           # 3 key dimensions (indices 16-18)
    'lyric_emotion': (19, 25), # Lyric valence/arousal + 4 moods (indices 19-24)
    'lyric_content': (25, 29), # Explicit, narrative, vocabulary, repetition (indices 25-28)
    'theme': (29, 30),         # Theme dimension (index 29)
    'language': (30, 31),      # Language dimension (index 30)
    'metadata': (31, 33),      # Popularity + Release Year (indices 31-32)
}
