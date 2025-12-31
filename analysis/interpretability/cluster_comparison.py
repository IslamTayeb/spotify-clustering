"""
Cluster Comparison Module

Statistical comparison of clusters using t-tests and effect sizes.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple

from analysis.interpretability.feature_importance import compute_cohens_d
from analysis.pipeline.config import get_cluster_name

logger = logging.getLogger(__name__)


# Full 33-dimensional interpretable feature list
# This should be used for all cluster comparisons to match the clustering space
FULL_33_DIM_FEATURES = [
    # ═══════════════════════════════════════════════════════════════════════
    # AUDIO FEATURES (dims 0-15)
    # ═══════════════════════════════════════════════════════════════════════
    'bpm', 'danceability', 'instrumentalness', 'valence', 'arousal',
    'engagement_score', 'approachability_score',
    'mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party',
    'emb_voice_gender', 'genre_fusion', 'emb_acoustic_electronic', 'emb_timbre_brightness',
    # ═══════════════════════════════════════════════════════════════════════
    # KEY FEATURES (dims 16-18)
    # ═══════════════════════════════════════════════════════════════════════
    'emb_key_sin', 'emb_key_cos', 'emb_key_scale',
    # ═══════════════════════════════════════════════════════════════════════
    # LYRIC FEATURES (dims 19-28)
    # ═══════════════════════════════════════════════════════════════════════
    'lyric_valence', 'lyric_arousal',
    'lyric_mood_happy', 'lyric_mood_sad', 'lyric_mood_aggressive', 'lyric_mood_relaxed',
    'lyric_explicit', 'lyric_narrative', 'lyric_vocabulary_richness', 'lyric_repetition',
    # ═══════════════════════════════════════════════════════════════════════
    # META FEATURES (dims 29-32)
    # ═══════════════════════════════════════════════════════════════════════
    'emb_theme', 'emb_language', 'popularity', 'emb_release_year',
]


def compare_two_clusters(
    df: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    features: List[str] = None,
) -> pd.DataFrame:
    """
    Compare two clusters using Welch's t-tests for continuous features.

    Welch's t-test does not assume equal variances, making it more robust
    for comparing clusters of different sizes or variances.

    Args:
        df: DataFrame with cluster assignments and features
        cluster_a: ID of first cluster
        cluster_b: ID of second cluster
        features: List of feature names to compare (default: full 33-dim interpretable vector)

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - cluster_a_mean, cluster_b_mean: Means for each cluster
        - difference: cluster_a_mean - cluster_b_mean
        - effect_size: Cohen's d
        - t_statistic, p_value: Statistical test results
        - significant: Boolean (p < 0.05)
    """
    if features is None:
        features = FULL_33_DIM_FEATURES.copy()

    # Filter to only include features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    cluster_a_df = df[df['cluster'] == cluster_a]
    cluster_b_df = df[df['cluster'] == cluster_b]

    results = []

    for feature in features:
        values_a = cluster_a_df[feature].dropna().values
        values_b = cluster_b_df[feature].dropna().values

        if len(values_a) == 0 or len(values_b) == 0:
            continue

        # Compute means
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        difference = mean_a - mean_b

        # Compute effect size (Cohen's d)
        effect_size = compute_cohens_d(values_a, values_b)

        # Welch's t-test (does not assume equal variance)
        try:
            t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        except Exception as e:
            logger.warning(f"t-test failed for feature {feature}: {e}")
            t_stat, p_value = np.nan, np.nan

        results.append({
            'feature': feature,
            'cluster_a_mean': mean_a,
            'cluster_b_mean': mean_b,
            'difference': difference,
            'effect_size': effect_size,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False,
        })

    results_df = pd.DataFrame(results)

    # Sort by absolute effect size
    if len(results_df) > 0:
        results_df['abs_effect_size'] = results_df['effect_size'].abs()
        results_df = results_df.sort_values('abs_effect_size', ascending=False)
        results_df = results_df.drop(columns=['abs_effect_size'])

    return results_df


def create_similarity_matrix(
    df: pd.DataFrame,
    feature: str = 'bpm',
) -> pd.DataFrame:
    """
    Create pairwise comparison matrix for a feature across all clusters.

    The matrix shows effect size (Cohen's d) of difference between each pair.
    Cell (i,j) = effect size when comparing cluster i to cluster j.

    Args:
        df: DataFrame with cluster assignments and features
        feature: Feature name to analyze

    Returns:
        DataFrame where rows and columns are cluster IDs, values are effect sizes
    """
    cluster_ids = sorted(df['cluster'].unique())

    matrix = pd.DataFrame(
        index=cluster_ids,
        columns=cluster_ids,
        dtype=float
    )

    for cluster_a in cluster_ids:
        for cluster_b in cluster_ids:
            if cluster_a == cluster_b:
                matrix.loc[cluster_a, cluster_b] = 0.0
            else:
                values_a = df[df['cluster'] == cluster_a][feature].dropna().values
                values_b = df[df['cluster'] == cluster_b][feature].dropna().values

                if len(values_a) > 0 and len(values_b) > 0:
                    effect_size = compute_cohens_d(values_a, values_b)
                    matrix.loc[cluster_a, cluster_b] = effect_size
                else:
                    matrix.loc[cluster_a, cluster_b] = 0.0

    return matrix


def find_most_different_pairs(
    df: pd.DataFrame,
    features: List[str] = None,
    top_n: int = 3,
) -> List[Tuple[int, int, float]]:
    """
    Identify cluster pairs with largest overall differences.

    Computes average effect size across all features for each pair.

    Args:
        df: DataFrame with cluster assignments and features
        features: List of features to consider (default: full 33-dim interpretable vector)
        top_n: Number of top pairs to return

    Returns:
        List of tuples: (cluster_a, cluster_b, avg_effect_size)
    """
    if features is None:
        features = FULL_33_DIM_FEATURES.copy()

    features = [f for f in features if f in df.columns]
    cluster_ids = sorted(df['cluster'].unique())

    pair_scores = []

    for i, cluster_a in enumerate(cluster_ids):
        for cluster_b in cluster_ids[i+1:]:  # Only compare each pair once
            comparison_df = compare_two_clusters(df, cluster_a, cluster_b, features)

            if len(comparison_df) > 0:
                avg_effect_size = comparison_df['effect_size'].abs().mean()
                pair_scores.append((cluster_a, cluster_b, avg_effect_size))

    # Sort by average effect size (descending)
    pair_scores = sorted(pair_scores, key=lambda x: x[2], reverse=True)

    return pair_scores[:top_n]


def compute_cluster_similarity_matrix(
    df: pd.DataFrame,
    features: List[str] = None,
    use_names: bool = True,
) -> pd.DataFrame:
    """
    Compute overall similarity matrix for all clusters.

    Uses average effect size across all features as the distance metric.
    Lower values = more similar clusters.

    Args:
        df: DataFrame with cluster assignments and features
        features: List of features to consider (default: full 33-dim interpretable vector)
        use_names: If True, use cluster names as index/columns; else use IDs

    Returns:
        DataFrame where rows and columns are cluster names (or IDs),
        values are average absolute effect sizes (dissimilarity)
    """
    if features is None:
        features = FULL_33_DIM_FEATURES.copy()

    features = [f for f in features if f in df.columns]
    cluster_ids = sorted(df['cluster'].unique())

    # Create index/column labels
    if use_names:
        labels = [get_cluster_name(cid) for cid in cluster_ids]
    else:
        labels = cluster_ids

    matrix = pd.DataFrame(
        index=labels,
        columns=labels,
        dtype=float
    )

    # Build a mapping from label to cluster_id for lookups
    label_to_id = {label: cid for label, cid in zip(labels, cluster_ids)}

    for label_a, cluster_a in zip(labels, cluster_ids):
        for label_b, cluster_b in zip(labels, cluster_ids):
            if cluster_a == cluster_b:
                matrix.loc[label_a, label_b] = 0.0
            else:
                comparison_df = compare_two_clusters(df, cluster_a, cluster_b, features)

                if len(comparison_df) > 0:
                    # Average absolute effect size = dissimilarity
                    avg_effect = comparison_df['effect_size'].abs().mean()
                    matrix.loc[label_a, label_b] = avg_effect
                else:
                    matrix.loc[label_a, label_b] = 0.0

    return matrix
