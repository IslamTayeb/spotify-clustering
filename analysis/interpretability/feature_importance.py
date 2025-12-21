"""
Feature Importance Analysis Module

Identifies which features make each cluster distinctive using Cohen's d effect sizes.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def compute_cohens_d(cluster_values: np.ndarray, global_values: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between cluster and global distribution.

    Cohen's d measures the difference between two means in terms of standard deviation.
    Interpretation:
    - |d| < 0.2: small effect
    - |d| < 0.5: medium effect
    - |d| >= 0.8: large effect

    Args:
        cluster_values: Array of feature values for songs in the cluster
        global_values: Array of feature values for all songs

    Returns:
        float: Cohen's d effect size (can be positive or negative)
    """
    # Handle edge cases
    if len(cluster_values) == 0 or len(global_values) == 0:
        return 0.0

    # Remove NaN values
    cluster_values = cluster_values[~np.isnan(cluster_values)]
    global_values = global_values[~np.isnan(global_values)]

    if len(cluster_values) == 0 or len(global_values) == 0:
        return 0.0

    cluster_mean = np.mean(cluster_values)
    global_mean = np.mean(global_values)

    cluster_std = np.std(cluster_values, ddof=1) if len(cluster_values) > 1 else 0.0
    global_std = np.std(global_values, ddof=1) if len(global_values) > 1 else 0.0

    # Pooled standard deviation
    if cluster_std == 0 and global_std == 0:
        return 0.0  # No variation

    pooled_std = np.sqrt((cluster_std**2 + global_std**2) / 2)

    if pooled_std == 0:
        return 0.0

    cohens_d = (cluster_mean - global_mean) / pooled_std
    return cohens_d


def compute_feature_importance(
    df: pd.DataFrame,
    cluster_id: int,
    continuous_features: List[str] = None,
    categorical_features: List[str] = None,
) -> pd.DataFrame:
    """
    Compute feature importance for a cluster using effect sizes.

    For continuous features: Uses Cohen's d effect size
    For categorical features: Uses probability ratio (cluster_freq / global_freq)

    Args:
        df: DataFrame with cluster assignments and features
        cluster_id: ID of the cluster to analyze
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - effect_size: Cohen's d (continuous) or probability_ratio (categorical)
        - cluster_mean: Mean value in cluster
        - global_mean: Mean value across all songs
        - cluster_std: Standard deviation in cluster
        - global_std: Standard deviation across all songs
        - importance_rank: Sorted by absolute effect size
    """
    if continuous_features is None:
        continuous_features = [
            'bpm', 'danceability', 'instrumentalness', 'valence', 'arousal',
            'engagement_score', 'approachability_score',
            'mood_happy', 'mood_sad', 'mood_aggressive', 'mood_relaxed', 'mood_party',
            'timbre_bright', 'timbre_dark',
            'voice_gender_male', 'voice_gender_female',
            'mood_acoustic', 'mood_electronic',
            'word_count',
        ]

    # Filter to only include features that exist in the dataframe
    continuous_features = [f for f in continuous_features if f in df.columns]

    cluster_mask = df['cluster'] == cluster_id
    cluster_df = df[cluster_mask]

    results = []

    for feature in continuous_features:
        cluster_values = cluster_df[feature].values
        global_values = df[feature].values

        # Compute Cohen's d
        effect_size = compute_cohens_d(cluster_values, global_values)

        # Compute means and stds (handling NaN)
        cluster_mean = np.nanmean(cluster_values) if len(cluster_values) > 0 else np.nan
        global_mean = np.nanmean(global_values)
        cluster_std = np.nanstd(cluster_values, ddof=1) if len(cluster_values) > 1 else 0.0
        global_std = np.nanstd(global_values, ddof=1)

        results.append({
            'feature': feature,
            'effect_size': effect_size,
            'cluster_mean': cluster_mean,
            'global_mean': global_mean,
            'cluster_std': cluster_std,
            'global_std': global_std,
            'type': 'continuous'
        })

    # Handle categorical features
    if categorical_features:
        categorical_features = [f for f in categorical_features if f in df.columns]

        for feature in categorical_features:
            # For categorical, compute frequency ratio
            cluster_counts = cluster_df[feature].value_counts(normalize=True)
            global_counts = df[feature].value_counts(normalize=True)

            for value in global_counts.index:
                cluster_freq = cluster_counts.get(value, 0)
                global_freq = global_counts.get(value, 0)

                if global_freq > 0:
                    prob_ratio = cluster_freq / global_freq
                else:
                    prob_ratio = 0.0

                # Treat prob_ratio > 1 as positive effect, < 1 as negative
                # Convert to log scale for better interpretation
                if prob_ratio > 0:
                    effect_size = np.log2(prob_ratio)  # Log2: 2x more common = +1, 2x less common = -1
                else:
                    effect_size = -10.0  # Very negative for absent features

                results.append({
                    'feature': f"{feature}={value}",
                    'effect_size': effect_size,
                    'cluster_mean': cluster_freq,
                    'global_mean': global_freq,
                    'cluster_std': 0.0,
                    'global_std': 0.0,
                    'type': 'categorical'
                })

    # Create DataFrame and sort by absolute effect size
    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        results_df['abs_effect_size'] = results_df['effect_size'].abs()
        results_df = results_df.sort_values('abs_effect_size', ascending=False)
        results_df['importance_rank'] = range(1, len(results_df) + 1)
        results_df = results_df.drop(columns=['abs_effect_size'])

    return results_df


def get_top_features(
    df: pd.DataFrame,
    cluster_id: int,
    n: int = 10,
    continuous_features: List[str] = None,
    categorical_features: List[str] = None,
) -> Dict[str, any]:
    """
    Get top N most distinctive features for a cluster.

    Args:
        df: DataFrame with cluster assignments and features
        cluster_id: ID of the cluster to analyze
        n: Number of top features to return
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names

    Returns:
        Dictionary with:
        - top_features: DataFrame of top N features
        - cluster_size: Number of songs in cluster
        - cluster_percentage: Percentage of total songs
    """
    importance_df = compute_feature_importance(
        df, cluster_id, continuous_features, categorical_features
    )

    top_features = importance_df.head(n)

    cluster_size = len(df[df['cluster'] == cluster_id])
    total_size = len(df)
    cluster_percentage = (cluster_size / total_size * 100) if total_size > 0 else 0.0

    return {
        'top_features': top_features,
        'cluster_size': cluster_size,
        'cluster_percentage': cluster_percentage,
        'all_features': importance_df,
    }


def get_feature_interpretation(effect_size: float) -> str:
    """
    Get human-readable interpretation of Cohen's d effect size.

    Args:
        effect_size: Cohen's d value

    Returns:
        String interpretation
    """
    abs_d = abs(effect_size)

    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    direction = "higher" if effect_size > 0 else "lower"

    return f"{magnitude} ({direction} than average)"
