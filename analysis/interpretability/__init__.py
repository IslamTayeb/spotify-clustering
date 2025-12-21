"""
Cluster Interpretability Module

This module provides tools for interpreting and understanding music clustering results:
- Feature importance analysis using Cohen's d effect sizes
- Statistical cluster comparisons using t-tests
- Lyric theme extraction using TF-IDF, sentiment analysis, and complexity metrics
"""

__version__ = "0.1.0"

from analysis.interpretability.feature_importance import (
    compute_cohens_d,
    compute_feature_importance,
    get_top_features,
)

from analysis.interpretability.cluster_comparison import (
    compare_two_clusters,
    create_similarity_matrix,
    find_most_different_pairs,
)

from analysis.interpretability.lyric_themes import (
    load_lyrics_for_cluster,
    extract_tfidf_keywords,
    analyze_sentiment,
    compute_lyric_complexity,
    extract_common_phrases,
    compare_cluster_keywords,
)

__all__ = [
    # Feature importance
    "compute_cohens_d",
    "compute_feature_importance",
    "get_top_features",
    # Cluster comparison
    "compare_two_clusters",
    "create_similarity_matrix",
    "find_most_different_pairs",
    # Lyric themes
    "load_lyrics_for_cluster",
    "extract_tfidf_keywords",
    "analyze_sentiment",
    "compute_lyric_complexity",
    "extract_common_phrases",
    "compare_cluster_keywords",
]
