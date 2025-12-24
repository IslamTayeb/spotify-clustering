"""Lyric Themes Tab Component

Analyzes lyrical content across clusters:
- TF-IDF keyword extraction (unigrams, bigrams, trigrams)
- Sentiment analysis (positive, negative, neutral)
- Lyric complexity metrics (vocabulary richness, reading ease)
- Common repeated phrases
- Word cloud visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from analysis.interpretability.lyric_themes import (
    load_lyrics_for_cluster,
    extract_tfidf_keywords,
    analyze_sentiment,
    compute_lyric_complexity,
    extract_common_phrases,
)


@st.cache_data
def load_all_lyrics(df: pd.DataFrame, lyrics_dir: str = "lyrics/temp/"):
    """Load all lyrics for the dataset (cached)."""
    all_lyrics = []

    for _, row in df.iterrows():
        filename = row.get("filename", "")
        if not filename:
            continue

        lyric_filename = filename.replace(".mp3", ".txt")
        lyric_file = Path(lyrics_dir) / lyric_filename

        if lyric_file.exists():
            try:
                with open(lyric_file, "r", encoding="utf-8") as f:
                    lyrics_text = f.read().strip()
                    if lyrics_text:
                        all_lyrics.append(lyrics_text)
            except:
                pass

    return all_lyrics


def render_lyric_themes(df: pd.DataFrame):
    """Render Lyric Themes view."""
    st.header("📝 Lyric Theme Analysis")

    st.write("Explore lyric themes, sentiment, and complexity across clusters.")

    # Lyrics directory configuration
    lyrics_dir = st.text_input(
        "Lyrics directory",
        value="lyrics/temp/",
        help="Directory containing lyric .txt files",
    )

    if not Path(lyrics_dir).exists():
        st.error(f"Lyrics directory not found: {lyrics_dir}")
        st.info("💡 Update the path above to point to your lyrics directory")
        return

    # Cluster selection
    cluster_ids = sorted(df["cluster"].unique())
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}",
        key="lyric_cluster",
    )

    with st.spinner("Loading lyrics..."):
        # Load lyrics for selected cluster
        cluster_lyrics_data = load_lyrics_for_cluster(df, selected_cluster, lyrics_dir)

        # Load all lyrics for TF-IDF comparison
        all_lyrics = load_all_lyrics(df, lyrics_dir)

        if not cluster_lyrics_data:
            st.warning(f"No lyrics found for Cluster {selected_cluster}")
            st.info(
                "Make sure lyrics are stored as .txt files matching the MP3 filenames"
            )
            return

        cluster_lyrics = [lyrics for _, lyrics in cluster_lyrics_data]

        # Basic metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            coverage_pct = (
                len(cluster_lyrics_data)
                / len(df[df["cluster"] == selected_cluster])
                * 100
            )
            st.metric("Songs with Lyrics", f"{len(cluster_lyrics_data)}")
            st.caption(f"{coverage_pct:.1f}% coverage")

        with col2:
            if cluster_lyrics:
                avg_word_count = sum(
                    len(lyrics.split()) for lyrics in cluster_lyrics
                ) / len(cluster_lyrics)
                st.metric("Avg Word Count", f"{avg_word_count:.0f}")

        with col3:
            if cluster_lyrics:
                total_words = sum(len(lyrics.split()) for lyrics in cluster_lyrics)
                unique_words = len(set(" ".join(cluster_lyrics).lower().split()))
                st.metric("Unique Words", unique_words)

    # Keyword Analysis
    st.markdown("---")
    st.subheader("🔑 Keyword Analysis (TF-IDF)")

    with st.spinner("Extracting keywords..."):
        if all_lyrics and cluster_lyrics:
            keywords_data = extract_tfidf_keywords(
                all_lyrics, cluster_lyrics, top_n=30, ngram_range=(1, 3)
            )

            if keywords_data["unigrams"]:
                tab1, tab2, tab3 = st.tabs(["Unigrams", "Bigrams", "Trigrams"])

                with tab1:
                    st.write("**Top 30 Single Words**")
                    unigrams_df = pd.DataFrame(
                        keywords_data["unigrams"], columns=["Word", "TF-IDF Score"]
                    )
                    unigrams_df["TF-IDF Score"] = unigrams_df["TF-IDF Score"].round(4)

                    # Bar chart
                    fig = px.bar(
                        unigrams_df.head(20),
                        x="TF-IDF Score",
                        y="Word",
                        orientation="h",
                        title="Top 20 Keywords",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(unigrams_df, use_container_width=True, hide_index=True)

                with tab2:
                    if keywords_data["bigrams"]:
                        st.write("**Top 30 Two-Word Phrases**")
                        bigrams_df = pd.DataFrame(
                            keywords_data["bigrams"], columns=["Phrase", "TF-IDF Score"]
                        )
                        bigrams_df["TF-IDF Score"] = bigrams_df["TF-IDF Score"].round(4)
                        st.dataframe(
                            bigrams_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No significant bigrams found")

                with tab3:
                    if keywords_data["trigrams"]:
                        st.write("**Top 30 Three-Word Phrases**")
                        trigrams_df = pd.DataFrame(
                            keywords_data["trigrams"],
                            columns=["Phrase", "TF-IDF Score"],
                        )
                        trigrams_df["TF-IDF Score"] = trigrams_df["TF-IDF Score"].round(
                            4
                        )
                        st.dataframe(
                            trigrams_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No significant trigrams found")

                # Word Cloud
                st.markdown("---")
                st.subheader("☁️ Word Cloud")

                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt

                    # Create word cloud from keywords
                    word_freq = {
                        word: score for word, score in keywords_data["unigrams"][:50]
                    }

                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="viridis",
                        relative_scaling=0.5,
                        min_font_size=10,
                    ).generate_from_frequencies(word_freq)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close()

                except ImportError:
                    st.warning(
                        "WordCloud library not available. Install with: pip install wordcloud"
                    )

            else:
                st.warning(
                    "No keywords extracted. Try adjusting the lyrics directory path."
                )

    # Sentiment Analysis
    st.markdown("---")
    st.subheader("😊 Sentiment Analysis")

    with st.spinner("Analyzing sentiment..."):
        if cluster_lyrics:
            sentiments = []

            for lyrics in cluster_lyrics:
                sentiment = analyze_sentiment(lyrics)
                sentiments.append(sentiment)

            # Average sentiment
            avg_compound = np.mean([s["compound_score"] for s in sentiments])
            avg_positive = np.mean([s["positive"] for s in sentiments])
            avg_negative = np.mean([s["negative"] for s in sentiments])
            avg_neutral = np.mean([s["neutral"] for s in sentiments])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Compound Score",
                    f"{avg_compound:.3f}",
                    help="Overall sentiment: -1 (very negative) to +1 (very positive)",
                )

            with col2:
                st.metric("Positive", f"{avg_positive:.3f}")

            with col3:
                st.metric("Negative", f"{avg_negative:.3f}")

            with col4:
                st.metric("Neutral", f"{avg_neutral:.3f}")

            # Sentiment distribution
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=[s["compound_score"] for s in sentiments],
                    nbinsx=20,
                    name="Sentiment Distribution",
                )
            )

            fig.update_layout(
                title="Distribution of Sentiment Scores",
                xaxis_title="Compound Sentiment Score",
                yaxis_title="Number of Songs",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Most positive/negative songs
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Positive Songs**")
                # Get top 5 most positive
                sorted_sentiments = sorted(
                    enumerate(sentiments),
                    key=lambda x: x[1]["compound_score"],
                    reverse=True,
                )
                positive_songs = []
                for idx, sentiment in sorted_sentiments[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    positive_songs.append(
                        {
                            "Song": track_name,
                            "Sentiment": f"{sentiment['compound_score']:.3f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(positive_songs),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.write("**Most Negative Songs**")
                negative_songs = []
                for idx, sentiment in sorted_sentiments[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    negative_songs.append(
                        {
                            "Song": track_name,
                            "Sentiment": f"{sentiment['compound_score']:.3f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(negative_songs),
                    use_container_width=True,
                    hide_index=True,
                )

    # Lyric Complexity
    st.markdown("---")
    st.subheader("📚 Lyric Complexity")

    with st.spinner("Computing complexity metrics..."):
        if cluster_lyrics:
            complexities = []

            for lyrics in cluster_lyrics:
                complexity = compute_lyric_complexity(lyrics)
                complexities.append(complexity)

            # Average complexity metrics
            avg_richness = np.mean([c["vocabulary_richness"] for c in complexities])
            avg_word_length = np.mean([c["avg_word_length"] for c in complexities])
            avg_flesch = np.mean(
                [
                    c["flesch_reading_ease"]
                    for c in complexities
                    if c["flesch_reading_ease"] > 0
                ]
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg Vocabulary Richness",
                    f"{avg_richness:.3f}",
                    help="Unique words / Total words (higher = more diverse vocabulary)",
                )

            with col2:
                st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")

            with col3:
                if avg_flesch > 0:
                    st.metric(
                        "Flesch Reading Ease",
                        f"{avg_flesch:.1f}",
                        help="0-100 scale, higher = easier to read",
                    )

            # Complexity extremes
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Complex Lyrics**")
                sorted_complexity = sorted(
                    enumerate(complexities),
                    key=lambda x: x[1]["vocabulary_richness"],
                    reverse=True,
                )
                complex_songs = []
                for idx, complexity in sorted_complexity[:5]:
                    track_name, _ = cluster_lyrics_data[idx]
                    complex_songs.append(
                        {
                            "Song": track_name,
                            "Richness": f"{complexity['vocabulary_richness']:.3f}",
                            "Words": complexity["word_count"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(complex_songs),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.write("**Simplest Lyrics**")
                simple_songs = []
                for idx, complexity in sorted_complexity[-5:][::-1]:
                    track_name, _ = cluster_lyrics_data[idx]
                    simple_songs.append(
                        {
                            "Song": track_name,
                            "Richness": f"{complexity['vocabulary_richness']:.3f}",
                            "Words": complexity["word_count"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(simple_songs),
                    use_container_width=True,
                    hide_index=True,
                )

    # Common Phrases
    st.markdown("---")
    st.subheader("🔁 Common Repeated Phrases")

    with st.spinner("Finding common phrases..."):
        if cluster_lyrics:
            common_phrases = extract_common_phrases(cluster_lyrics, top_n=20)

            if common_phrases:
                phrases_df = pd.DataFrame(common_phrases, columns=["Phrase", "Count"])
                st.write("**Top 20 Repeated Phrases (appear more than once)**")
                st.dataframe(phrases_df, use_container_width=True, hide_index=True)
            else:
                st.info("No commonly repeated phrases found")
