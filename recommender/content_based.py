import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # -------------------------
        # Feature engineering
        # -------------------------
        self.df["overview_feat"] = self.df["Overview"].fillna("")
        self.df["genre_feat"] = (
            self.df["Genre"].fillna("").astype(str).str.replace(",", " ")
        )
        self.df["lang_feat"] = self.df["Original_Language"].fillna("")

        self.df["combined_features"] = (
            self.df["overview_feat"] + " " +
            (self.df["genre_feat"] + " ") * 3 +
            (self.df["lang_feat"] + " ")
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_features"]
        )

        self.similarity = cosine_similarity(self.tfidf_matrix)

    # -------------------------------------------------
    # Recommend similar movies
    # -------------------------------------------------
    def recommend_similar(
        self,
        title,
        top_n=10,
        genre_filter=None,
        language_filter=None
    ):
        # Guard clause
        if title not in self.df["Title"].values:
            return pd.DataFrame()

        # Base movie
        idx = self.df[self.df["Title"] == title].index[0]
        base_movie = self.df.iloc[idx]

        results = []

        # Iterate over similarity scores
        for i, sim_score in enumerate(self.similarity[idx]):
            if i == idx:
                continue  # skip the same movie

            movie = self.df.iloc[i]

            # -------------------------
            # Filters
            # -------------------------
            if genre_filter:
                movie_genres = [g.strip() for g in str(movie["Genre"]).split(",")]
                if genre_filter not in movie_genres:
                    continue

            if language_filter and language_filter != movie["Original_Language"]:
                continue

            # -------------------------
            # Scoring
            # -------------------------
            popularity_boost = np.log1p(movie["Popularity"]) * 0.05
            final_score = sim_score + popularity_boost

            results.append({
                "index": i,
                "similarity": sim_score,
                "final_score": final_score
            })

        # -------------------------
        # Sort & select top N
        # -------------------------
        top_movies = sorted(
            results,
            key=lambda x: x["final_score"],
            reverse=True
        )[:top_n]

        # -------------------------
        # Build output DataFrame
        # -------------------------
        data = []
        for item in top_movies:
            movie = self.df.iloc[item["index"]]
            data.append({
                "Title": movie["Title"],
                "Poster_Url": movie["Poster_Url"],
                "Similarity": round(item["similarity"] * 100, 2),
                "Why": self._explain(base_movie, movie)
            })

        return pd.DataFrame(data)

    # -------------------------------------------------
    # Explain recommendation
    # -------------------------------------------------
    def _explain(self, base, candidate):
        reasons = []

        # Genre overlap
        base_genres = set(str(base["Genre"]).split(","))
        cand_genres = set(str(candidate["Genre"]).split(","))
        common = base_genres & cand_genres

        if common:
            reasons.append(f"🎭 Shared genres: {', '.join(common)}")

        # Language
        if base["Original_Language"] == candidate["Original_Language"]:
            reasons.append("🗣 Same original language")

        # Story similarity (lightweight)
        base_words = set(str(base["Overview"]).lower().split())
        cand_words = set(str(candidate["Overview"]).lower().split())
        if len(base_words & cand_words) > 5:
            reasons.append("📖 Similar storyline")

        return reasons if reasons else ["Content similarity"]
