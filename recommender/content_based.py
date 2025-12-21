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
        if title not in self.df["Title"].values:
            return pd.DataFrame()

        idx = self.df[self.df["Title"] == title].index[0]
        base_movie = self.df.iloc[idx]

        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

        scored_movies = []

        for i, sim_score in scores:
            movie = self.df.iloc[i]

            # Genre filter (SAFE)
            if genre_filter:
                movie_genres = [g.strip() for g in movie["Genre"].split(",")]
                if genre_filter not in movie_genres:
                    continue

            # Language filter
            if language_filter and language_filter != movie["Original_Language"]:
                continue

            popularity_boost = np.log1p(movie["Popularity"]) * 0.05
            final_score = sim_score + popularity_boost

            scored_movies.append((i, sim_score, popularity_boost))

            scored_movies = sorted(
                scored_movies,
                key=lambda x: (x[1], x[2]),  # (similarity, popularity)
                reverse=True
            )[:top_n]


            data = []
            for i, sim_score, popularity_boost in scored_movies:
                movie = self.df.iloc[i]
                data.append({
                    "Title": movie["Title"],
                    "Poster_Url": movie["Poster_Url"],
                    "Similarity": round(sim_score * 100, 2),
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
