import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame):
        # Store dataframe
        self.df = df.reset_index(drop=True)

        # -------------------------
        # Feature engineering
        # -------------------------
        self.df["overview_feat"] = self.df["Overview"].fillna("")
        self.df["genre_feat"] = (
            self.df["Genre"]
            .fillna("")
            .astype(str)
            .str.replace(",", " ")
        )
        self.df["lang_feat"] = self.df["Original_Language"].fillna("")

        # Weighted combination (ONLY integer repetition)
        self.df["combined_features"] = (
            self.df["overview_feat"] + " " +
            (self.df["genre_feat"] + " ") * 3 +   # genre weight
            (self.df["lang_feat"] + " ") * 1      # language weight
        )

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_features"]
        )

        # Cosine similarity
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
        scores = list(enumerate(self.similarity[idx]))

        # Sort & remove the same movie
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]

        results = []
        base_movie = self.df.iloc[idx]

        for i, sim_score in scores:
            movie = self.df.iloc[i]

            # Filters
            if genre_filter and genre_filter not in movie["Genre"]:
                continue
            if language_filter and language_filter != movie["Original_Language"]:
                continue

            # Popularity boost (safe)
            popularity_boost = np.log1p(movie["Popularity"]) * 0.05
            final_score = sim_score + popularity_boost

            results.append((i, final_score, sim_score))

            if len(results) >= top_n:
                break

        data = []
        for i, final_score, raw_sim in results:
            movie = self.df.iloc[i]
            data.append({
                "Title": movie["Title"],
                "Poster_Url": movie["Poster_Url"],
                "Similarity": round(raw_sim * 100, 2),
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
        common_genres = base_genres & cand_genres

        if common_genres:
            reasons.append(f"Shared genres: {', '.join(common_genres)}")

        # Language match
        if base["Original_Language"] == candidate["Original_Language"]:
            reasons.append("Same original language")

        # Story similarity
        base_words = set(str(base["Overview"]).lower().split())
        cand_words = set(str(candidate["Overview"]).lower().split())
        if len(base_words & cand_words) > 5:
            reasons.append("Similar storyline")

        return " • ".join(reasons) if reasons else "Content similarity"
