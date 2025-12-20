import streamlit as st
import pandas as pd
from recommender.content_based import ContentBasedRecommender

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")
    df.dropna(subset=["Title", "Poster_Url"], inplace=True)
    return df

df = load_data()
recommender = ContentBasedRecommender(df)

st.title("🎥 Movie Recommendation System")

# ------------------------
# SEARCH
# ------------------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "breadcrumbs" not in st.session_state:
    st.session_state.breadcrumbs = []


movie_list = sorted(df["Title"].unique())
selected_movie = st.selectbox(
    "Search for a movie",
    movie_list,
    index=movie_list.index(st.session_state.selected_movie)
    if st.session_state.selected_movie in movie_list
    else 0
)
if (
    st.session_state.selected_movie
    and (
        not st.session_state.breadcrumbs
        or st.session_state.breadcrumbs[-1] != st.session_state.selected_movie
    )
):
    st.session_state.breadcrumbs.append(st.session_state.selected_movie)


st.session_state.selected_movie = selected_movie

# ------------------------
# FILTERS
# ------------------------
col1, col2 = st.columns(2)

with col1:
    genre_filter = st.selectbox(
        "Filter by Genre (Recommendations)",
        ["All"] + sorted(df["Genre"].dropna().unique())
    )

genre_filter = None if genre_filter == "All" else genre_filter

# ------------------------
# DISPLAY
# ------------------------
col_nav, col_clear = st.columns([4, 1])

if st.session_state.breadcrumbs:
    col_nav, col_clear = st.columns([4, 1])

    with col_nav:
        st.markdown("### 🧭 Navigation")

    with col_clear:
        if st.button("🧹 Clear"):
            st.session_state.selected_movie = None
            st.session_state.breadcrumbs = []
            st.experimental_rerun()

    crumb_cols = st.columns(len(st.session_state.breadcrumbs))

    for i, movie in enumerate(st.session_state.breadcrumbs):
        with crumb_cols[i]:
            if st.button(movie, key=f"crumb_{i}"):
                st.session_state.selected_movie = movie
                st.session_state.breadcrumbs = st.session_state.breadcrumbs[: i + 1]
                st.experimental_rerun()




if selected_movie:
    movie = df[df["Title"] == selected_movie].iloc[0]

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(movie["Poster_Url"], use_column_width=True)

    with col2:
        st.subheader(movie["Title"])
        st.write(movie["Overview"])
        st.markdown(f"**Genre:** {movie['Genre']}")
        st.markdown(f"**Language:** {movie['Original_Language']}")
        st.markdown(f"**Rating:** ⭐ {movie['Vote_Average']}")

    st.markdown("---")
    st.subheader("🎯 Recommended Movies")

    recommendations = recommender.recommend_similar(
        st.session_state.selected_movie,
        genre_filter=genre_filter
    )

    cols = st.columns(5)

    for i, row in recommendations.iterrows():
        with cols[i % 5]:
            st.image(row["Poster_Url"])
            st.caption(f"{row['Title']} ({row['Similarity']}%)")

            # CLICK BUTTON
            if st.button("View details", key=f"btn_{row['Title']}"):
                st.session_state.selected_movie = row["Title"]
                st.experimental_rerun()

