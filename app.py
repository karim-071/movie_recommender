import streamlit as st
import pandas as pd
from recommender.content_based import ContentBasedRecommender

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown("""
<style>
    div.stButton > button {
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
}

.movie-card {
    background-color: #111;
    border-radius: 12px;
    padding: 12px;
    height: 520px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.movie-card:hover {
    transform: scale(1.03);
    box-shadow: 0 9px 27px rgba(0, 0, 0, 0.6);
}

.movie-poster {
    border-radius: 8px;
    height: 280px;
    object-fit: cover;
}

.movie-title {
    font-weight: 600;
    font-size: 14px;
    margin-top: 6px;
}
.movie-score {
    font-size: 13px;
    font-weight: 600;
    margin-top: 6px 0 4px 0;
    color: #f5c518;
}

.movie-reason {
    margin: 0;
    padding-left: 16px;
    font-size: 12px;
    color: #bbb;
}
.movie-reason li {
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True
)


st.markdown(
    """

    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset.csv")
    df.dropna(subset=["Title", "Poster_Url"], inplace=True)
    return df

@st.cache_resource(show_spinner=False)
def load_recommender(df):
    return ContentBasedRecommender(df)

df = load_data()
recommender = load_recommender(df)

st.title("🎥 Movie Recommendation System")

# ------------------------
# SEARCH
# ------------------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "breadcrumbs" not in st.session_state:
    st.session_state.breadcrumbs = []

if "in_recommendation_flow" not in st.session_state:
    st.session_state.in_recommendation_flow = False



movie_list = sorted(df["Title"].unique())
selected_movie = st.selectbox(
    "Search for a movie",
    movie_list,
    index=movie_list.index(st.session_state.selected_movie)
    if st.session_state.selected_movie in movie_list
    else 0
)

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
if st.session_state.in_recommendation_flow and st.session_state.breadcrumbs:
    col_nav, col_clear = st.columns([4, 1])

    with col_nav:
        st.markdown("### 🧭 Navigation")

    with col_clear:
        if st.button("🧹 Clear"):
            st.session_state.selected_movie = None
            st.session_state.breadcrumbs = []
            st.session_state.in_recommendation_flow = False
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

    placeholder = st.empty()

    with placeholder.container():
        with st.spinner("🎬 Loading movie details..."):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(movie["Poster_Url"], use_column_width=True)

            with col2:
                st.subheader(movie["Title"])
                st.write(movie["Overview"])
                st.markdown(f"**Genre:** {movie['Genre']}")
                st.markdown(f"**Language:** {movie['Original_Language']}")
                st.markdown(f"**Rating:** ⭐ {movie['Vote_Average']}")
                st.markdown(f"**Votes:** 💬 {movie['Vote_Count']}")

    st.markdown("---")
    st.subheader("🎯 Recommended Movies")

    with st.spinner("🔍 Finding similar movies you might like..."):
        recommendations = recommender.recommend_similar(
            st.session_state.selected_movie,
            genre_filter=genre_filter
        )

    cols = st.columns(5)

    for i, row in recommendations.iterrows():
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class="movie-card">
                    <img src="{row['Poster_Url']}" class="movie-poster"/>
                    <div class="movie-title">{row['Title']}</div>
                    <div class="movie-score">{row['Similarity']}% Similar <br/></div>
                    <ul class="movie-reason">
                        {''.join([f'<li>{r}</li>' for r in row['Why']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("View details", key=f"rec_{row['Title']}"):
                st.session_state.selected_movie = row["Title"]
                st.session_state.in_recommendation_flow = True


                if (
                    not st.session_state.breadcrumbs
                    or st.session_state.breadcrumbs[-1] != row["Title"]
                ):
                    st.session_state.breadcrumbs.append(row["Title"])

                st.experimental_rerun()

# ------------------------
# ABOUT THIS APP
# ------------------------
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.markdown("### 🎬 What can this app do?")
    st.info(
        "This movie recommendation system suggests movies similar to a selected title "
        "using content-based filtering. Recommendations are generated by analyzing "
        "movie overviews, genres, and original language."
    )

    st.markdown("### 🧭 How to use the app?")
    st.warning(
        '1. Search for a movie using the search box.\n'
        '2. View detailed information such as poster, overview, genre, and rating.\n'
        '3. Browse recommended movies ranked by similarity score.\n'
        '4. Click on any recommended movie to explore its details and get further recommendations.\n'
        '5. Use genre filters to refine recommendation results.'
    )

    st.markdown("### 🧠 How are recommendations generated?")
    st.success("""
    - **TF-IDF** converts movie text into numerical features  
    - **Cosine similarity** measures how closely movies are related  
    - Recommendations are ranked by **similarity score** (popularity as secondary signal)  
    - Each recommendation includes an **explanation** for transparency
    """)
st.markdown("---")
st.caption("Made with ❤️ for learning recommendation systems")
