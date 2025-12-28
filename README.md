# 🎬 Movie Recommendation System

A **content-based movie recommendation system** built with *Python*, *Streamlit*, and *scikit-learn*. The application suggests movies similar to a selected title based on *overview*, *genre*, and *original language* with an *interactive UI*, *explainable recommendations*, and *genre & language filters*.

## 🚀 Live Demo
👉 https://movies--recommend.streamlit.app/

## 🚀 Features
- Search and select movies by title
- View detailed movie information (poster, overview, genre, rating)
- Content-based recommendations using **TF-IDF + cosine similarity**
- Get personalized recommendations ranked by similarity score(with popularity as a secondary signal)
- Genre filtering (atomic genres like Action, Horror, Drama)  
- Language filtering 
- Clear explanation for each recommendation (why it was suggested)
- Fast and interactive **Streamlit UI**

## 🧠 How It Works
- Uses **TF-IDF Vectorization** to convert movie text (overview, genres, language) into numerical features  
- Applies **Cosine Similarity** to identify movies with similar content  
- Ranks recommendations by similarity score (with a popularity boost)  
- Displays explanations such as:
  - Shared genres  
  - Same original language  
  - Similar themes based on TF-IDF keywords  

## 🛠 Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- HTML/CSS for UI styling

## 🔮 Future Improvements
- Multi-genre and multi-language filtering
- User profiles and personalization
- Collaborative filtering (SVD / matrix factorization)
- Hybrid recommendation system (content + collaborative)


## ❤️ Acknowledgements
- Inspired by content-based recommendation system tutorials
- Built for learning **Python, Machine Learning, and interactive UI design**

