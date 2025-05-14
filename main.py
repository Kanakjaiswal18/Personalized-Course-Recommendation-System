import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Deep Learning Model
class CourseEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(CourseEmbeddingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)

# Main Recommender
class CourseRecommender:
    def __init__(self, courses_df):
        self.df = courses_df
        self.content_matrix = None
        self.course_indices = None
        self.tfidf_vectorizer = None
        self.deep_model = None
        self.embeddings = None

    def fit(self):
        self.df['content'] = ''
        if 'description' in self.df.columns:
            self.df['content'] += (self.df['description'].fillna('') + ' ') * 3
        if 'skills' in self.df.columns:
            self.df['content'] += self.df['skills'].fillna('') + ' '
        self.df['content'] += self.df['course_name'].fillna('')
        if 'difficulty' in self.df.columns:
            self.df['content'] += ' ' + self.df['difficulty'].fillna('')

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        mask = self.df['content'].str.strip() != ''
        if mask.sum() > 0:
            self.df = self.df.loc[mask].reset_index(drop=True)
            self.content_matrix = self.tfidf_vectorizer.fit_transform(self.df['content'])
            self.course_indices = pd.Series(self.df.index, index=self.df['course_name'])
        else:
            st.warning("No courses with content found")
            self.content_matrix = None
            self.course_indices = None

        return self

    def fit_deep_model(self, embedding_dim=128, epochs=20, lr=0.001):
        if self.content_matrix is None:
            st.warning("Content matrix is empty. Run fit() first.")
            return

        X = torch.tensor(self.content_matrix.toarray(), dtype=torch.float32)
        model = CourseEmbeddingNet(X.shape[1], embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X[:, :embedding_dim])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                st.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.deep_model = model
        self.embeddings = model(X).detach().numpy()

    def recommend(self, course_name, n=10):
        if self.content_matrix is None:
            st.warning("Model not fitted.")
            return pd.DataFrame()

        if course_name not in self.course_indices.index:
            matches = [c for c in self.course_indices.index if course_name.lower() in c.lower()]
            if matches:
                course_name = matches[0]
                st.info(f"Using closest match: '{course_name}'")
            else:
                st.warning(f"Course '{course_name}' not found")
                return pd.DataFrame()

        idx = self.course_indices[course_name]
        course_vec = self.content_matrix[idx]
        sim_scores = cosine_similarity(course_vec, self.content_matrix).flatten()
        sim_scores[idx] = -1
        sim_indices = np.argsort(sim_scores)[-n:][::-1]

        similar_courses = self.df.iloc[sim_indices].copy()
        similar_courses['similarity_score'] = sim_scores[sim_indices]

        return similar_courses

    def recommend_deep_learning(self, course_name, n=10):
        if self.embeddings is None or self.deep_model is None:
            st.warning("Deep model not trained. Run fit_deep_model() first.")
            return pd.DataFrame()

        if course_name not in self.course_indices.index:
            matches = [c for c in self.course_indices.index if course_name.lower() in c.lower()]
            if matches:
                course_name = matches[0]
                st.info(f"Using closest match: '{course_name}'")
            else:
                st.warning(f"Course '{course_name}' not found")
                return pd.DataFrame()

        idx = self.course_indices[course_name]
        course_vec = self.embeddings[idx].reshape(1, -1)

        sim_scores = cosine_similarity(course_vec, self.embeddings).flatten()
        sim_scores[idx] = -1
        sim_indices = np.argsort(sim_scores)[-n:][::-1]

        similar_courses = self.df.iloc[sim_indices].copy()
        similar_courses['deep_similarity_score'] = sim_scores[sim_indices]

        return similar_courses


def main():
    st.title("Course Recommendation System")

    st.write("Using dataset: `integrated_educational_courses.csv`")

    # Load dataset
    df = pd.read_csv("integrated_educational_courses.csv", low_memory=False)
    df.rename(columns={"integrated_22_Course Name": "course_name"}, inplace=True)

    recommender = CourseRecommender(df)
    recommender.fit()

    train_dl = st.checkbox("Train Deep Learning Model (May take time)")
    if train_dl:
        st.write("Training deep learning model...")
        recommender.fit_deep_model()

    course_input = st.text_input("Enter a course name to get recommendations:")
    method = st.radio("Recommendation Method:", ["TF-IDF Similarity", "Deep Learning Similarity"])

    num_recommendations = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations") and course_input:
        if method == "Deep Learning Similarity":
            results = recommender.recommend_deep_learning(course_input, num_recommendations)
        else:
            results = recommender.recommend(course_input, num_recommendations)

        if not results.empty:
            # Rename columns for display
            rename_map = {
                'course_name': 'Course Name',
                'similarity_score': 'Similarity Score',
                'deep_similarity_score': 'Similarity Score',
                'integrated_0_level': 'Course Level',
                'integrated_2_Rating': 'Ratings',
                'integrated_7_Course_description': 'Description',
                'integrated_17_University': 'University'
            }

            # Select columns for display
            if 'deep_similarity_score' in results.columns:
                display_cols = ['course_name', 'deep_similarity_score']
            else:
                display_cols = ['course_name', 'similarity_score']

            # Add optional renamed columns if they exist
            for orig, new in rename_map.items():
                if orig not in ['course_name', 'similarity_score', 'deep_similarity_score'] and orig in results.columns:
                    display_cols.append(orig)

            # Rename for display
            results_display = results[display_cols].rename(columns={col: rename_map[col] for col in display_cols if col in rename_map})

            st.subheader("Top Recommendations:")
            st.dataframe(results_display)

        else:
            st.warning("No recommendations found.")
    st.markdown("---")
    st.subheader("⭐ User Feedback")
    stars = st.slider("How would you rate this app?", min_value=1, max_value=5, value=5)
    comments = st.text_area("Leave your comments:")

    if st.button("Submit Feedback"):
        feedback_entry = pd.DataFrame({'Rating': [stars], 'Comments': [comments]})
        feedback_file = "feedback.csv"
        if os.path.exists(feedback_file):
            feedback_entry.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_entry.to_csv(feedback_file, index=False)
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()

