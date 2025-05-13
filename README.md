# Personalized Course Recommendation System
This project is a big data-driven recommendation system designed to help students discover relevant online courses tailored to their interests and academic progress. It integrates multiple educational datasets, performs schema matching using deep learning, and offers two recommendation techniques: traditional content-based filtering and a neural embedding-based approach.

## Features
- Integrates diverse educational datasets through schema alignment (ADnEV)
- Supports content filtering utilizing TF-IDF + Cosine Similarity
- Incorporates deep learning-based semantic similarity using PyTorch
- Interactive and user-friendly web application developed with Streamlit
- User feedback mechanism for collecting ratings and comments
- Adapts input handling including partial course name matches

## Technologies Used
- Python
- PyTorch
- Streamlit
- TF-IDF & Cosine Similarity
- Pandas / NumPy / scikit-learn

## Architecture
- Data Integration: Schema matching using a modified ADnEV framework
- Content-Based Recommender: Uses course descriptions, skills, and difficulty levels
- Embedding-Based Recommender: Learns semantic similarity using a custom PyTorch neural network
- UI: Streamlit app with user input, sliders, radio buttons, and feedback form

## Dataset Sources
- [Coursera Courses Dataset 2021](https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021)
- [Udemy Dataset](https://www.kaggle.com/datasets/shailx/course-recommendation-system-dataset)
- [Personalized Recommendation Systems Dataset](https://www.kaggle.com/datasets/alfarisbachmid/personalized-recommendation-systems-dataset)
- [Online Course Student Engagement Metrics](https://www.kaggle.com/datasets/thedevastator/online-course-student-engagement-metrics)

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/course-recommendation-system.git
   cd course-recommendation-system
