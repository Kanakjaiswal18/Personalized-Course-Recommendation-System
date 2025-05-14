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

## Project Structure
- **`main.py`** – Streamlit app that runs the course recommendation interface with both TF-IDF and deep learning models  
- **`integrated_educational_courses.csv`** – Preprocessed and merged dataset used for generating recommendations  
- **`feedback.csv`** – Stores user ratings and feedback from the app interface  
- **`Data_Integration.ipynb`** – Jupyter notebook demonstrating data integration and schema matching using ADnEV  
- **`requirements.txt`** – Specifies all Python packages required to install and run the application

## Dataset Sources
- [Coursera Courses Dataset 2021](https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021)
- [Udemy Dataset](https://www.kaggle.com/datasets/shailx/course-recommendation-system-dataset)
- [Personalized Recommendation Systems Dataset](https://www.kaggle.com/datasets/alfarisbachmid/personalized-recommendation-systems-dataset)
- [Online Course Student Engagement Metrics](https://www.kaggle.com/datasets/thedevastator/online-course-student-engagement-metrics)

## How to Run the Project
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. Start the Streamlit application:
   ```bash
   streamlit run main.py

## Screenshots

![image](https://github.com/user-attachments/assets/30015e8e-d5ef-4838-acdc-196ecbd0f82f)
![image](https://github.com/user-attachments/assets/94f944c7-1f4c-45bd-9d12-e98a9ef534bb)

