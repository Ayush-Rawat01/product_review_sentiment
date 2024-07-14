import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Load the saved models
model_path = 'Models/model_rf.pkl'
cv_path = 'Models/countVectorizer.pkl'
scaler_path = 'Models/scaler.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(cv_path, 'rb') as file:
    cv = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Function to preprocess the input review
def preprocess_review(review):
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    return review

# Streamlit UI
st.title('Product Review Feedback Prediction')

review_input = st.text_area("Enter the review:", "")

if st.button('Predict Feedback'):
    if review_input.strip() == "":
        st.error("Please enter your review.")
    else:
        processed_review = preprocess_review(review_input)
        review_vectorized = cv.transform([processed_review]).toarray()
        review_scaled = scaler.transform(review_vectorized)
        prediction = model.predict(review_scaled)

        if prediction[0] == 1:
            st.success("The feedback is Positive.")
        else:
            st.error("The feedback is Negative.")
