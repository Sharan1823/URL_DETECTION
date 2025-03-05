import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')

A = pd.read_csv('/content/phishing_site_urls.csv')

X = A.iloc[:, 0]  
Y = A['Label']  

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y_encoded, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)  
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print(f'Accuracy: {accuracy:.2f}')

def predict_coupon(url):
    url_vectorized = vectorizer.transform([url])

    prediction = model.predict(url_vectorized)

    predicted_label = label_encoder.inverse_transform(prediction)

    return predicted_label[0] 

iface = gr.Interface(
    fn=predict_coupon,
    inputs=[gr.Textbox(label="URL")], 
    outputs="text",
    title="URL Prediction"
)

iface.launch()

