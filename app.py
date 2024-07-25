from flask import Flask, jsonify, request
import numpy as np
import json

app = Flask(__name__)

# Load pre-computed data
with open('model_data.json', 'r') as f:
    model_data = json.load(f)

vocabulary = model_data['vocabulary']
idf = np.array(model_data['idf'])
feature_log_prob = np.array(model_data['feature_log_prob'])
class_log_prior = np.array(model_data['class_log_prior'])
classes = np.array(model_data['classes'])

def compute_tfidf(text):
    words = text.lower().split()
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    
    tfidf = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        if word in tf:
            tfidf[i] = tf[word] * idf[i]
    
    return tfidf

def predict(text):
    tfidf = compute_tfidf(text)
    log_prob_x = np.log(tfidf + 1)  # Add 1 to avoid log(0)
    log_prob = class_log_prior + np.dot(log_prob_x, feature_log_prob.T)
    predicted_class = classes[np.argmax(log_prob)]
    return predicted_class == 'spam'  # True if spam, False otherwise

@app.route('/')
def home():
    return 'Spam Detection Server is Running!'

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        comment = data.get('comment', '')
        result = predict(comment)
        return jsonify({'result': result})
    except Exception as e:
        app.logger.error(f"Error in predict_api: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500