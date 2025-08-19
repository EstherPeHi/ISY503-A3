from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
from prepare_data.preprocessing import Preprocessor
import json

#init Flask
app = Flask(__name__, static_folder='static')

#load trained model
model = tf.keras.models.load_model('best_model.h5')
with open('models/HYBRID/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

#size of input for model
MAX_LENGTH = 300
preprocessor = Preprocessor()

with open("models/HYBRID/threshold.json") as f:
    threshold = json.load(f)["threshold"]

#running HTML from templates.index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json.get('text', '')
        if not user_input:
            raise ValueError("Here is nothing, try again..")

        #input preparation
        clean = preprocessor.clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([clean])
        prepared = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LENGTH)

        #prediction
        prediction = model.predict(prepared)[0][0]
        sentiment = "Positive" if prediction >= threshold else "Negative"
        confidence = round(float(prediction if sentiment == "Positive" else 1 - prediction), 4)

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
