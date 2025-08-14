from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
from prepare_data.preprocessing import Preprocessor

#init Flask
app = Flask(__name__)

#load trained model
model = tf.keras.models.load_model('best_model.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
#size of input
MAX_LENGTH = 350
preprocessor = Preprocessor()

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
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 4)

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
    app.run(debug=True)
