from flask import Flask, request, jsonify
import cv2
import numpy as np
from labels import label_mapping
from dotenv import load_dotenv
load_dotenv()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
import google.generativeai as genai
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class PestDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/agrovoicev3.h5')

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224))
        return preprocess_input(resized)

    def predict(self, image):
        image = self.preprocess_image(image)
        img_reshape = image[np.newaxis,...]
        prediction = self.model.predict(img_reshape)
        pred_class = prediction.argmax()
        confidence = prediction[0][pred_class]
        return pred_class, confidence

class GeminiChatBot:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-pro")
        self.chat = self.model.start_chat(history=[])

    def get_response(self, question):
        response = self.chat.send_message(question, stream=True)
        return ' '.join(chunk.text for chunk in response)

detector = PestDetector()
gemini = GeminiChatBot()


@app.route('/')
def start():
    return 'Server is running'

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    print(image)

    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    pred_class, confidence = detector.predict(image)
    pest_name = label_mapping.get(pred_class, 'Unknown')

    response = gemini.get_response(pest_name + "is the pest identified in the image. What are the pesticides that can be used for " + pest_name + "?" + "and give suggestions for the farmer using organic methods and safeguard their crops from " + pest_name )

    data = {
        'pest': pest_name,
        'confidence': float(confidence),
        'response': response
    }

    print(data)
    return jsonify(data)