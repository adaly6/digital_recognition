from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

app = Flask(__name__)

# Load the trained model
model = load_model('model/saved_model/mnist_cnn.h5')

def preprocess_image(image):
    # Convert the image to grayscale
    img = ImageOps.grayscale(image)
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert the image to a numpy array
    img = np.array(img)
    # Normalize the image
    img = img / 255.0
    # Reshape the image to fit the model input
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        # Open the image
        img = Image.open(file.stream)
        # Preprocess the image
        img = preprocess_image(img)
        # Make a prediction
        prediction = model.predict(img)
        # Get the digit with the highest probability
        digit = np.argmax(prediction)
        return render_template('index.html', digit=digit)
    return render_template('index.html', digit=None)

if __name__ == '__main__':
    app.run(debug=True)
