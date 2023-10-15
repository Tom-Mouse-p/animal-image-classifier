from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet_v2 import preprocess_input
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model("model/ResNet152V2.h5")

# Define a dictionary to map class indices to class names
class_names = {
    0: "Buffalo",
    1: "Elephant",
    2: "Rhino",
    3: "Zebra",
    # Add more class mappings as needed
}

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Load the image and resize it to match the expected input shape (256x256)
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add an extra dimension for the batch

    # Make predictions
    predictions = model.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=-1)

    # Use the dictionary to get the class name
    predicted_class_name = class_names.get(predicted_class_index[0], "Unknown Class")

    return render_template('index.html', prediction=f'Predicted Class: {predicted_class_name}')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
