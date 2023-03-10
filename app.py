import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np




app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Load model
cauliflower_model = tf.keras.models.load_model(STATIC_FOLDER + "/models/" + "Cauliflower_best_model.h5")

sizeofimage = 256

# Preprocess an image
def preprocess(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [sizeofimage, sizeofimage])
    image /= 255.0  # normalize
    return image

# Read the image from path
def load(path):
    image = tf.io.read_file(path)
    return preprocess(image)


# Add a pre-screenr to catch image that doesn't belong to the class
# Load pre-trained ResNet50 model
pre_trained_model = tf.keras.applications.ResNet50(weights='imagenet')

# Define a function to preprocess the image and predict the class using the pre-trained model
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = pre_trained_model.predict(x)
    return tf.keras.applications.resnet50.decode_predictions(preds, top=1)[0][0][1]

# Predict & classify image
def classify(model, image_path):
    # Add a pre-screener to catch images that are not cauliflower or do not belong to the same category
    if predict_image(image_path) != "cauliflower":
        return "Not a cauliflower image", 0.0

    finalimage = load(image_path)
    finalimage = tf.reshape(
        finalimage, (1, sizeofimage, sizeofimage, 3)
    )

    probability = cauliflower_model.predict(finalimage)
    label = "Healthy" if probability[0][0] <= 0.5 \
        else "Diseased"

    classified_probability = probability[0][0] \
        if probability[0][0] >= 0.5 \
        else 1 - probability[0][0]

    return label, classified_probability





# index page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "POST":
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)
        label = predict_image(upload_image_path)
        label, probability = classify(cauliflower_model, upload_image_path)
        probability = round((probability * 100), 2)

    else:
        if request.method == "GET":
            return render_template("index.html")

    return render_template(
        "classify.html", imagefile=file.filename, label=label, prob=probability
    )



@app.route('/classify/resize_image/<filename>/<int:width>/<int:height>')
def resize_image(filename, width, height):
    # Load the image
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    img = Image.open(file_path)

    # Resize the image
    img = img.resize((width, height))

    # Save the resized image to a file
    resized_filename = f"resized_{filename}"
    img.save(resized_filename)

    # Return the filename of the resized image
    return resized_filename 



@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
