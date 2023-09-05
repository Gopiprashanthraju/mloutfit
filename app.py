
from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
import keras
import os
# pricetransform = pickle.load("transformprices.pkl")
script_dir = os.path.dirname(__file__)

# Construct the full file path
model_path = os.path.join(script_dir, "model.h5")
pricemodel = load_model(model_path)
model_path1 = os.path.join(script_dir, "model.h5")
ratingsmodel= load_model(model_path1)
meanp=2318.33610378
sdp=1351.56659225
meanr=3.89976697
sdr=0.23189436
# ratingmodel = joblib.load("model1.pkl")
# Create a Flask app
app = Flask(__name__)

@app.route("/")
def func():
    return render_template("index.html",price_prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    print(request.files)
    print("called")
    if 'image_uploaded' in request.files:
        from PIL import Image

        uploaded_image = request.files['image_uploaded']
        image = Image.open(uploaded_image)

        # Display the image (optional)
        # image.show()
        image=np.array(image.resize((120,120)))
        img = Image.open(uploaded_image)

        # Resize the image to (120, 120) pixels and convert to NumPy array
        img = img.resize((120, 120))
        img_array = np.array(img)
        if img_array.shape[-1] != 3:
            img_array = img_array[:, :, :3]

        img_array = img_array.astype(np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)
        if "prices" in request.form:
            price_prediction = pricemodel.predict(input_data)
            print("me")
            return render_template("index.html",predicted_price="The Predicted Price of the outfit is "+str(int(price_prediction[0][0]*sdp+meanp)))
        else:
            Ratings_prediction=ratingsmodel.predict(input_data)
            print(Ratings_prediction)
            return render_template("index.html",predicted_price="The predicted ratings of the outfit is "+str(round(Ratings_prediction[0][0]*sdr+meanr)))
    else:
        print( "No image uploaded")
        app.logger.debug(f"Price Prediction:")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0')
