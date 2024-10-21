from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import numpy as np
from PIL import Image  # Explicitly import PIL

app = Flask(__name__)

labels_dict = {
    0: 'Healthy',
    1: 'RedRot',
    2: 'RedRust'
}

# Load the model using pickle
with open('sugarDenseNet_Model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']
        if not imagefile:
            raise ValueError("No file selected.")
        
        image_path = "./userImages/" + imagefile.filename
        imagefile.save(image_path)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        label = model.predict(image)
        classification = '%s (%.2f%%)' % (labels_dict[np.argmax(label)], np.max(label) * 100)
        return render_template('index.html', prediction=classification)
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        return render_template('index.html', error="An unexpected error occurred: " + str(e))

if __name__ == "__main__":
    app.run(port=3000, debug=True)


