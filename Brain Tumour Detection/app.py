from flask import Flask, render_template, request
import tensorflow as tf

import keras.utils as image
import numpy as np

app = Flask(__name__)
model= tf.keras.models.load_model('tumor.h5',compile=False)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    file.save(file.filename)
    img=image.load_img(file.filename,target_size=(224,224))
    img = image.img_to_array(img)
    img=img/255
    img = img.reshape((1,) + img.shape)
    a=model.predict(img)
    a=a.astype(np.float32)
    a=a[0][0]
    if a > 0.5:
        return render_template('index.html',msg="The person has a possibility of brain tumour! Consult the doctor ASAP")
    else:
        return render_template('index.html',msg="Relax! The person does not have brain tumour!")



if __name__ == '__main__':
    app.run(debug=True)
