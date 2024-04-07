from flask import Flask, render_template, request, send_file
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub

app = Flask(__name__)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        
        # Load and preprocess the uploaded images
        content_image = PIL.Image.open(content_file.stream)
        style_image = PIL.Image.open(style_file.stream)
        
        # Stylize the content image with the style image
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(np.array(content_image) / 255.0, dtype=tf.float32)[tf.newaxis, ...],
                                   tf.constant(np.array(style_image) / 255.0, dtype=tf.float32)[tf.newaxis, ...])[0]
        
        # Convert the stylized image tensor to PIL image and save it
        stylized_image_pil = tensor_to_image(stylized_image)
        stylized_image_path = 'static/stylized_image.png'
        stylized_image_pil.save(stylized_image_path)
        
        return render_template('result.html', image_path=stylized_image_path)


if __name__ == "__main__":
    app.run(debug=True)
