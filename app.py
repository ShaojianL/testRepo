from flask import Flask, render_template, request, send_file
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
from style_transfer_2 import load_img, StyleContentModel, tensor_to_image, train_step


app = Flask(__name__)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(image_data):
    max_dim = 512
    img = tf.image.decode_image(image_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/transfer', methods=['POST'])
# def transfer():
#     if request.method == 'POST':
#         content_file = request.files['content_image']
#         style_file = request.files['style_image']
        
#         # Read the uploaded files into memory
#         content_image_data = content_file.read()
#         style_image_data = style_file.read()
        
#         # Load and preprocess the uploaded images
#         content_image = load_img(content_image_data)
#         style_image = load_img(style_image_data)
        
#         # Stylize the content image with the style image
#         stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        
#         # Convert the stylized image tensor to PIL image and save it
#         stylized_image_pil = tensor_to_image(stylized_image)
#         stylized_image_path = 'static/stylized_image.png'
#         stylized_image_pil.save(stylized_image_path)
        
#         return render_template('result.html', image_path=stylized_image_path)

@app.route('/transfer', methods=['POST'])
def transfer():
    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']
        
        # Convert the uploaded files into a format suitable for the style transfer functions
        content_image = load_img(content_file.read())
        style_image = load_img(style_file.read())
        
        # Define style and content layers
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        # Create an instance of the style and content model
        extractor = StyleContentModel(style_layers, content_layers)

        # Extract style and content targets
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        # Initialize the target image for optimization
        image = tf.Variable(content_image)

        # Setup optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        # Run the style transfer
        for i in range(10):  # You might need to adjust the number of iterations
            train_step(image, extractor, style_targets, content_targets, len(style_layers), len(content_layers), opt)

        # Convert the result tensor to an image and save
        stylized_image = tensor_to_image(image)
        stylized_image_path = 'static/stylized_image.png'
        stylized_image.save(stylized_image_path)
        
        return render_template('result.html', image_path=stylized_image_path)


if __name__ == "__main__":
    app.run(debug=True)