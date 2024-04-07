from flask import Flask, render_template, request, send_file
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
from style_transfer_2 import load_img, StyleContentModel, tensor_to_image, train_step, train_step_2, high_pass_x_y, total_variation_loss, clip_0_1, style_content_loss
from style_transfer_2 import vgg_layers, gram_matrix, style_content_loss
import time
import IPython.display as display


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
        for i in range(3):  # You might need to adjust the number of iterations
            train_step(image, extractor, style_targets, content_targets, len(style_layers), len(content_layers), opt)

        # Convert the result tensor to an image and save
        stylized_image = tensor_to_image(image)
        stylized_image_path = 'static/stylized_image.png'
        stylized_image.save(stylized_image_path)

        ##########################################################

        x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
        x = tf.image.resize(x, (224, 224))
        vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        prediction_probabilities = vgg(x)
        prediction_probabilities.shape

        predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
        [(class_name, prob) for (number, class_name, prob) in predicted_top_5]

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        print()
        for layer in vgg.layers:
            print(layer.name)

        content_layers = ['block5_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        style_extractor = vgg_layers(style_layers)
        style_outputs = style_extractor(style_image*255)

        #Look at the statistics of each layer's output
        for name, output in zip(style_layers, style_outputs):
            print(name)
            print("  shape: ", output.numpy().shape)
            print("  min: ", output.numpy().min())
            print("  max: ", output.numpy().max())
            print("  mean: ", output.numpy().mean())
            print()

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']
        image = tf.Variable(content_image)
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        style_weight=1e-2
        content_weight=1e4  

        return render_template('result.html', image_path=stylized_image_path)


if __name__ == "__main__":
    app.run(debug=True)