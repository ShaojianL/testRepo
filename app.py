from flask import Flask, request, render_template, send_from_directory
import os
from style_transfer import perform_style_transfer  # Make sure to refactor your script as a function

app = Flask(__name__)

@app.route('/')
def index():
    # Render the upload form
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    if request.method == 'POST':
        # Save uploaded images
        content_image = request.files['content_image']
        style_image = request.files['style_image']

        content_path = os.path.join('static', 'content.jpg')
        style_path = os.path.join('static', 'style.jpg')
        output_path = os.path.join('static', 'stylized_image.jpg')

        content_image.save(content_path)
        style_image.save(style_path)

        # Perform style transfer
        perform_style_transfer(content_path, style_path, output_path)

        # Render the result page and display the stylized image
        return render_template('result.html', image_path=output_path)

@app.route('/static/<filename>')
def static_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
