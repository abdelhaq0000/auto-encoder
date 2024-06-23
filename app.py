# from flask import Flask, render_template, request, redirect, url_for
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)

# # Folder to store uploaded images and models
# UPLOAD_FOLDER = 'uploads'
# MODEL_FOLDER = 'models'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Get the encoder and decoder model names
#         encoder_model_name = request.form.get('encoder_model')
#         decoder_model_name = request.form.get('decoder_model')

#         # Get the uploaded file
#         file = request.files['file']
#         if file:
#             # Save the uploaded file
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)

#             return redirect(url_for('result', encoder_model=encoder_model_name, decoder_model=decoder_model_name, filename=file.filename))
#     return render_template('index.html')

# @app.route('/result', methods=['GET'])
# def result():
#     encoder_model_name = request.args.get('encoder_model')
#     decoder_model_name = request.args.get('decoder_model')
#     filename = request.args.get('filename')

#     # Check if any of the required parameters are missing
#     if not all([encoder_model_name, decoder_model_name, filename]):
#         return "Missing parameters, please ensure you upload an image and select both models.", 400

#     # Load the image
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     resized_image = cv2.resize(image, (256, 256))
#     sample_image = resized_image.reshape((1, 256, 256, 1)).astype('float32') / 255.0

#     # Load models
#     encoder_path = os.path.join(MODEL_FOLDER, encoder_model_name + '.h5')
#     decoder_path = os.path.join(MODEL_FOLDER, decoder_model_name + '.h5')

#     if not os.path.exists(encoder_path):
#         return f"Encoder model '{encoder_model_name}' not found.", 404

#     if not os.path.exists(decoder_path):
#         return f"Decoder model '{decoder_model_name}' not found.", 404

#     encoder = load_model(encoder_path)
#     decoder = load_model(decoder_path)

#     # Process the image
#     encoded_img = encoder.predict(sample_image)
#     decoded_img = decoder.predict(encoded_img)
#     encoded_img_2d = np.mean(encoded_img.squeeze(), axis=-1)

#     # Plot the images
#     fig, axs = plt.subplots(1, 3, figsize=(20, 6))

#     axs[0].imshow(sample_image.squeeze(), cmap='gray')
#     axs[0].set_title('Original')

#     axs[1].imshow(encoded_img_2d, cmap='gray')
#     axs[1].set_title('Encoded (2D)')

#     axs[2].imshow(decoded_img.squeeze(), cmap='gray')
#     axs[2].set_title('Decoded')

#     # Save plot to a PNG image in memory
#     img_io = io.BytesIO()
#     plt.savefig(img_io, format='png')
#     img_io.seek(0)
#     plot_url = base64.b64encode(img_io.getvalue()).decode()

#     return render_template('result.html', plot_url=plot_url)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Folder to store uploaded images and models
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
HISTORY_FOLDER = 'history'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload, models, and history folders exist
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, HISTORY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Model stories
model_stories = {
    'encoder_model200_128': "This encoder model has 200 and 128 layers...",
    'encoder_model64_32_16': "This encoder model has 64, 32, and 16 layers...",
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the encoder and decoder model names
        encoder_model_name = request.form.get('encoder_model')
        decoder_model_name = request.form.get('decoder_model')
        model_history_name = request.form.get('model_history')

        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            return redirect(url_for('result', encoder_model=encoder_model_name, decoder_model=decoder_model_name, model_history=model_history_name, filename=file.filename))
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    encoder_model_name = request.args.get('encoder_model')
    decoder_model_name = request.args.get('decoder_model')
    model_history_name = request.args.get('model_history')
    filename = request.args.get('filename')

    # Check if any of the required parameters are missing
    if not all([encoder_model_name, decoder_model_name, model_history_name, filename]):
        return "Missing parameters, please ensure you upload an image, select both models, and choose the model history.", 400

    # Load the image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (256, 256))
    sample_image = resized_image.reshape((1, 256, 256, 1)).astype('float32') / 255.0

    # Load models
    encoder_path = os.path.join(MODEL_FOLDER, encoder_model_name + '.h5')
    decoder_path = os.path.join(MODEL_FOLDER, decoder_model_name + '.h5')

    if not os.path.exists(encoder_path):
        return f"Encoder model '{encoder_model_name}' not found.", 404

    if not os.path.exists(decoder_path):
        return f"Decoder model '{decoder_model_name}' not found.", 404

    encoder = load_model(encoder_path)
    decoder = load_model(decoder_path)

    # Process the image
    encoded_img = encoder.predict(sample_image)
    decoded_img = decoder.predict(encoded_img)
    encoded_img_2d = np.mean(encoded_img.squeeze(), axis=-1)

    # Plot the images
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].imshow(sample_image.squeeze(), cmap='gray')
    axs[0].set_title('Original')

    axs[1].imshow(encoded_img_2d, cmap='gray')
    axs[1].set_title('Encoded (2D)')

    axs[2].imshow(decoded_img.squeeze(), cmap='gray')
    axs[2].set_title('Decoded')

    # Save plot to a PNG image in memory
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()

    # Load model history
    history_path = os.path.join(HISTORY_FOLDER, model_history_name)

    if os.path.exists(history_path):
        model_history = np.load(history_path, allow_pickle=True).item()
    else:
        model_history = None

    return render_template('result.html', 
                           plot_url=plot_url, 
                           encoder_model_name=encoder_model_name,
                           decoder_model_name=decoder_model_name,
                           encoder_story=model_stories.get(encoder_model_name, "No story available."),
                        #    decoder_story=model_stories.get(decoder_model_name, "No story available."),
                           model_history=model_history)

if __name__ == '__main__':
    app.run(debug=True)
