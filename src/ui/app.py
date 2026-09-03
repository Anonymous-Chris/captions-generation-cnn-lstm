import pickle

import gradio as gr
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_file = "models/image_caption_model.keras"

caption_model = load_model(model_file)
# Create tokenizer
# load tokenizer and max length
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/max_length.txt", "r") as f:
    max_length = int(f.read())

# Load VGG16
vgg = VGG16()
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)


def extract_feature(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg.predict(image, verbose=0)
    return feature


def generate_caption(photo):
    generated_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = caption_model.predict([photo, sequence], verbose=0)
        predicted_index = np.argmax(prediction)

        predicted_word = tokenizer.index_word.get(predicted_index)
        # print("Predicted word:", predicted_word)
        if predicted_word is None:
            break

        if predicted_word == "endseq":
            break

        generated_text += " " + predicted_word
    caption = generated_text.replace("startseq", "")
    return caption


def caption_image(image):
    feature = extract_feature(image)
    caption = generate_caption(feature)
    return caption

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator",
    description="Upload an image and the CNN-LSTM model will describe it.",
)

demo.launch()
