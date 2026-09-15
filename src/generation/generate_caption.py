import os
import pickle
import string

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical

caption_file = "data/flickr8k/captions.txt"
feature_file = "data/vgg16_features.npy"
model_file = "models/image_caption_model.keras"

# load_captions.py
# Create batches while training
data = pd.read_csv(caption_file)


def clean_caption_function(caption):
    caption = caption.lower()

    # Remove punctuation
    caption = caption.translate(str.maketrans("", "", string.punctuation))

    # Remove words containing numbers
    words = caption.split()
    words = [word for word in words if word.isalpha()]

    # Remove extra spaces
    caption = " ".join(words)
    return caption


data["clean_caption"] = data["caption"].apply(clean_caption_function)

data["clean_caption"] = "startseq " + data["clean_caption"] + " endseq"

# Create tokenizer
# load tokenizer and max length
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/max_length.txt", "r") as f:
    max_length = int(f.read())

# Load features and model
features = np.load(feature_file, allow_pickle=True).item()
model = load_model(model_file)


# Generate Caption
def generate_caption(model, tokenizer, photo, max_length):
    generated_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([photo, sequence], verbose=0)
        predicted_index = np.argmax(prediction)

        predicted_word = tokenizer.index_word.get(predicted_index)
        # print("Predicted word:", predicted_word)
        if predicted_word is None:
            break
        generated_text += " " + predicted_word

        if predicted_word == "endseq":
            break
    return generated_text


# Pick one image
image_name = list(features.keys())[0]
print(image_name)
photo = features[image_name]
print(photo)
# Model expects batch dimension
photo = np.expand_dims(photo, axis=0)
caption = generate_caption(model, tokenizer, photo, max_length)

print("Image:", image_name)
print("Generated caption:", caption)
