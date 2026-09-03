import pickle
import string

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

captions_file = "data/flickr8k/captions.txt"
feature_file = "data/vgg16_features.npy"
model_file = "models/image_caption_model.keras"

# Load data
data = pd.read_csv(captions_file)
features = np.load(feature_file, allow_pickle=True).item()
model = load_model(model_file)

# Load tokenizer and max length
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/max_length.txt", "r") as f:
    max_length = int(f.read())


#  Clean captions
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

# Split images
images = data["image"].unique()
train_images, temp_images = train_test_split(
    images, train_size=6000, random_state=42, shuffle=True
)
val_images, remaining_images = train_test_split(
    temp_images, train_size=1000, random_state=42, shuffle=True
)
test_images = remaining_images[:1000]


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


# Evaluate first 100 test images
actual = []
predicted = []

# for count, image_name in enumerate(test_images[:100]):
for count, image_name in enumerate(test_images):
    if image_name not in features:
        continue

    photo = features[image_name]
    photo = np.expand_dims(photo, axis=0)
    generated = generate_caption(model, tokenizer, photo, max_length)

    # Remove startseq and endseq
    generated_words = generated.split()
    generated_words = [
        word for word in generated_words if word not in ["startseq", "endseq"]
    ]

    # Get 5 real captions
    real_captions = data[data["image"] == image_name]["clean_caption"].tolist()
    # print(real_captions)
    references = [caption.split() for caption in real_captions]
    # print(references)

    actual.append(references)
    # print(actual)
    predicted.append(generated_words)

    if (count + 1) % 10 == 0:
        print(f"Evaluated {count +1} images")

# BLEU scores
bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

results = {"BLEU-1": bleu1, "BLEU-2": bleu2, "BLEU-3": bleu3, "BLEU-4": bleu4}

results_data = pd.DataFrame([results])
results_data.to_csv("models/bleu_results.csv", index=False)

print("Bleu results saved")

print("BLEU-1:", bleu1)
print("BLEU-2:", bleu2)
print("BLEU-3:", bleu3)
print("BLEU-4:", bleu4)
