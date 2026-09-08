import os
import pickle
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# File paths
caption_file = "data/flickr8k/captions.txt"
feature_file = "data/vgg16_features.npy"
model_file = "models/best_image_caption_model.keras"
tokenizer_file = "models/tokenizer.pkl"
max_length_file = "models/max_length.txt"
output_file = "results/test_predictions.csv"


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


# Load captions
data = pd.read_csv(caption_file)
data["clean_caption"] = data["caption"].apply(clean_caption_function)
data["clean_caption"] = "startseq " + data["clean_caption"] + " endseq"

# Recreate the SAME train / validation / test split
# Split images
images = data["image"].unique()
train_images, temp_images = train_test_split(
    images, train_size=6000, random_state=42, shuffle=True
)
val_images, remaining_images = train_test_split(
    temp_images, train_size=1000, random_state=42, shuffle=True
)
test_images = remaining_images[:1000]

# Load tokenizer and max caption length
with open(tokenizer_file, "rb") as f:
    tokenizer = pickle.load(f)

with open(max_length_file, "rb") as f:
    max_length = pickle.load(f)

# Load VGG16 features
features = np.load(feature_file, allow_pickle=True).item()

# Load best trained model
model = load_model(model_file)


# Generate caption
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


# Remove special tokens
def remove_special_tokens(caption):
    words = caption.split()
    words = [word for word in words if word not in ["startseq", "endseq"]]
    return " ".join(words)


# Generate captions for all test images
results = []
for i, image_name in enumerate(test_images):
    # Make sure feature exists
    if image_name not in features:
        print("Features not found:", image_name)
        continue

    # Get features
    photo = features[image_name]
    # Model expects batch dimension
    photo = np.expand_dims(photo, axis=0)
    # Generate caption
    generated_caption = generate_caption(model, tokenizer, photo, max_length)
    generated_caption = remove_special_tokens(generated_caption)

    # Get 5 real captions
    reference_captions = data[data["image"] == image_name]["clean_caption"].tolist()

    # Remove startseq/endseq
    reference_captions = [
        remove_special_tokens(caption) for caption in reference_captions
    ]

    # Add to results
    results.append(
        {
            "image": image_name,
            "generated_caption": generated_caption,
            "references": " ||| ".join(reference_captions),
        }
    )

    # Show progress every 100 images
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / " f"{len(test_images)} images")

# Save results
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print()
print("Evaluation complete.")
print("Predictions saved to:")
print(output_file)
print()
print("Number of predictions:", len(results_df))
print()
print(results_df.head())
