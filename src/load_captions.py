import string

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

captions_path = "data/flickr8k/captions.txt"

data = pd.read_csv(captions_path)
# print(data.head(10))
# print()
# print("Total rows:", len(data))
# print("Unique images:", data["image"].nunique())


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
# print(data[["caption", "clean_caption"]].head(10))

# Add special tokens
# data["clean_caption"] = "startseq " + data["clean_caption"] + " endseq"
# print(data[["caption", "clean_caption"]].head(10))

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["clean_caption"])
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

# print("\nFirst 20 words:")
# for word,index in list(tokenizer.word_index.items())[:20]:
#     print(index,word)

# print("\nExample caption:")
# print(data["clean_caption"].iloc[0])
# print("\nTokenized:")
# print(tokenizer.texts_to_sequences([data["clean_caption"].iloc[0]]))

# Find maximum caption length (as LSTM requires all captions of same length)
max_length = max(len(caption.split()) for caption in data["clean_caption"])
print("Maximum caption length:", max_length)
