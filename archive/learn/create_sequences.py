import string

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

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
data["clean_caption"] = "startseq " + data["clean_caption"] + " endseq"
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

caption = data["clean_caption"].iloc[0]
sequence = tokenizer.texts_to_sequences([caption])[0]
print("\nCaption:")
print(caption)

print("\nEncoded caption:")
print(sequence)

x = []
y = []

# Create training examples
for i in range(1, len(sequence)):
    input_sequence = sequence[:i]
    # print(input_sequence)
    output_word = sequence[i]
    # print(output_word)

    # Pad to make them same dimensions
    input_sequence = pad_sequences([input_sequence], maxlen=max_length)[0]
    # print(input_sequence)

    # Creates a vector containing zeroes except at the position of output_word
    # One hot encoding
    output_word = to_categorical(output_word, num_classes=vocab_size)
    # print(output_word)

    x.append(input_sequence)
    y.append(output_word)

# print(len(x))
# print(x)
# print(len(y))
x = np.array(x)
y = np.array(y)
# print(x)
# print(y)

# 18 training examples, each input is padded to 37 tokens
print("\nInput shape:", x.shape)
# 18 expected outputs, each output is one hot vector across 8778 vocabulary words and can predict any one
print("Output shape:", y.shape)

print("\nFirst training input:")
print(x[0])

print("\nExpected word index:")
# Provides index of maximum value
print(np.argmax(y[0]))

print("Expected word:", tokenizer.index_word[np.argmax(y[0])])
