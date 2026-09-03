import os
import string

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import Sequence, to_categorical

# load_captions.py
# Create batches while training
caption_file = "data/flickr8k/captions.txt"
feature_file = "data/vgg16_features.npy"

data = pd.read_csv(caption_file)
features = np.load(feature_file, allow_pickle=True).item()


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
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["clean_caption"])
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in data["clean_caption"])
print("Maximum caption length:", max_length)

print(data[0:5])


# split_dataset.py
class CaptionDataGenerator(Sequence):
    def __init__(
        self,
        dataframe,
        features,
        tokenizer,
        max_length,
        vocab_size,
        batch_size=32,
        **kwargs
    ):
        # Dataframe is whole caption table, batch_images is a part of images
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.images = dataframe["image"].unique()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.images[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        x_image = []
        x_sequence = []
        y = []

        for image_name in batch_images:
            if image_name not in self.features:
                continue

            image_feature = self.features[image_name]
            # Returns only those that are true or match
            captions = self.dataframe[self.dataframe["image"] == image_name][
                "clean_caption"
            ]

            for caption in captions:
                sequence = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(sequence)):
                    input_sequence = sequence[:i]
                    output_word = sequence[i]
                    input_sequence = pad_sequences(
                        [input_sequence], maxlen=self.max_length
                    )[0]
                    output_word = to_categorical(
                        output_word, num_classes=self.vocab_size
                    )

                    x_image.append(image_feature)
                    x_sequence.append(input_sequence)
                    y.append(output_word)

        return ((np.array(x_image), np.array(x_sequence)), np.array(y))


generator = CaptionDataGenerator(
    dataframe=data,
    features=features,
    tokenizer=tokenizer,
    max_length=max_length,
    vocab_size=vocab_size,
    batch_size=32,
)

print("Number of batches:", len(generator))

(x_image, x_sequence), y = generator[0]
print("Image input shape:", x_image.shape)
print("Sequence input shape:", x_sequence.shape)
print("Output shape:", y.shape)
