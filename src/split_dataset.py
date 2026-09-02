import os

import pandas as pd
from sklearn.model_selection import train_test_split

images_path = "data/flickr8k/images"
captions_path = "data/flickr8k/captions.txt"

data = pd.read_csv(captions_path)
# Unique images names from captions
images = data["image"].unique()

print(len(images))
# Keep only images that actually exist
images = [image for image in images if os.path.join(images_path, image)]
print("Available images:", len(images))

train_images, temp_images = train_test_split(
    images, train_size=6000, random_state=42, shuffle=True
)
val_images, remaining_images = train_test_split(
    temp_images, train_size=1000, random_state=42, shuffle=True
)

test_images = remaining_images[:1000]


print("Train images:", len(train_images))
print("Validation images:", len(val_images))
print("Test images:", len(test_images))
