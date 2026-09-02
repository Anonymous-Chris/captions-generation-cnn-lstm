import os

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

image_path = "data/flickr8k/images"

# Load pretrained VGG16
model = VGG16()

# Remove the final classification layer
"""
VGG16() loads a model already trained on ImageNet. 
We remove the final classification layer because we do not want 
VGG16 to tell us “dog,” “car,” etc. Instead, we want the learned 
visual representation from the layer just before classification. 
That feature vector becomes the image input to the LSTM decoder.
"""
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# print(model.summary())

features = {}
# enumerate gives both position and filename, listdir gets all images
for i, filename in enumerate(os.listdir(image_path)):
    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(image_path, filename)

    # Load image at VGG16 input size
    image = load_img(path, target_size=(224, 224))

    # Convert image to NumPy array
    image = img_to_array(image)
    # print(image.shape)

    # Add batch dimension
    # Neural networks normally expect multiple images at once:
    image = np.expand_dims(image, axis=0)
    # print(image.shape)

    # Prepare image for VGG16
    image = preprocess_input(image)
    # print(image)

    # Extract features
    # features contain featuer vectors for all images
    feature = model.predict(image, verbose=0)
    features[filename] = feature[0]

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} images")

print("Total extracted:", len(features))

np.save("data/vgg16_features.npy", features, allow_pickle=True)
print("Features saved.")

"""
Flickr8k image
      ↓
resize to 224×224
      ↓
convert to NumPy
      ↓
preprocess for VGG16
      ↓
VGG16
      ↓
4096-number feature vector
      ↓
store by filename
      ↓
save all features to vgg16_features.npy
"""
