from tensorflow.keras.layers import LSTM, Add, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model

vocab_size = 8778
max_length = 37

# Image branch
image_input = Input(shape=(4096,), name="image_input")
image_features = Dropout(0.5)(image_input)
image_features = Dense(256, activation="relu")(image_features)

# Caption branch
caption_input = Input(shape=(max_length,), name="caption_input")
# The Embedding layer converts each word into a 256-number vector.
caption_features = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(
    caption_input
)
caption_features = Dropout(0.5)(caption_features)
caption_features = LSTM(256)(caption_features)

"""
Input → defines what data enters the network
Dense → regular fully connected neural-network layer
Dropout → randomly removes some values during training to reduce overfitting
Embedding → converts word IDs into vectors
LSTM → understands/sequences words over time
Add → combines image information and caption information

Image branch:
image → 4096 → Dense → 256 numbers

Caption branch:
caption → Embedding → LSTM → 256 numbers

i.e.

Image information   → [256 values]
Caption information → [256 values]
"""

# Merge image and caption
decoder = Add()([image_features, caption_features])
# This lets the neural network learn relationships between the combined image and caption information.
# This layer predicts the next word
decoder = Dense(256, activation="relu")(decoder)
output = Dense(vocab_size, activation="softmax")(decoder)

# Create model
model = Model(inputs=[image_input, caption_input], outputs = output)
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()