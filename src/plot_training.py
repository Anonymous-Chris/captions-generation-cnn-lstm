import matplotlib.pyplot as plt
import pandas as pd

history = pd.read_csv("models/training_history.csv")
print(history)

plt.plot(history["loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs validation loss")
plt.legend()
plt.show()
