import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv('data/encoded_data.csv')
brand_columns = [col for col in df.columns if col.startswith('brand_')]
style_columns = [col for col in df.columns if col.startswith('style_')]

# Define input shapes
image_input_shape = (300, 300, 3)
brand_input_shape = (len(brand_columns),)

# Build model
image_input = layers.Input(shape=image_input_shape, name="image_input")
base_model = tf.keras.applications.MobileNetV2(input_shape=image_input_shape, include_top=False, weights="imagenet")
base_model.trainable = False
image_features = base_model(image_input)
image_features = layers.GlobalAveragePooling2D()(image_features)
image_features = layers.Dense(128, activation="relu")(image_features)

brand_input = layers.Input(shape=brand_input_shape, name="brand_input")
brand_features = layers.Dense(64, activation="relu")(brand_input)

combined = layers.concatenate([image_features, brand_features])
style_output = layers.Dense(len(style_columns), activation="sigmoid", name="style_output")(combined)

model = models.Model(inputs=[image_input, brand_input], outputs=style_output)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

# Print model summary
model.summary()

# Load and preprocess images
image_data = np.array([image.img_to_array(image.load_img(path, target_size=(300, 300))) / 255.0 for path in df["image_path"]])
# Split data
X_train_images, X_val_images, X_train_brands, X_val_brands, y_train, y_val = train_test_split(
    image_data, df[brand_columns], df[style_columns], test_size=0.2, random_state=42
)

# Train model
history = model.fit(
    [X_train_images, X_train_brands], y_train,
    validation_data=([X_val_images, X_val_brands], y_val),
    epochs=50,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate on validation set
y_pred = model.predict([X_val_images, X_val_brands])
y_pred = (y_pred > 0.5).astype(int)  # Apply threshold to convert probabilities to binary labels

# Calculate metrics
print("Hamming Loss:", hamming_loss(y_val, y_pred))
print("Precision (Micro):", precision_score(y_val, y_pred, average="micro"))
print("Recall (Micro):", recall_score(y_val, y_pred, average="micro"))
print("F1-Score (Micro):", f1_score(y_val, y_pred, average="micro"))

# Save model
model.save("style_prediction_model.keras")