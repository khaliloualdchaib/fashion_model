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
tops_fit_columns = [col for col in df.columns if col.startswith('tops_fit')]
pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
bottoms_fit_columns = [col for col in df.columns if col.startswith('bottoms_fit')]
sleeve_type_columns = [col for col in df.columns if col.startswith('sleeve_type')]
more_attributes_columns = [col for col in df.columns if col.startswith('more_attributes')]

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

# Concatenate image and brand features
combined = layers.concatenate([image_features, brand_features])

# Task-specific outputs
style_output = layers.Dense(len(style_columns), activation="sigmoid", name="style_output")(combined)
tops_fit_output = layers.Dense(len(tops_fit_columns), activation="sigmoid", name="tops_fit_output")(combined)
pattern_output = layers.Dense(len(pattern_columns), activation="sigmoid", name="pattern_output")(combined)
bottoms_fit_output = layers.Dense(len(bottoms_fit_columns), activation="sigmoid", name="bottoms_fit_output")(combined)
sleeve_type_output = layers.Dense(len(sleeve_type_columns), activation="sigmoid", name="sleeve_type_output")(combined)
more_attributes_output = layers.Dense(len(more_attributes_columns), activation="sigmoid", name="more_attributes_output")(combined)

# Build the multi-task model
model = models.Model(
    inputs=[image_input, brand_input],
    outputs=[style_output, tops_fit_output, pattern_output, bottoms_fit_output, sleeve_type_output, more_attributes_output]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss={
        "style_output": "binary_crossentropy",
        "tops_fit_output": "binary_crossentropy",
        "pattern_output": "binary_crossentropy",
        "bottoms_fit_output": "binary_crossentropy",
        "sleeve_type_output": "binary_crossentropy",
        "more_attributes_output": "binary_crossentropy",
    },
    metrics={
        "style_output": ["binary_accuracy"],
        "tops_fit_output": ["binary_accuracy"],
        "pattern_output": ["binary_accuracy"],
        "bottoms_fit_output": ["binary_accuracy"],
        "sleeve_type_output": ["binary_accuracy"],
        "more_attributes_output": ["binary_accuracy"],
    }
)

# Print model summary
model.summary()

# Load and preprocess images
image_data = np.array([image.img_to_array(image.load_img(path, target_size=(300, 300))) / 255.0 for path in df["image_path"]])

# Split data
X_train_images, X_val_images, X_train_brands, X_val_brands, y_train, y_val = train_test_split(
    image_data, df[brand_columns], df[style_columns + tops_fit_columns + pattern_columns + bottoms_fit_columns + sleeve_type_columns + more_attributes_columns], test_size=0.2, random_state=42
)

# Prepare target labels
y_train_style = y_train[style_columns]
y_train_tops_fit = y_train[tops_fit_columns]
y_train_pattern = y_train[pattern_columns]
y_train_bottoms_fit = y_train[bottoms_fit_columns]
y_train_sleeve_type = y_train[sleeve_type_columns]
y_train_more_attributes = y_train[more_attributes_columns]

y_val_style = y_val[style_columns]
y_val_tops_fit = y_val[tops_fit_columns]
y_val_pattern = y_val[pattern_columns]
y_val_bottoms_fit = y_val[bottoms_fit_columns]
y_val_sleeve_type = y_val[sleeve_type_columns]
y_val_more_attributes = y_val[more_attributes_columns]

# Train model
history = model.fit(
    [X_train_images, X_train_brands],
    [y_train_style, y_train_tops_fit, y_train_pattern, y_train_bottoms_fit, y_train_sleeve_type, y_train_more_attributes],
    validation_data=(
        [X_val_images, X_val_brands],
        [y_val_style, y_val_tops_fit, y_val_pattern, y_val_bottoms_fit, y_val_sleeve_type, y_val_more_attributes],
    ),
    epochs=50,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate on validation set
y_pred = model.predict([X_val_images, X_val_brands])
y_pred_style = (y_pred[0] > 0.5).astype(int)  # Apply threshold for style
y_pred_tops_fit = (y_pred[1] > 0.5).astype(int)  # Apply threshold for tops_fit
y_pred_pattern = (y_pred[2] > 0.5).astype(int)  # Apply threshold for pattern
y_pred_bottoms_fit = (y_pred[3] > 0.5).astype(int)  # Apply threshold for bottoms_fit
y_pred_sleeve_type = (y_pred[4] > 0.5).astype(int)  # Apply threshold for sleeve_type
y_pred_more_attributes = (y_pred[5] > 0.5).astype(int)  # Apply threshold for more_attributes

# Calculate metrics for each task
print("Style - Hamming Loss:", hamming_loss(y_val_style, y_pred_style))
print("Style - Precision (Micro):", precision_score(y_val_style, y_pred_style, average="micro"))
print("Style - Recall (Micro):", recall_score(y_val_style, y_pred_style, average="micro"))
print("Style - F1-Score (Micro):", f1_score(y_val_style, y_pred_style, average="micro"))

print("Tops Fit - Hamming Loss:", hamming_loss(y_val_tops_fit, y_pred_tops_fit))
print("Tops Fit - Precision (Micro):", precision_score(y_val_tops_fit, y_pred_tops_fit, average="micro"))
print("Tops Fit - Recall (Micro):", recall_score(y_val_tops_fit, y_pred_tops_fit, average="micro"))
print("Tops Fit - F1-Score (Micro):", f1_score(y_val_tops_fit, y_pred_tops_fit, average="micro"))

print("Pattern - Hamming Loss:", hamming_loss(y_val_pattern, y_pred_pattern))
print("Pattern - Precision (Micro):", precision_score(y_val_pattern, y_pred_pattern, average="micro"))
print("Pattern - Recall (Micro):", recall_score(y_val_pattern, y_pred_pattern, average="micro"))
print("Pattern - F1-Score (Micro):", f1_score(y_val_pattern, y_pred_pattern, average="micro"))

print("Bottoms Fit - Hamming Loss:", hamming_loss(y_val_bottoms_fit, y_pred_bottoms_fit))
print("Bottoms Fit - Precision (Micro):", precision_score(y_val_bottoms_fit, y_pred_bottoms_fit, average="micro"))
print("Bottoms Fit - Recall (Micro):", recall_score(y_val_bottoms_fit, y_pred_bottoms_fit, average="micro"))
print("Bottoms Fit - F1-Score (Micro):", f1_score(y_val_bottoms_fit, y_pred_bottoms_fit, average="micro"))

print("Sleeve Type - Hamming Loss:", hamming_loss(y_val_sleeve_type, y_pred_sleeve_type))
print("Sleeve Type - Precision (Micro):", precision_score(y_val_sleeve_type, y_pred_sleeve_type, average="micro"))
print("Sleeve Type - Recall (Micro):", recall_score(y_val_sleeve_type, y_pred_sleeve_type, average="micro"))
print("Sleeve Type - F1-Score (Micro):", f1_score(y_val_sleeve_type, y_pred_sleeve_type, average="micro"))

print("More Attributes - Hamming Loss:", hamming_loss(y_val_more_attributes, y_pred_more_attributes))
print("More Attributes - Precision (Micro):", precision_score(y_val_more_attributes, y_pred_more_attributes, average="micro"))
print("More Attributes - Recall (Micro):", recall_score(y_val_more_attributes, y_pred_more_attributes, average="micro"))
print("More Attributes - F1-Score (Micro):", f1_score(y_val_more_attributes, y_pred_more_attributes, average="micro"))

# Save model
model.save("multi_task_model.keras")