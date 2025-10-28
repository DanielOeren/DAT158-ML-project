import json
import pathlib
import tensorflow as tf

# ----- Paths -----
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fruits-360_100x100"
TRAIN_DIR = str(DATA_ROOT / "Training")
TEST_DIR  = str(DATA_ROOT / "Test")

# Sanity check (remind user to download)
if not (DATA_ROOT / "Training").exists():
    raise SystemExit("Dataset not found. Run: python src/get_data.py")

# ----- Config -----
IMG_SIZE = (100, 100)   # 100x100 branch
BATCH = 64
SEED = 1337

# ----- Datasets -----
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, label_mode="categorical"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, label_mode="categorical", shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Detected {num_classes} classes.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(2048).prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ----- Model -----
base = tf.keras.applications.EfficientNetB0(
    include_top=False, input_shape=IMG_SIZE + (3,), weights="imagenet"
)
base.trainable = False  # warmup

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

inp = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.efficientnet.preprocess_input(inp)
x = augment(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

ckpt_path = ROOT / "fruit_model.h5"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_accuracy"),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
]

print("Training (frozen base)…")
model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=callbacks)

print("Fine-tuning (unfreeze base)…")
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)
model.fit(train_ds, validation_data=test_ds, epochs=10, callbacks=callbacks)

# Evaluate & save artifacts
model.load_weights(str(ckpt_path))
print("Final evaluation:")
model.evaluate(test_ds, verbose=2)

with open(ROOT / "class_names.json", "w") as f:
    json.dump(class_names, f)
print("Saved:", ckpt_path, "and", ROOT / "class_names.json")
