import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming your input shape is (time_steps, num_channels)
# For example: (500, 3) for 500 timesteps and 3 force plate signals (Fx, Fy, Fz)

time_steps = 500  # adjust depending on your recording
num_channels = 3  # adjust depending on how many signals (Fx, Fy, Fz, COPx, COPy, etc.)
num_classes = 20  # 20 people

model = models.Sequential([
    layers.Input(shape=(time_steps, num_channels)),

    layers.Conv1D(filters=32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # assuming labels are integer encoded
    metrics=['accuracy']
)

model.summary()
