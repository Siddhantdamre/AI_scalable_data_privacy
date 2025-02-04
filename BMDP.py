import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the input
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

# Build a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28*28,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Set up the DP optimizer
l2_norm_clip = 1.0  # Clipping value for the gradients
noise_multiplier = 1.1  # Amount of noise to add
num_microbatches = 256  # Number of microbatches (split batches for DP calculation)
learning_rate = 0.15

optimizer = DPAdamGaussianOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate
)

# Compile the model with the DP optimizer
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=256)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Compute privacy guarantee (epsilon)
epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
    n=len(x_train),
    batch_size=256,
    noise_multiplier=noise_multiplier,
    epochs=5,
    delta=1e-5  # Privacy loss
)
print(f'Privacy guarantee (epsilon): {epsilon}')
