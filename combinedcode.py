# Install the required libraries
# pip install tensorflow tensorflow-privacy syft cleverhans

import tensorflow as tf
import numpy as np
from cleverhans.tf2.attacks import fast_gradient_method
import syft as sy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# Hook PyTorch to enable PySyft functionalities for Federated Learning
hook = sy.TorchHook(tf)

# Create virtual workers (simulated devices)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Load the dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Split the dataset for federated learning
def split_data(x, y):
    split = len(x) // 2
    return (x[:split], y[:split], x[split:], y[split:])

x_alice, y_alice, x_bob, y_bob = split_data(x_train, y_train)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Set up Differential Privacy Optimizer
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,  # Clipping for each gradient
    noise_multiplier=1.1,  # Privacy noise
    num_microbatches=256,  # Divide batch into microbatches for DP
    learning_rate=0.15
)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Simulate federated learning by federating data
federated_train_loader = sy.FederatedDataLoader(
    [(x_alice, y_alice), (x_bob, y_bob)], batch_size=256, shuffle=True)

# Differentially Private Training in Federated Setting
for epoch in range(5):
    print(f"Epoch {epoch+1}")
    for batch, (data, target) in enumerate(federated_train_loader):
        model.train_on_batch(data, target)

# Compute privacy budget (epsilon)
epsilon, _ = compute_dp_sgd_privacy(
    n=len(x_train),
    batch_size=256,
    noise_multiplier=1.1,
    epochs=5,
    delta=1e-5
)
print(f"Privacy guarantee (epsilon): {epsilon}")

# Generate adversarial examples using CleverHans (FGSM)
def generate_adversarial_examples(model, x_test, eps=0.3):
    x_adv = fast_gradient_method(model, x_test, eps, np.inf)
    return x_adv

x_adv_test = generate_adversarial_examples(model, x_test)

# Evaluate on clean test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Clean Test Accuracy: {accuracy}")

# Evaluate on adversarial examples
adv_loss, adv_acc = model.evaluate(x_adv_test, y_test)
print(f"Adversarial Test Accuracy: {adv_acc}")
