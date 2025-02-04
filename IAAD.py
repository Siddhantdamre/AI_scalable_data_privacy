from cleverhans.tf2.attacks import fast_gradient_method
import numpy as np

# Define a function to create adversarial examples using Fast Gradient Sign Method (FGSM)
def generate_adversarial_examples(model, x, eps=0.3):
    x_adv = fast_gradient_method(model, x, eps, np.inf)
    return x_adv

# Create adversarial examples from test data
x_adv = generate_adversarial_examples(model, x_test)

# Evaluate the model on adversarial examples
adv_loss, adv_acc = model.evaluate(x_adv, y_test)
print(f'Adversarial accuracy: {adv_acc}')

def adversarial_training(model, x_train, y_train, eps=0.3):
    for epoch in range(5):
        for i in range(0, len(x_train), 256):
            x_batch = x_train[i:i+256]
            y_batch = y_train[i:i+256]
            x_adv_batch = generate_adversarial_examples(model, x_batch, eps)
            # Mix adversarial and original examples
            model.train_on_batch(np.concatenate([x_batch, x_adv_batch]),
                                 np.concatenate([y_batch, y_batch]))

# Perform adversarial training
adversarial_training(model, x_train, y_train)
