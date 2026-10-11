
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. Generate the dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
y = y.reshape(-1, 1)

np.random.seed(42)

# 2. Initialize the neural network: 2 -> 8 -> 1
W1 = np.random.randn(2, 8) * 0.5
b1 = np.zeros((1, 8))

W2 = np.random.randn(8, 1) * 0.5
b2 = np.zeros((1, 1))


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def forward(X):
    # Hidden layer
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU

    # Output layer
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    return Z1, A1, A2


def calculate_loss(y, predictions):
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)

    return -np.mean(
        y * np.log(predictions)
        + (1 - y) * np.log(1 - predictions)
    )


# 3. Train the network
learning_rate = 0.1
epochs = 10000
loss_history = []
m = X.shape[0]

for epoch in range(epochs):

    # FORWARD PASS
    Z1, A1, A2 = forward(X)

    # LOSS
    loss = calculate_loss(y, A2)
    loss_history.append(loss)

    # BACKPROPAGATION
    dZ2 = (A2 - y) / m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # GRADIENT DESCENT
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f}")


# 4. Evaluate the trained network
_, _, probabilities = forward(X)

predictions = (probabilities >= 0.5).astype(int)
accuracy = np.mean(predictions == y)

print("\nTraining complete!")
print(f"Initial loss: {loss_history[0]:.4f}")
print(f"Final loss:   {calculate_loss(y, forward(X)[2]):.4f}")
print(f"Training accuracy: {accuracy * 100:.2f}%")


# 5. Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network Learning")
plt.grid(True, alpha=0.3)
plt.tight_layout()


# 6. Visualize the learned decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]
_, _, grid_probabilities = forward(grid)
Z = grid_probabilities.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(
    xx, yy, Z,
    levels=np.linspace(0, 1, 11),
    cmap="coolwarm",
    alpha=0.4
)

plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

plt.scatter(
    X[:, 0], X[:, 1],
    c=y.ravel(),
    cmap="coolwarm",
    edgecolors="black",
    s=35
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Learned Decision Boundary")
plt.tight_layout()
plt.show()