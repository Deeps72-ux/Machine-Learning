import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.slope = 0.0
        self.intercept = 0.0
        self.learning_rate = learning_rate

    def predict(self, x):
        return self.slope * x + self.intercept

    def update(self, x, y_true):
        y_pred = self.predict(x)
        error = y_pred - y_true

        # Gradient descent update
        self.slope -= self.learning_rate * error * x
        self.intercept -= self.learning_rate * error

    def show_equation(self):
        print(f"Current model: y = {self.slope:.4f} * x + {self.intercept:.4f}")

# Interactive Training Loop
model = SimpleLinearRegression(learning_rate=0.1)

x_data = []
y_true_data = []
y_pred_data = []

print("Linear Regression Interactive Trainer")
print("Type 'exit' to stop the training.\n")

while True:
    try:
        x_input = input("Enter input x: ")
        if x_input.lower() == 'exit':
            break

        x = float(x_input)
        y_pred = model.predict(x)
        print(f"Predicted y: {y_pred:.4f}")

        y_true_input = input("Enter true y: ")
        if y_true_input.lower() == 'exit':
            break

        y_true = float(y_true_input)
        model.update(x, y_true)
        model.show_equation()
        print("Learning rate is ", model.learning_rate)
        print("\n--- Next Iteration ---\n")

        # Store data for plotting
        x_data.append(x)
        y_true_data.append(y_true)
        y_pred_data.append(model.predict(x))  # Updated prediction after model update

    except ValueError:
        print("Invalid input. Please enter numerical values.\n")

# Plotting
plt.scatter(x_data, y_true_data, color='blue', label='True Data Points')
plt.plot(x_data, y_pred_data, color='red', label='Predicted Line (model fit)')
plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
