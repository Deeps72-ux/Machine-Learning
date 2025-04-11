import matplotlib.pyplot as plt
import random
import time
from tabulate import tabulate

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
        self.slope -= self.learning_rate * error * x
        self.intercept -= self.learning_rate * error

    def show_equation(self):
        return f"y = {self.slope:.4f} * x + {self.intercept:.4f}"

# Take true slope and intercept as input
try:
    true_slope = float(input("Enter the true slope (m): "))
    true_intercept = float(input("Enter the true intercept (b): "))
except ValueError:
    print("Invalid input. Exiting.")
    exit()

# Generate 100 data points
x_data = [random.uniform(-10, 10) for _ in range(180)]
print("Now x is ",x_data)
y_data = [true_slope * x + true_intercept for x in x_data]

# Create and train model
model = SimpleLinearRegression(learning_rate=0.01)
y_predicted = []

for x, y in zip(x_data, y_data):
    model.update(x, y)
    y_predicted.append(model.predict(x))

# Sort for proper plotting
combined = sorted(zip(x_data, y_data, y_predicted), key=lambda t: t[0])
x_sorted, y_sorted, y_pred_sorted = zip(*combined)

# Plotting
plt.plot(x_sorted, y_sorted, 'bo', label='True y (Actual Data)', alpha=0.6)
plt.plot(x_sorted, y_pred_sorted, 'r--', label='Predicted y (Model Output)', linewidth=2)
plt.title("True vs Predicted Values using Simple Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)


# Final learned equation
print("\nFinal Learned Equation by Model:")
print(model.show_equation())
# Display the test data table
print("\nTest Data Table:\n")
table = [["S.No", "x", "True y", "Predicted y"]]
for i, (x, y_true, y_pred) in enumerate(zip(x_sorted, y_sorted, y_pred_sorted), start=1):
    table.append([i, f"{x:.4f}", f"{y_true:.4f}", f"{y_pred:.4f}"])
print(tabulate(table, headers="firstrow", tablefmt="grid"))

plt.show()
