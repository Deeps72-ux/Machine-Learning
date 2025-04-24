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
        error = (y_pred - y_true)
        self.slope -= self.learning_rate * error * x
        self.intercept -= self.learning_rate * error

    def show_equation(self):
        return f"y = {self.slope:.4f} * x + {self.intercept:.4f}"

# Take true slope and intercept as input
try:
    true_slope = float(input("Enter the slope (m): "))
    true_intercept = float(input("Enter the y-intercept: "))
    print(f"Equation entered to train the model:\n y={true_slope}*x+{true_intercept}")
except ValueError:
    print("Invalid input. Exiting.")
    exit()

# --- Training Phase ---

# Generate 180 data points
x_data = [random.uniform(-10, 10) for _ in range(300)]
y_data = [true_slope * x + true_intercept for x in x_data]

# Create and train model
model = SimpleLinearRegression(learning_rate=0.01)
y_predicted = []
print("The learning rate of the model is ",model.learning_rate)

for x, y in zip(x_data, y_data):
    print("Now x = ",x,", y (predicted by model)= ",model.predict(x),", y expected = ",y)
    y_predicted.append(model.predict(x))
    model.update(x, y)

# Sort for proper plotting
combined = sorted(zip(x_data, y_data, y_predicted), key=lambda t: t[0])
x_sorted, y_sorted, y_pred_sorted = zip(*combined)

# Plotting
plt.plot(x_sorted, y_sorted, 'bo', label='True y (Actual Data)', alpha=0.6)
plt.plot(x_sorted, y_pred_sorted, 'r--', label='Predicted y (Model Output)', linewidth=2)
plt.title("True vs Predicted Values (Training Phase) using Simple Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)


# Final learned equation
print("\nFinal Learned Equation by Model:")
print(model.show_equation())
# Display the test data table
print("\nTraining Data Table:\n")
table = [["S.No", "x", "True y", "Predicted y"]]
for i, (x, y_true, y_pred) in enumerate(zip(x_sorted, y_sorted, y_pred_sorted), start=1):
    table.append([i, f"{x:.4f}", f"{y_true:.4f}", f"{y_pred:.4f}"])
print(tabulate(table, headers="firstrow", tablefmt="grid"))

plt.show()


# --- Testing Phase ---

# Generate 100 test data points (x from 10 to 20)
x_test = [random.uniform(10, 20) for _ in range(180)]
y_test_true = [true_slope * x + true_intercept for x in x_test]
y_test_pred = [model.predict(x) for x in x_test]

# Sort for better plotting
combined_test = sorted(zip(x_test, y_test_true, y_test_pred), key=lambda t: t[0])
x_test_sorted, y_test_true_sorted, y_test_pred_sorted = zip(*combined_test)

# Plotting Test Results
plt.figure(figsize=(10, 6))
plt.plot(x_test_sorted, y_test_true_sorted, 'go', label='True y (Actual Test Data)', alpha=0.6)
plt.plot(x_test_sorted, y_test_pred_sorted, 'm--', label='Predicted y (Model Output)', linewidth=2)
plt.title("Testing Phase: True vs Predicted Values")
plt.xlabel("x (Test Data)")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Performance Metrics
mae = sum(abs(y_true - y_pred) for y_true, y_pred in zip(y_test_true_sorted, y_test_pred_sorted)) / len(y_test_true_sorted)
mse = sum((y_true - y_pred) ** 2 for y_true, y_pred in zip(y_test_true_sorted, y_test_pred_sorted)) / len(y_test_true_sorted)
rmse = mse ** 0.5

print("\n--- Model Performance on Test Data ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Display the Test Data Table
print("\nTest Data Table (Testing Phase):\n")
table_test = [["S.No", "x", "True y", "Predicted y"]]
for i, (x, y_true, y_pred) in enumerate(zip(x_test_sorted, y_test_true_sorted, y_test_pred_sorted), start=1):
    table_test.append([i, f"{x:.4f}", f"{y_true:.4f}", f"{y_pred:.4f}"])
print(tabulate(table_test, headers="firstrow", tablefmt="grid"))

plt.show()

