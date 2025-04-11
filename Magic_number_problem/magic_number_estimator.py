# The user can decide on a number and call it the magic number. My neural network model will try to predict it each time and the user can give a score based on which it will try to get closer to this magic number
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define a tiny model with 1 input -> hidden -> 1 output
class MagicNumberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output is score [0, 1]
        )

    def forward(self, x):
        return self.net(x)

model = MagicNumberNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("🎯 Magic Number Learner (Neural Net Version)")
print("Instructions: Think of a magic number between 1 and 100.")
print("I'll guess, and you rate my guess from 0 (far) to 1 (very close).")

# Learning loop
for step in range(1, 100):
    guess = random.randint(1, 100)
    x = torch.tensor([[guess]], dtype=torch.float32)

    with torch.no_grad():
        pred_score = model(x).item()
    print(f"\n🔢 Round {step}: My guess is... {guess} (Predicted score: {pred_score:.2f})")

    try:
        actual_score = float(input("🎯 Your feedback (0 to 1): "))
        actual_score = max(0.0, min(1.0, actual_score))  # clamp
    except:
        print("⚠️ Invalid input. Using 0.")
        actual_score = 0.0

    # Training step
    model.train()
    optimizer.zero_grad()
    target = torch.tensor([[actual_score]], dtype=torch.float32)
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
