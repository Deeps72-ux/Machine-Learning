import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# === Model ===
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

# === Init or Load ===
vectorizer = CountVectorizer()
texts = []
labels = []

model = None
optimizer = None
loss_fn = nn.CrossEntropyLoss()

print("📧 Email urgency classifier - learning from you!\n")

while True:
    email = input("Enter a new email content (or type 'exit'): ").strip()
    if email.lower() == "exit":
        break

    # === Predict if we have a model ===
    if model and len(texts) > 0:
        try:
            vec = vectorizer.transform([email]).toarray()
            inputs = torch.tensor(vec, dtype=torch.float32)
            with torch.no_grad():
                pred = torch.argmax(model(inputs)).item()
                print(f"🤖 I predict this is {'URGENT' if pred == 1 else 'NOT urgent'}")
        except Exception as e:
            print("⚠️ Couldn't predict — likely vocabulary changed:", e)

    # === Feedback ===
    label = int(input("Is this urgent? (1 = yes, 0 = no): "))
    texts.append(email)
    labels.append(label)

    # === Retrain model ===
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    y = torch.tensor(labels)

    model = SimpleClassifier(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(30):
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        loss = loss_fn(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # === Update hardcoded rules ===
    keywords = vectorizer.get_feature_names_out()
    weights = model.linear.weight[1].detach().numpy()
    top_keywords = [k for k, w in sorted(zip(keywords, weights), key=lambda x: -x[1])[:5]]

    with open("filter_rules.py", "w") as f:
        f.write("# Auto-generated keyword filter\n")
        f.write("def is_urgent(text):\n")
        f.write("    text = text.lower()\n")
        for kw in top_keywords:
            f.write(f"    if '{kw}' in text:\n")
            f.write("        return True\n")
        f.write("    return False\n")

    print(f"🔁 Updated rules based on your feedback: {top_keywords}")
