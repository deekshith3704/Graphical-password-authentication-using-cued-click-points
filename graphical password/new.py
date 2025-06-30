import os
import cv2
import torch
import sqlite3
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from argon2 import PasswordHasher
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, vit_b_16

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Database for user storage
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    click_points TEXT,
    image_sequence TEXT,
    password_hash TEXT
)
""")
conn.commit()

ph = PasswordHasher()

def store_user(username, click_points, image_sequence, password):
    hashed_password = ph.hash(password)
    cursor.execute("INSERT INTO users (username, click_points, image_sequence, password_hash) VALUES (?, ?, ?, ?)", 
                   (username, str(click_points), str(image_sequence), hashed_password))
    conn.commit()

def authenticate_user(username, entered_click_points, entered_image_sequence, password):
    cursor.execute("SELECT click_points, image_sequence, password_hash FROM users WHERE username=?", (username,))
    user_data = cursor.fetchone()
    if not user_data:
        return False
    stored_click_points, stored_image_sequence, stored_password = user_data
    try:
        ph.verify(stored_password, password)
        return entered_image_sequence == eval(stored_image_sequence) and entered_click_points == eval(stored_click_points)
    except:
        return False

# Dataset Loader
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize
        label = idx % 10  # Example labels (10 classes)
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = ImageDataset("dataset_path", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# DRN Model
class DeepResidualNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepResidualNetwork, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# Train Function
def train_model(model, dataloader, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        accuracy = correct / len(dataset)
        history.append((total_loss, accuracy))
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}")
    return history

# Train and Compare Models
drn_model = DeepResidualNetwork(num_classes=10)
vit_model = VisionTransformer(num_classes=10)

drn_history = train_model(drn_model, dataloader)
vit_history = train_model(vit_model, dataloader)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot([h[1] for h in drn_history], label="DRN Accuracy", marker='o')
plt.plot([h[1] for h in vit_history], label="ViT Accuracy", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Performance Comparison: DRN vs ViT")
plt.show()