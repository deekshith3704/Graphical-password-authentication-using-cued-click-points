

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18 
from transformers import ViTForImageClassification, ViTFeatureExtractor
import matplotlib.pyplot as plt 

# ... (rest of your code for loading dataset, defining models, etc.) ...# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define Deep Residual Network (ResNet-18)
resnet_model = resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
resnet_model = resnet_model.cuda()

# Define Vision Transformer (ViT)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)
vit_model = vit_model.cuda()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
vit_optimizer = optim.Adam(vit_model.parameters(), lr=0.001)

# Lists to store training loss and accuracy for plotting
resnet_train_losses = []
resnet_train_accuracies = []
vit_train_losses = []
vit_train_accuracies = []

# Training function (modified to store loss and accuracy)
def train_model(model, optimizer, model_name, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs) if isinstance(model, torchvision.models.resnet.ResNet) else model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total

        # Store loss and accuracy for plotting
        if model_name == "ResNet":
            resnet_train_losses.append(epoch_loss)
            resnet_train_accuracies.append(epoch_accuracy)
        elif model_name == "ViT":
            vit_train_losses.append(epoch_loss)
            vit_train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# ... (rest of your code for evaluate_model function) ...

def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs) if isinstance(model, torchvision.models.resnet.ResNet) else model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
# Train and evaluate both models
print("Training ResNet...")
train_model(resnet_model, resnet_optimizer, "ResNet")
resnet_accuracy = evaluate_model(resnet_model)
print(f"ResNet Accuracy: {resnet_accuracy * 100:.2f}%")

# Define optimizer for ViT model
vit_optimizer = optim.Adam(vit_model.parameters(), lr=0.001)

print("Training ViT...")
train_model(vit_model, vit_optimizer, "ViT")
vit_accuracy = evaluate_model(vit_model)
print(f"ViT Accuracy: {vit_accuracy * 100:.2f}%")

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(resnet_train_losses, label="ResNet")
plt.plot(vit_train_losses, label="ViT")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(resnet_train_accuracies, label="ResNet")
plt.plot(vit_train_accuracies, label="ViT")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.get_device_name(0))  # Should print your GPU name