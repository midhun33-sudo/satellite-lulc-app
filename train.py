import os
import ssl
from model_utils import CLASSES, build_model
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Bypass SSL verification issues on some local environments when downloading EuroSAT
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    print("Downloading / loading EuroSAT dataset...")
    full_dataset = datasets.EuroSAT(root="./data", download=True)

    indices = list(range(len(full_dataset)))
    targets = [full_dataset[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=targets, random_state=42
    )

    train_data = Subset(
        datasets.EuroSAT(root="./data", transform=train_transform), train_idx
    )
    val_data = Subset(
        datasets.EuroSAT(root="./data", transform=val_transform), val_idx
    )

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False, num_workers=0
    )

    model = build_model(num_classes=len(CLASSES), pretrained=True).to(device)

    # Freeze backbone layers for fast CPU training; train only the classifier head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    epochs = 4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        epoch_loss = running_loss / len(train_data)
        print(
            f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f} | Val Accuracy ="
            f" {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/eurosat_resnet18.pth")
            print(f"--> Saved new best model (Val Acc: {best_acc:.4f})")


if __name__ == "__main__":
    main()