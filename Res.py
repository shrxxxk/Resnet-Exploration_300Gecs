import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import cv2
import os
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    ToTensor, 
    Normalize, 
    Compose,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter
)

# --- TOP-LEVEL DEFINITIONS ---

# Set random seed for reproducibility
RANDOM_SEED = 2025
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Utility function for device setup
def check_set_gpu(override=None):
    if override is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

# Custom Dataset Class
class MakananIndo(Dataset):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, data_dir, img_size, transform=None, split='train'):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform

        all_image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))])
        
        csv_path = os.path.join(os.path.dirname(data_dir), 'train.csv')
        df = pd.read_csv(csv_path)
        label_dict = dict(zip(df['filename'], df['label']))
        
        all_data = [(f, label_dict.get(f)) for f in all_image_files if label_dict.get(f)]
        
        indices = list(range(len(all_data)))
        random.shuffle(indices)
        train_len = int(0.8 * len(all_data))
        
        if split == 'train':
            self.data = [all_data[i] for i in indices[:train_len]]
        elif split == 'val':
            self.data = [all_data[i] for i in indices[train_len:]]
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        self.default_transform = Compose([
            ToTensor(),
            Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = self.default_transform(image)
            
        return image, label, img_path

# Helper function for label encoding
def create_label_encoder(dataset):
    all_labels = [label for _, label, _ in dataset]
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    return label_to_idx, idx_to_label, unique_labels

# Training and validation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device, label_to_idx):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels_tuple, _ in pbar:
        inputs = inputs.to(device)
        label_indices = [label_to_idx[label] for label in labels_tuple]
        targets = torch.tensor(label_indices, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'acc': 100.*correct/total})
    return total_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device, label_to_idx):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels_tuple, _ in pbar:
            inputs = inputs.to(device)
            label_indices = [label_to_idx[label] for label in labels_tuple]
            targets = torch.tensor(label_indices, dtype=torch.long).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'acc': 100.*correct/total})
    return total_loss/len(dataloader), 100.*correct/total

# Worker init function for DataLoader
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# --- NEW RESNET MODEL DEFINITION ---

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # The residual connection
        out = F.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_resnet34(num_classes=5):
    return ResNet34(num_classes=num_classes)

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    
    device = check_set_gpu()
    
    # Hyperparameters
    num_epochs = 10
    batch_size = 24
    learning_rate = 0.0005
    weight_decay = 0.0005
    img_size = (300, 300)
    data_directory = 'IF25-4041-dataset/train'
    
    train_transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Normalize(mean=MakananIndo.IMAGENET_MEAN, std=MakananIndo.IMAGENET_STD)
    ])
    val_transform = Compose([
        ToTensor(),
        Normalize(mean=MakananIndo.IMAGENET_MEAN, std=MakananIndo.IMAGENET_STD)
    ])
    
    print("Loading datasets...")
    train_dataset = MakananIndo(data_dir=data_directory, img_size=img_size, transform=train_transform, split='train')
    val_dataset = MakananIndo(data_dir=data_directory, img_size=img_size, transform=val_transform, split='val')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print("\nCreating label encoder...")
    label_to_idx, idx_to_label, unique_labels = create_label_encoder(train_dataset)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")
    
    nworkers = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers, 
        pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers, 
        pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True
    )
    
    print("\n" + "="*60)
    print("INITIALIZING RESNET-34 MODEL FOR TRAINING") # <-- Updated print statement
    print("="*60)
    model = create_resnet34(num_classes=num_classes).to(device) # <-- Using the new model
    summary(model, input_size=(batch_size, 3, img_size[0], img_size[1]))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    model_save_path = 'resnet34_best_model.pth' # <-- Updated model name

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, label_to_idx)
        val_loss, val_acc = validate(model, val_loader, criterion, device, label_to_idx)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✅ New best validation accuracy: {best_val_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), model_save_path)
            
    print("\n" + "="*60)
    print(f"🏆 TRAINING FINISHED! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy History'); ax2.legend(); ax2.grid(True)
    plt.show()
    
    print("\nPERFORMING FINAL EVALUATION USING THE BEST MODEL\n")
    best_model = create_resnet34(num_classes=num_classes).to(device) # <-- Use new model
    best_model.load_state_dict(torch.load(model_save_path))
    
    all_labels, all_preds = [], []
    best_model.eval()
    with torch.no_grad():
        for inputs, labels_tuple, _ in tqdm(val_loader, desc="Generating final predictions"):
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend([label_to_idx[l] for l in labels_tuple])
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=unique_labels, digits=4))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.show()