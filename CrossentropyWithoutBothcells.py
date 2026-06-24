import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy
import math
import time
from collections import Counter, deque
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import sklearn
import pandas as pd


train_dir = "data/cell-images/train"
healthy_count = len(os.listdir(os.path.join(train_dir, "healthy")))
rubbish_count = len(os.listdir(os.path.join(train_dir, "rubbish")))
unhealthy_count = len(os.listdir(os.path.join(train_dir, "unhealthy")))

print(f"Healthy: {healthy_count}, Rubbish: {rubbish_count}, Unhealthy: {unhealthy_count}")

class APACCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
        :param image_paths: a list of the image paths.
        :param labels: a list of the image paths, corresponding to the image above, using class_code
        :param transform: PyTorch transforms to apply to the given image.
        """
        self.image_paths = image_paths
        self.labels=labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path=self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class TestDataset(Dataset):

    def __init__(self, folder_path, transform=None):
        self.folder_path=folder_path
        self.transform=transform
        self.image_files = [f for f in os.listdir(folder_path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path= os.path.join(self.folder_path, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image,filename


train_data_transforms = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize( # Standard ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_data_transforms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class_code = {
    0: 'healthy',
    1: 'rubbish',
    2: 'unhealthy',
}

if __name__ == '__main__':

    #TRAIN LOAD
    train_paths = []
    train_labels = []
    eval_paths = []
    eval_labels = []


    folder_path = ["data/cell-images/train/healthy", "data/cell-images/train/rubbish", "data/cell-images/train/unhealthy"]
    category = -1
    for current_path in folder_path:
        category+=1
        for filename in os.listdir(current_path):
            image_path=os.path.join(current_path,filename)
            train_paths.append(image_path)
            train_labels.append(category)

    train_paths, eval_paths, train_labels, eval_labels = train_test_split(train_paths,train_labels, test_size=0.15, stratify=train_labels)

    data = APACCDataset(image_paths=train_paths, labels=train_labels, transform=train_data_transforms)
    data_loader = DataLoader(data, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

    data = APACCDataset(image_paths=eval_paths, labels=eval_labels, transform=test_data_transforms)
    eval_loader = DataLoader(data,batch_size=32, pin_memory=True, num_workers=4)

    counts = Counter(train_labels)
    class_count = [counts[0],counts[1],counts[2]]
    class_weights = [math.sqrt(sum(class_count)/c) for c in class_count]


########################
#     TEST LOADING     #
########################

    #KAGGLE TEST LOAD
    folder_path = "data/cell-images/test/"
    data = TestDataset(folder_path=folder_path, transform=test_data_transforms)
    test_loader = DataLoader(data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    #OTHER TEST LOAD
    folder_path = "data/cell-images/test"
    csv_path = "data/cell-images/isbi2025-ps3c-test-dataset-annotated.csv"
    df = pd.read_csv(csv_path)
    label_mapping = {v: k for k, v in class_code.items()}
    test_paths = []
    test_labels = []

    for idx, row in df.iterrows():
        filename=row['image_name']
        category_text = row['label']
        image_path = os.path.join(folder_path, filename)
        category_init = label_mapping[category_text]
        test_paths.append(image_path)
        test_labels.append(category_init)

    data = APACCDataset(image_paths=test_paths, labels=test_labels, transform=test_data_transforms)
    test_loader2 = DataLoader(data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)



###############
#    MODEL    #
###############

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[4:].parameters():
        param.requires_grad = True

    model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    for param in model.classifier.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    lr_optimizer = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_f1=0.0
    #LOADING IN PREVIOUS BEST MODEL // Áramszünet miatt újrakezdés
    MODEL_WEIGHTS = "best_model_epoch_24.pth"
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))

    best_model_weights=copy.deepcopy(model.state_dict())



#############
#   TRAIN   #
#############
    print("Training started!")
    num_epochs = 30
    start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        full_loss = 0.0
        for images, labels in data_loader:
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            full_loss+=loss.item()
        epoch_loss = full_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")

        # ==========================================
        # EVAL
        # ==========================================

        model.eval()
        all_epoch_probs = []
        all_epoch_labels = []

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                outputs = model(images)

                # Get smooth probabilities
                probs = torch.softmax(outputs, dim=1)

                all_epoch_probs.extend(probs.cpu().numpy())
                all_epoch_labels.extend(labels.cpu().numpy())

        import numpy as np

        best_epoch_thresh = 0.50
        best_epoch_f1 = 0.0
        best_epoch_preds = []

        # Test thresholds to find the best possible F1 for this specific epoch
        for thresh in np.arange(0.30, 0.95, 0.05):
            test_preds = []
            for prob in all_epoch_probs:
                if prob[2] >= thresh:
                    test_preds.append(2)
                else:
                    test_preds.append(np.argmax(prob[:2]))

            simulated_f1 = f1_score(all_epoch_labels, test_preds, average='weighted')

            if simulated_f1 > best_epoch_f1:
                best_epoch_f1 = simulated_f1
                best_epoch_thresh = thresh
                best_epoch_preds = test_preds

        # Epoch stats
        current_f1 = best_epoch_f1
        accuracy = accuracy_score(all_epoch_labels, best_epoch_preds)
        conf_matrix = confusion_matrix(all_epoch_labels, best_epoch_preds)
        rep = classification_report(all_epoch_labels, best_epoch_preds,
                                    target_names=['Healthy', 'Rubbish', 'Unhealthy'])

        print(f"--- Statistics of epoch number {epoch + 1} ---")
        print(f"Optimal Epoch Threshold: {best_epoch_thresh:.2f}")
        print(f"F1-Score: {current_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")
        print("Confusion Matrix\n", conf_matrix)
        print("Classification Report\n", rep)
        print("\n\n")


        if current_f1 > best_f1:
            print(f"New best model found at {epoch + 1}. epoch! (Using Thresh {best_epoch_thresh:.2f})")
            print(f"Accuracy:  {accuracy:.4f} ({(accuracy * 100):.2f}%)")
            print(f"F1-Score:  {current_f1:.4f}")
            best_f1 = current_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            file_path = f"best_model_v2_epoch_{epoch + 1}.pth"
            torch.save(best_model_weights, file_path)

            GLOBAL_BEST_THRESH = best_epoch_thresh

        epoch_end = time.time()
        print(f"Epoch runtime: {(epoch_end - epoch_start) / 3600:.2f} hours")
        print(f"Epoch runtime: {(epoch_end - epoch_start) / 60:.2f} mins")

        lr_optimizer.step()
    end = time.time()

    print(f"Runtime:{(end-start)/3600} hours")
    print(f"Runtime:{(end-start)//60} mins")

    model.load_state_dict(best_model_weights)

    model.eval()
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Move to CPU so we can use numpy for the hunt
            all_val_probs.extend(probs.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    import numpy as np

    best_thresh = 0.50
    best_f1_hunt = 0.0

    for thresh in np.arange(0.30, 0.95, 0.05):
        test_preds = []
        for prob in all_val_probs:
            if prob[2] >= thresh:
                test_preds.append(2)
            else:
                #Fallback class (rubbish or healthy), if unhealthy doesnt go above the threshold
                test_preds.append(np.argmax(prob[:2]))

        simulated_f1 = f1_score(all_val_labels, test_preds, average='weighted')
        print(f"Threshold: {thresh:.2f} | Simulated F1: {simulated_f1:.4f}")

        if simulated_f1 > best_f1_hunt:
            best_f1_hunt = simulated_f1
            best_thresh = thresh

    print(f"\nWINNER: Using Threshold {best_thresh:.2f} for Final Testing!")

    # ==========================================
    # EVALUATION (Using the Test CSV)
    # ==========================================
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader2:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # USE THE WINNING THRESHOLD HERE
            probs = torch.softmax(outputs, dim=1)
            fallback_classes = torch.argmax(probs[:, :2], dim=1)
            unhealthy_mask = probs[:, 2] >= best_thresh
            predicted_classes = torch.where(unhealthy_mask, torch.tensor(2).to(device), fallback_classes)

            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

    print("\n--- Final Test Results ---")
    print(f"Accuracy:  {accuracy:.4f} ({(accuracy * 100):.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # ==========================================
    # KAGGLE SUBMISSION (Using the Blind Test Folder)
    # ==========================================
    print("\n--- Generating Kaggle Submission ---")
    all_predictions = []
    all_file_names = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)

            # USE THE WINNING THRESHOLD HERE TOO
            probs = torch.softmax(outputs, dim=1)
            fallback_classes = torch.argmax(probs[:, :2], dim=1)
            unhealthy_mask = probs[:, 2] >= best_thresh
            predicted_classes = torch.where(unhealthy_mask, torch.tensor(2).to(device), fallback_classes)

            for i in range(len(filenames)):
                fname = filenames[i]
                pred_int = predicted_classes[i].item()
                pred_text = class_code[pred_int]

                all_file_names.append(fname)
                all_predictions.append(pred_text)

    print(f"Total files matched: {len(all_file_names)}")
    print(f"Total labels translated: {len(all_predictions)}")

    submission_df = pd.DataFrame({
        'image_name': all_file_names,
        'label': all_predictions
    })

    submission_df.to_csv("submission.csv", index=False)
    print("CSV successfully translated, formatted, and saved!")