import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ==========================================
# 1. Define the Custom Dataset Worker
# (MUST be at the top level for Windows Multiprocessing)
# ==========================================
class CSVTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'healthy': 0, 'rubbish': 1, 'unhealthy': 2}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]['image_name']
        raw_label = self.data_frame.iloc[idx]['label']

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if isinstance(raw_label, str):
            label = self.label_map[raw_label]
        else:
            label = raw_label

        return image, label, img_path


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
def main():
    # ==========================================
    # 2. Set Up the Environment
    # ==========================================
    CSV_PATH = "data/cell-images/isbi2025-ps3c-test-dataset-annotated.csv"
    IMAGE_DIR = "data/cell-images/test/"
    MODEL_WEIGHTS = "best_model_v2_epoch_24.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reverse map to turn the AI's numbers back into Kaggle words
    reverse_label_map = {0: 'healthy', 1: 'rubbish', 2: 'unhealthy'}

    # ==========================================
    # 3. Load the Model
    # ==========================================
    print("Loading model architecture and weights...")
    model = models.efficientnet_b3(weights=None)
    model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval()

    # ==========================================
    # 4. Prepare the DataLoader (NOW WITH WORKERS!)
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CSVTestDataset(csv_file=CSV_PATH, img_dir=IMAGE_DIR, transform=transform)

    # ADDED: num_workers=4 and pin_memory=True (Speeds up data transfer to the GPU)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ==========================================
    # 5. Run the Evaluation Loop
    # ==========================================
    all_preds = []
    all_true = []

    # New lists specifically for the CSV
    all_file_names = []
    all_pred_strings = []

    print(f"\nStarting evaluation on {len(test_dataset)} images...")

    ###### CURRENT BEST SETUP DATA ######
    # Model: best_model_v2_epoch_27
    BEST_THRESH = 0.45
    print(f"Applying Unhealthy Confidence Threshold: {BEST_THRESH}")

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # --- THE THRESHOLD LOGIC ---
            probs = torch.softmax(outputs, dim=1)
            fallback_classes = torch.argmax(probs[:, :2], dim=1)
            unhealthy_mask = probs[:, 2] >= BEST_THRESH
            predicted = torch.where(unhealthy_mask, torch.tensor(2).to(device), fallback_classes)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

            # --- COLLECT DATA FOR CSV ---
            for i in range(len(paths)):
                fname = os.path.basename(paths[i])
                pred_int = predicted[i].item()
                pred_text = reverse_label_map[pred_int]

                all_file_names.append(fname)
                all_pred_strings.append(pred_text)

    # ==========================================
    # 6. Print the Final Report
    # ==========================================
    print("\n--- Final Test Results ---")

    target_names = ['Healthy', 'Rubbish', 'Unhealthy']
    print(classification_report(all_true, all_preds, target_names=target_names))

    f1 = f1_score(all_true, all_preds, average="weighted")
    print(f"F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    # ==========================================
    # 7. Generate Kaggle Submission CSV
    # ==========================================
    print("\n--- Generating Kaggle Submission ---")

    submission_df = pd.DataFrame({
        'image_name': all_file_names,
        'label': all_pred_strings
    })

    submission_df.to_csv("submission.csv", index=False)

    print(f"Total files matched: {len(all_file_names)}")
    print(f"Total labels translated: {len(all_pred_strings)}")
    print("CSV successfully translated, formatted, and saved!")


# ==========================================
# WINDOWS MULTIPROCESSING SAFETY LOCK
# ==========================================
if __name__ == '__main__':
    main()