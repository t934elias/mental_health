import ast
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]

# Paths
SCALER_PATH = "./models/policy_scaler_llm_2.pkl"
MODEL_PATH = "./models/policy_model_llm_2.pt"
M2_MODEL_PATH = "./models/policy_without_uncertainty.pt"
M1_MODEL_PATH = "./models/policy_with_uncertainty.pt"
M3_MODEL_PATH = "./models/policy_without_risk.pt"
M4_MODEL_PATH = "./models/policy_without_emotion.pt"
CSV_PATH = "./data/action_label_dataset_llm_combined.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# DATASET CLASS
# ----------------------------
class MentalHealthDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = torch.tensor(df["action_id"].values, dtype=torch.long)
        emotions, risks, risks_vec, uncerts, embeddings = [], [], [], [], []

        for idx, raw_state in enumerate(df["state_vector"]):
            state_dict = raw_state
            while isinstance(state_dict, str):
                state_dict = re.sub(r"np\.\w+\((.*?)\)", r"\1", state_dict)
                state_dict = re.sub(r"(?:array|tensor)\((.*?)\)", r"\1", state_dict)
                state_dict = state_dict.replace("\n", " ").replace("\r", " ")
                try:
                    state_dict = ast.literal_eval(state_dict)
                except:
                    try:
                        state_dict = json.loads(state_dict.replace("'", '"'))
                    except:
                        state_dict = {}

            emotion_vec = [float(state_dict.get("emotion", {}).get(k, 0.0)) for k in EMOTION_ORDER]
            risk = float(state_dict.get("risk", 0.0))
            risk_vector = [float(x) for x in state_dict.get("risk_vector", [0.0] * 9)]
            uncert = float(state_dict.get("risk_uncertainty", 0.0))
            emb_key = "embedding" if "embedding" in state_dict else "embeddings"
            embedding = [float(x) for x in state_dict.get(emb_key, [0.0] * 384)]

            if len(embedding) < 384:
                embedding += [0.0] * (384 - len(embedding))
            else:
                embedding = embedding[:384]

            emotions.append(emotion_vec)
            risks.append([risk])
            risks_vec.append(risk_vector)
            uncerts.append([uncert])
            embeddings.append(embedding)

        self.emotions = torch.tensor(emotions, dtype=torch.float32)
        self.risks = torch.tensor(risks, dtype=torch.float32)
        self.risks_vec = torch.tensor(risks_vec, dtype=torch.float32)
        self.uncerts = torch.tensor(uncerts, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emotions[idx],
            self.risks[idx],
            self.risks_vec[idx],
            self.uncerts[idx],
            self.embeddings[idx],
            self.labels[idx],
        )


# ----------------------------
# MODEL ARCHITECTURES
# ----------------------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotion_net = nn.Linear(6, 16)
        self.risk_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.risk_vec_net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64))
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)
        self.classifier = nn.Sequential(
            nn.Linear(16 + 32 + 64 + 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        e = torch.relu(self.emotion_net(emotion))
        r = torch.relu(self.risk_net(risk))
        rv = torch.relu(self.risk_vec_net(risk_vec))
        u = torch.relu(self.uncert_net(uncert))
        emb = torch.relu(self.embed_net(embed))
        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)


class PolicyNoUncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotion_net = nn.Linear(6, 16)
        self.risk_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.risk_vec_net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64))
        self.embed_net = nn.Linear(384, 128)
        self.classifier = nn.Sequential(
            nn.Linear(16 + 32 + 64 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        e = torch.relu(self.emotion_net(emotion))
        r = torch.relu(self.risk_net(risk))
        rv = torch.relu(self.risk_vec_net(risk_vec))
        emb = torch.relu(self.embed_net(embed))
        x = torch.cat([e, r, rv, emb], dim=1)
        return self.classifier(x)


class PolicyNoRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotion_net = nn.Linear(6, 16)
        # self.risk_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        # self.risk_vec_net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64))
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)
        self.classifier = nn.Sequential(
            nn.Linear(16 + 4 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        e = torch.relu(self.emotion_net(emotion))
        # r = torch.relu(self.risk_net(risk))
        # rv = torch.relu(self.risk_vec_net(risk_vec))
        u = torch.relu(self.uncert_net(uncert))
        emb = torch.relu(self.embed_net(embed))
        x = torch.cat([e, u, emb], dim=1)
        return self.classifier(x)





class PolicyNoEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        # self.emotion_net = nn.Linear(6, 16)
        self.risk_net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.risk_vec_net = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64))
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)
        self.classifier = nn.Sequential(
            nn.Linear(4 + 32 + 64 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        # e = torch.relu(self.emotion_net(emotion))
        r = torch.relu(self.risk_net(risk))
        rv = torch.relu(self.risk_vec_net(risk_vec))
        u = torch.relu(self.uncert_net(uncert))
        emb = torch.relu(self.embed_net(embed))
        x = torch.cat([r, rv, u, emb], dim=1)
        return self.classifier(x)
# =====================================================================
# PART 1: EVALUATE MODEL 1 (ALREADY TRAINED)
# =====================================================================
print("\n=== RUNNING EVALUATION FOR MODEL 1 (PRE-TRAINED) ===")

# Load existing scaler saved during Model 1's training
with open(SCALER_PATH, "rb") as f:
    scaler_m1 = pickle.load(f)

dataset_m1 = MentalHealthDataset(CSV_PATH)
labels_m1 = dataset_m1.labels.numpy()
indices_m1 = np.arange(len(dataset_m1))

# Stratified split to extract the same test indices used originally
train_idx, temp_idx = train_test_split(
    indices_m1, test_size=0.2, stratify=labels_m1, random_state=42
)
temp_labels = labels_m1[temp_idx]
val_idx, test_idx_m1 = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)

# Apply saved scaler to dataset features
all_features_m1 = torch.cat(
    [
        dataset_m1.emotions,
        dataset_m1.risks,
        dataset_m1.risks_vec,
        dataset_m1.uncerts,
        dataset_m1.embeddings,
    ],
    dim=1,
).numpy()

scaled_features_m1 = scaler_m1.transform(all_features_m1)

e_dim = dataset_m1.emotions.shape[1]
r_dim = dataset_m1.risks.shape[1]
rv_dim = dataset_m1.risks_vec.shape[1]
u_dim = dataset_m1.uncerts.shape[1]

dataset_m1.emotions = torch.tensor(scaled_features_m1[:, :e_dim], dtype=torch.float32)
dataset_m1.risks = torch.tensor(scaled_features_m1[:, e_dim : e_dim + r_dim], dtype=torch.float32)
dataset_m1.risks_vec = torch.tensor(scaled_features_m1[:, e_dim + r_dim : e_dim + r_dim + rv_dim], dtype=torch.float32)
dataset_m1.uncerts = torch.tensor(scaled_features_m1[:, e_dim + r_dim + rv_dim : e_dim + r_dim + rv_dim + u_dim], dtype=torch.float32)
dataset_m1.embeddings = torch.tensor(scaled_features_m1[:, e_dim + r_dim + rv_dim + u_dim :], dtype=torch.float32)

train_ds_m1 = Subset(dataset_m1, train_idx)
val_ds_m1 = Subset(dataset_m1, val_idx)
test_ds_m1 = Subset(dataset_m1, test_idx_m1)

train_loader_m1 = DataLoader(train_ds_m1, batch_size=32, shuffle=True)
val_loader_m1 = DataLoader(val_ds_m1, batch_size=32, shuffle=False)
test_loader_m1 = DataLoader(test_ds_m1, batch_size=32, shuffle=False)

# Load weights and evaluate
model_m1 = Policy().to(device)
# model_m1.load_state_dict(torch.load(MODEL_PATH, map_location=device))
df = pd.read_csv(CSV_PATH)
class_counts = df["action_id"].value_counts().sort_index()
weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32, device=device)
weights = weights / weights.sum() * len(weights)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model_m1.parameters(), lr=3e-4)

# Training Loop
EPOCHS = 40
for epoch in range(EPOCHS):
    model_m1.train()
    correct, total_samples = 0, 0
    epoch_loss = 0

    for e, r, rv, u, emb, y in train_loader_m1:
        e, r, rv, u, emb, y = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device), y.to(device)

        logits = model_m1(e, r, rv, u, emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_samples += y.size(0)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(train_loader_m1):.4f} | Train Acc: {correct/total_samples:.4f}")

# Save the newly trained weights for Model 2
torch.save(model_m1.state_dict(), M1_MODEL_PATH)
print(f"Model 1 weights saved to {M1_MODEL_PATH}")
model_m1.eval()

preds_m1 = []
labels_m1_list = []

with torch.no_grad():
    for e, r, rv, u, emb, y in test_loader_m1:
        e, r, rv, u, emb = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device)
        logits = model_m1(e, r, rv, u, emb)
        pred = logits.argmax(1)
        preds_m1.extend(pred.cpu().numpy())
        labels_m1_list.extend(y.numpy())

print("\nModel 1 (With Uncertainty) Test Results:")
print(classification_report(
    labels_m1_list, preds_m1,
    target_names=["empathy", "reflection", "clarifying", "coping", "safety", "professional", "abstain"]
))
precision_m1, recall_m1, f1_m1, _ = precision_recall_fscore_support(
    labels_m1_list, preds_m1, average="macro", zero_division=0
)
print(f"Accuracy : {accuracy_score(labels_m1_list, preds_m1):.4f}")
print(f"Precision: {precision_m1:.4f}")
print(f"Recall   : {recall_m1:.4f}")
print(f"Macro F1 : {f1_m1:.4f}")


# =====================================================================
# PART 2: TRAIN & EVALUATE MODEL 2 (NO UNCERTAINTY)
# =====================================================================
print("\n=== TRAINING MODEL 2 (WITHOUT UNCERTAINTY) FROM SCRATCH ===")

# Create a clean data instance for training Model 2
dataset_m2 = MentalHealthDataset(CSV_PATH)
labels_m2 = dataset_m2.labels.numpy()
indices_m2 = np.arange(len(dataset_m2))

# Train / Val / Test Split
train_idx, temp_idx = train_test_split(
    indices_m2, test_size=0.2, stratify=labels_m2, random_state=42
)
temp_labels = labels_m2[temp_idx]
val_idx, test_idx_m2 = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)

# Fit scaler *only* on Model 2's training set indices to prevent leaks
all_features_m2 = torch.cat(
    [
        dataset_m2.emotions,
        dataset_m2.risks,
        dataset_m2.risks_vec,
        dataset_m2.uncerts,
        dataset_m2.embeddings,
    ],
    dim=1,
).numpy()

scaler_m2 = StandardScaler()
scaler_m2.fit(all_features_m2[train_idx])
scaled_features_m2 = scaler_m2.transform(all_features_m2)

dataset_m2.emotions = torch.tensor(scaled_features_m2[:, :e_dim], dtype=torch.float32)
dataset_m2.risks = torch.tensor(scaled_features_m2[:, e_dim : e_dim + r_dim], dtype=torch.float32)
dataset_m2.risks_vec = torch.tensor(scaled_features_m2[:, e_dim + r_dim : e_dim + r_dim + rv_dim], dtype=torch.float32)
dataset_m2.uncerts = torch.tensor(scaled_features_m2[:, e_dim + r_dim + rv_dim : e_dim + r_dim + rv_dim + u_dim], dtype=torch.float32)
dataset_m2.embeddings = torch.tensor(scaled_features_m2[:, e_dim + r_dim + rv_dim + u_dim :], dtype=torch.float32)

# Save Model 2's scaler (Optional: change path if you don't want to overwrite Model 1's)
with open("./models/policy_scaler_without_uncertainty.pkl", "wb") as f:
    pickle.dump(scaler_m2, f)

train_ds_m2 = Subset(dataset_m2, train_idx)
val_ds_m2 = Subset(dataset_m2, val_idx)
test_ds_m2 = Subset(dataset_m2, test_idx_m2)

train_loader_m2 = DataLoader(train_ds_m2, batch_size=32, shuffle=True)
val_loader_m2 = DataLoader(val_ds_m2, batch_size=32, shuffle=False)
test_loader_m2 = DataLoader(test_ds_m2, batch_size=32, shuffle=False)

# Model setup
model_m2 = PolicyNoUncertainty().to(device)

# Handle class imbalance weights
df = pd.read_csv(CSV_PATH)
class_counts = df["action_id"].value_counts().sort_index()
weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32, device=device)
weights = weights / weights.sum() * len(weights)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model_m2.parameters(), lr=3e-4)

# Training Loop
EPOCHS = 40
for epoch in range(EPOCHS):
    model_m2.train()
    correct, total_samples = 0, 0
    epoch_loss = 0

    for e, r, rv, u, emb, y in train_loader_m2:
        e, r, rv, u, emb, y = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device), y.to(device)

        logits = model_m2(e, r, rv, u, emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_samples += y.size(0)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(train_loader_m2):.4f} | Train Acc: {correct/total_samples:.4f}")

# Save the newly trained weights for Model 2
torch.save(model_m2.state_dict(), M2_MODEL_PATH)
print(f"Model 2 weights saved to {M2_MODEL_PATH}")

# Evaluation
model_m2.eval()
preds_m2 = []
labels_m2_list = []

with torch.no_grad():
    for e, r, rv, u, emb, y in test_loader_m2:
        e, r, rv, u, emb = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device)
        logits = model_m2(e, r, rv, u, emb)
        pred = logits.argmax(1)
        preds_m2.extend(pred.cpu().numpy())
        labels_m2_list.extend(y.numpy())

print("\nModel 2 (Without Uncertainty) Test Results:")
print(classification_report(
    labels_m2_list, preds_m2,
    target_names=["empathy", "reflection", "clarifying", "coping", "safety", "professional", "abstain"]
))

acc_m2 = accuracy_score(labels_m2_list, preds_m2)
precision_m2, recall_m2, f1_m2, _ = precision_recall_fscore_support(
    labels_m2_list, preds_m2, average="macro", zero_division=0
)
print(f"Accuracy : {acc_m2:.4f}")
print(f"Precision: {precision_m2:.4f}")
print(f"Recall   : {recall_m2:.4f}")
print(f"Macro F1 : {f1_m2:.4f}")





# =====================================================================
# PART 2: TRAIN & EVALUATE MODEL 3 (NO RISK)
# =====================================================================
print("\n=== TRAINING MODEL 3 (WITHOUT RISK) FROM SCRATCH ===")

# Create a clean data instance for training Model 2
dataset_m3 = MentalHealthDataset(CSV_PATH)
labels_m3 = dataset_m3.labels.numpy()
indices_m3 = np.arange(len(dataset_m3))

# Train / Val / Test Split
train_idx, temp_idx = train_test_split(
    indices_m3, test_size=0.2, stratify=labels_m3, random_state=42
)
temp_labels = labels_m3[temp_idx]
val_idx, test_idx_m3 = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)

# Fit scaler *only* on Model 2's training set indices to prevent leaks
all_features_m3 = torch.cat(
    [
        dataset_m3.emotions,
        dataset_m3.risks,
        dataset_m3.risks_vec,
        dataset_m3.uncerts,
        dataset_m3.embeddings,
    ],
    dim=1,
).numpy()

scaler_m3 = StandardScaler()
scaler_m3.fit(all_features_m3[train_idx])
scaled_features_m3 = scaler_m3.transform(all_features_m3)

dataset_m3.emotions = torch.tensor(scaled_features_m3[:, :e_dim], dtype=torch.float32)
dataset_m3.risks = torch.tensor(scaled_features_m3[:, e_dim : e_dim + r_dim], dtype=torch.float32)
dataset_m3.risks_vec = torch.tensor(scaled_features_m3[:, e_dim + r_dim : e_dim + r_dim + rv_dim], dtype=torch.float32)
dataset_m3.uncerts = torch.tensor(scaled_features_m3[:, e_dim + r_dim + rv_dim : e_dim + r_dim + rv_dim + u_dim], dtype=torch.float32)
dataset_m3.embeddings = torch.tensor(scaled_features_m3[:, e_dim + r_dim + rv_dim + u_dim :], dtype=torch.float32)

# Save Model 2's scaler (Optional: change path if you don't want to overwrite Model 1's)
with open("./models/policy_scaler_without_risk.pkl", "wb") as f:
    pickle.dump(scaler_m3, f)

train_ds_m3 = Subset(dataset_m3, train_idx)
val_ds_m3 = Subset(dataset_m3, val_idx)
test_ds_m3 = Subset(dataset_m3, test_idx_m3)

train_loader_m3 = DataLoader(train_ds_m3, batch_size=32, shuffle=True)
val_loader_m3 = DataLoader(val_ds_m3, batch_size=32, shuffle=False)
test_loader_m3 = DataLoader(test_ds_m3, batch_size=32, shuffle=False)

# Model setup
model_m3 = PolicyNoRisk().to(device)

# Handle class imbalance weights
df = pd.read_csv(CSV_PATH)
class_counts = df["action_id"].value_counts().sort_index()
weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32, device=device)
weights = weights / weights.sum() * len(weights)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model_m3.parameters(), lr=3e-4)

# Training Loop
EPOCHS = 40
for epoch in range(EPOCHS):
    model_m3.train()
    correct, total_samples = 0, 0
    epoch_loss = 0

    for e, r, rv, u, emb, y in train_loader_m3:
        e, r, rv, u, emb, y = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device), y.to(device)

        logits = model_m3(e, r, rv, u, emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_samples += y.size(0)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(train_loader_m3):.4f} | Train Acc: {correct/total_samples:.4f}")

# Save the newly trained weights for Model 2
torch.save(model_m3.state_dict(), M3_MODEL_PATH)
print(f"Model 3 weights saved to {M3_MODEL_PATH}")

# Evaluation
model_m3.eval()
preds_m3 = []
labels_m3_list = []

with torch.no_grad():
    for e, r, rv, u, emb, y in test_loader_m3:
        e, r, rv, u, emb = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device)
        logits = model_m3(e, r, rv, u, emb)
        pred = logits.argmax(1)
        preds_m3.extend(pred.cpu().numpy())
        labels_m3_list.extend(y.numpy())

print("\nModel 3 (Without RISK) Test Results:")
print(classification_report(
    labels_m3_list, preds_m3,
    target_names=["empathy", "reflection", "clarifying", "coping", "safety", "professional", "abstain"]
))

acc_m3 = accuracy_score(labels_m3_list, preds_m3)
precision_m3, recall_m3, f1_m3, _ = precision_recall_fscore_support(
    labels_m3_list, preds_m3, average="macro", zero_division=0
)
print(f"Accuracy : {acc_m3:.4f}")
print(f"Precision: {precision_m3:.4f}")
print(f"Recall   : {recall_m3:.4f}")
print(f"Macro F1 : {f1_m3:.4f}")



# =====================================================================
# PART 2: TRAIN & EVALUATE MODEL 34 (NO EMOTION)
# =====================================================================
print("\n=== TRAINING MODEL 4 (WITHOUT EMOTION) FROM SCRATCH ===")

# Create a clean data instance for training Model 2
dataset_m4 = MentalHealthDataset(CSV_PATH)
labels_m4 = dataset_m4.labels.numpy()
indices_m4 = np.arange(len(dataset_m4))

# Train / Val / Test Split
train_idx, temp_idx = train_test_split(
    indices_m4, test_size=0.2, stratify=labels_m4, random_state=42
)
temp_labels = labels_m4[temp_idx]
val_idx, test_idx_m4 = train_test_split(
    temp_idx, test_size=0.5, stratify=temp_labels, random_state=42
)

# Fit scaler *only* on Model 2's training set indices to prevent leaks
all_features_m4 = torch.cat(
    [
        dataset_m4.emotions,
        dataset_m4.risks,
        dataset_m4.risks_vec,
        dataset_m4.uncerts,
        dataset_m4.embeddings,
    ],
    dim=1,
).numpy()

scaler_m4 = StandardScaler()
scaler_m4.fit(all_features_m4[train_idx])
scaled_features_m4 = scaler_m4.transform(all_features_m4)

dataset_m4.emotions = torch.tensor(scaled_features_m4[:, :e_dim], dtype=torch.float32)
dataset_m4.risks = torch.tensor(scaled_features_m4[:, e_dim : e_dim + r_dim], dtype=torch.float32)
dataset_m4.risks_vec = torch.tensor(scaled_features_m4[:, e_dim + r_dim : e_dim + r_dim + rv_dim], dtype=torch.float32)
dataset_m4.uncerts = torch.tensor(scaled_features_m4[:, e_dim + r_dim + rv_dim : e_dim + r_dim + rv_dim + u_dim], dtype=torch.float32)
dataset_m4.embeddings = torch.tensor(scaled_features_m4[:, e_dim + r_dim + rv_dim + u_dim :], dtype=torch.float32)

# Save Model 2's scaler (Optional: change path if you don't want to overwrite Model 1's)
with open("./models/policy_scaler_without_emotion.pkl", "wb") as f:
    pickle.dump(scaler_m4, f)

train_ds_m4 = Subset(dataset_m4, train_idx)
val_ds_m4 = Subset(dataset_m4, val_idx)
test_ds_m4 = Subset(dataset_m4, test_idx_m4)

train_loader_m4 = DataLoader(train_ds_m4, batch_size=32, shuffle=True)
val_loader_m4 = DataLoader(val_ds_m4, batch_size=32, shuffle=False)
test_loader_m4 = DataLoader(test_ds_m4, batch_size=32, shuffle=False)

# Model setup
model_m4 = PolicyNoEmotion().to(device)

# Handle class imbalance weights
df = pd.read_csv(CSV_PATH)
class_counts = df["action_id"].value_counts().sort_index()
weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32, device=device)
weights = weights / weights.sum() * len(weights)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model_m4.parameters(), lr=3e-4)

# Training Loop
EPOCHS = 40
for epoch in range(EPOCHS):
    model_m4.train()
    correct, total_samples = 0, 0
    epoch_loss = 0

    for e, r, rv, u, emb, y in train_loader_m4:
        e, r, rv, u, emb, y = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device), y.to(device)

        logits = model_m4(e, r, rv, u, emb)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total_samples += y.size(0)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(train_loader_m4):.4f} | Train Acc: {correct/total_samples:.4f}")

# Save the newly trained weights for Model 4
torch.save(model_m4.state_dict(), M4_MODEL_PATH)
print(f"Model 4 weights saved to {M4_MODEL_PATH}")

# Evaluation
model_m4.eval()
preds_m4 = []
labels_m4_list = []

with torch.no_grad():
    for e, r, rv, u, emb, y in test_loader_m4:
        e, r, rv, u, emb = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device)
        logits = model_m4(e, r, rv, u, emb)
        pred = logits.argmax(1)
        preds_m4.extend(pred.cpu().numpy())
        labels_m4_list.extend(y.numpy())

print("\nModel 4 (Without emotions) Test Results:")
print(classification_report(
    labels_m4_list, preds_m4,
    target_names=["empathy", "reflection", "clarifying", "coping", "safety", "professional", "abstain"]
))

acc_m4 = accuracy_score(labels_m4_list, preds_m4)
precision_m4, recall_m4, f1_m4, _ = precision_recall_fscore_support(
    labels_m4_list, preds_m4, average="macro", zero_division=0
)
print(f"Accuracy : {acc_m4:.4f}")
print(f"Precision: {precision_m4:.4f}")
print(f"Recall   : {recall_m4:.4f}")
print(f"Macro F1 : {f1_m4:.4f}")