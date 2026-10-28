# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from transformers import pipeline
# import numpy as np
# import pandas as pd
# import json
# import ast
# import re
# from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler
# import pickle

# CSV_PATH = "./data/action_label_dataset.csv"

# EMOTION_ORDER = [
#     "sadness",
#     "fear",
#     "anger",
#     "confusion",
#     "joy",
#     "neutral"
# ]

# # df = pd.read_csv(CSV_PATH)

# # states = [json.loads(s) for s in df['state_vector']]
# # actions = df['action_id'].values


# class MentalHealthDataset(Dataset):
#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)
#         self.texts = df['text'].values
#         self.actions = torch.tensor(df['action_id'].values, dtype=torch.long)
       
#         self.states = []
#         for idx, raw_state in enumerate(df['state_vector']):
#             state_dict = raw_state
            
#             while isinstance(state_dict, str):
#                 state_dict = re.sub(r'np\.\w+\((.*?)\)', r'\1', state_dict)
                
#                 state_dict = re.sub(r'(?:array|tensor)\((.*?)\)', r'\1', state_dict)
                
#                 state_dict = state_dict.replace('\n', ' ').replace('\r', ' ')
                
#                 try:
#                     state_dict = ast.literal_eval(state_dict)
#                 except (ValueError, SyntaxError):
#                     json_ready_string = state_dict.replace("'", '"')
#                     json_ready_string = re.sub(r'\b(nan|NaN)\b', 'null', json_ready_string)
                    
#                     try:
#                         state_dict = json.loads(json_ready_string)
#                     except json.JSONDecodeError:
#                         print(f"CRITICAL: Row {idx} failed to parse completely. Using fallback empty state.")
#                         state_dict = {"emotion": {}, "risk": 0.0, "embedding": [0.0] * 384}

#             flat_vector = []
            
#             if 'emotion' in state_dict and isinstance(state_dict['emotion'], dict):
#                 flat_vector.extend([float(v) if v is not None else 0.0 for v in state_dict['emotion'].values()])
#                 # flat_vector.extend([
#                 #     float(state_dict["emotion"].get(k, 0.0))
#                 #     for k in EMOTION_ORDER
#                 # ])
#             else:
#                 flat_vector.extend([0.0] * 6)
            
#             if 'risk' in state_dict:
#                 if isinstance(state_dict['risk'], dict):
#                     flat_vector.extend([float(v) if v is not None else 0.0 for v in state_dict['risk'].values()])
#                 else:
#                     v = state_dict['risk']
#                     flat_vector.append(float(v) if v is not None else 0.0)
#             else:
#                 flat_vector.append(0.0)
                    
#             if 'risk_uncertainty' in state_dict:
#                 flat_vector.append(float(state_dict['risk_uncertainty']))
#             else:
#                 flat_vector.append(0.0)

#             emb_key = 'embedding' if 'embedding' in state_dict else 'embeddings'
#             if emb_key in state_dict and state_dict[emb_key] is not None:
#                 flat_vector.extend([float(v) if v is not None else 0.0 for v in state_dict[emb_key]])
                
#             if len(flat_vector) < 414:
#                 flat_vector += [0.0] * (414 - len(flat_vector))
#             elif len(flat_vector) > 414:
#                 flat_vector = flat_vector[:414]
                
#             self.states.append(flat_vector)
            
#         self.states = torch.tensor(self.states, dtype=torch.float32)

#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, idx):
#         return self.states[idx], self.actions[idx]


# class Policy(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # separate encoders
#         self.emotion_net = nn.Linear(6, 16)
#         self.risk_net = nn.Linear(1, 8)        # or 9 if you use full vector
#         self.embed_net = nn.Linear(384, 128)
#         self.uncert_net = nn.Linear(1, 4)

#         self.classifier = nn.Sequential(
#             nn.Linear(16 + 8 + 128 + 4, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, 7)
#         )

#     def forward(self, x_emotion, x_risk, x_embed, x_uncert):
#         e = torch.relu(self.emotion_net(x_emotion))
#         r = torch.relu(self.risk_net(x_risk))
#         emb = torch.relu(self.embed_net(x_embed))
#         u = torch.relu(self.uncert_net(x_uncert))

#         x = torch.cat([e, r, emb, u], dim=1)
#         return self.classifier(x)


# print("--- Training Supervised Policy Model ---")

# dataset = MentalHealthDataset(CSV_PATH)

# scaler = StandardScaler()
# dataset.states = torch.tensor(
#     scaler.fit_transform(dataset.states.numpy()),
#     dtype=torch.float32
# )

# with open("./models/policy_scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Policy().to(device)

# class_counts = [1507, 4036, 216, 331, 114, 2924, 1186]
# total_samples = sum(class_counts)

# raw_weights = [total_samples / (7.0 * count) for count in class_counts]

# # 3. Apply an aggressive multiplier to critical rare actions (Safety Check & Clarifying Questions)
# raw_weights[4] *= 4.0  # Massive penalty if it ignores a safety check crisis!
# raw_weights[2] *= 2.0  # Boost clarifying questions
# raw_weights[3] *= 2.0  # Boost coping strategies

# class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
# print(f"Applied Class Weights: {np.round(raw_weights, 2)}")

# model = Policy().to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 30
# model.train()

# for epoch in range(epochs):
#     correct = 0
#     total = 0
#     epoch_loss = 0.0
#     for states, actions in dataloader:
#         states, actions = states.to(device), actions.to(device)
        
#         outputs = model(states)
#         loss = criterion(outputs, actions)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         epoch_loss += loss.item()

#     preds = outputs.argmax(dim=1)
#     correct += (preds == actions).sum().item()
#     total += actions.size(0)
#     acc = correct / total
        
#     print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(dataloader):.4f} Acc={acc:.4f}")

# print("\nSupervised policy training complete!")

# from sklearn.metrics import confusion_matrix

# model.eval()

# y_true = []
# y_pred = []

# with torch.no_grad():
#     for states, actions in dataloader:
#         states = states.to(device)

#         outputs = model(states)

#         preds = outputs.argmax(dim=1)

#         y_true.extend(actions.numpy())
#         y_pred.extend(preds.cpu().numpy())
        
# cm = confusion_matrix(y_true, y_pred)
# print(cm)

# MODEL_SAVE_PATH = "./models/policy_model.pt"

# import os
# os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# torch.save(model.state_dict(), MODEL_SAVE_PATH)
# print(f"Model successfully saved to {MODEL_SAVE_PATH}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import ast
import json
import re
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import random
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.metrics import classification_report

CSV_PATH = "./data/action_label_dataset_llm_combined.csv"
SCALER_PATH = "./models/policy_scaler_llm_2.pkl"
MODEL_PATH = "./models/policy_model_llm_2.pt"



import pandas as pd

def combine_two_csvs(file1_path, file2_path, output_path):
    # Safe function to read a file with encoding fallbacks
    def safe_read_csv(filepath):
        try:
            return pd.read_csv(filepath, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(filepath, encoding="cp1252")
            except UnicodeDecodeError:
                # Ultimate safety net: replaces bad characters instead of crashing
                return pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")

    print(f"Loading {file1_path}...")
    df1 = safe_read_csv(file1_path)
    
    print(f"Loading {file2_path}...")
    df2 = safe_read_csv(file2_path)
    
    print("Combining datasets...")
    # Put them in a list and stack them vertically (row-wise)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save the result as a clean, standardized UTF-8 file
    combined_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully saved combined file to: {output_path}")
    print(f"Total rows in combined dataset: {len(combined_df)}")
    


# ----------------------------
# FIXED FEATURE ORDER (CRITICAL)
# ----------------------------
EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


# ----------------------------
# DATASET (STRUCTURED OUTPUT)
# ----------------------------
class MentalHealthDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.labels = torch.tensor(df["action_id"].values, dtype=torch.long)

        emotions, risks, risks_vec, uncerts, embeddings = [], [], [], [], []

        for idx, raw_state in enumerate(df["state_vector"]):

            state_dict = raw_state

            while isinstance(state_dict, str):
                state_dict = re.sub(r'np\.\w+\((.*?)\)', r'\1', state_dict)
                state_dict = re.sub(r'(?:array|tensor)\((.*?)\)', r'\1', state_dict)
                state_dict = state_dict.replace("\n", " ").replace("\r", " ")

                try:
                    state_dict = ast.literal_eval(state_dict)
                except:
                    try:
                        state_dict = json.loads(state_dict.replace("'", '"'))
                    except:
                        state_dict = {}

            # ---------------- EMOTION ----------------
            emotion_vec = [
                float(state_dict.get("emotion", {}).get(k, 0.0))
                for k in EMOTION_ORDER
            ]

            # ---------------- RISK ----------------
            risk = float(state_dict.get("risk", 0.0))
            risk_vector = state_dict.get("risk_vector", [0.0] * 9)
            risk_vector = [float(x) for x in risk_vector]

            # ---------------- UNCERTAINTY ----------------
            uncert = float(state_dict.get("risk_uncertainty", 0.0))

            # ---------------- EMBEDDING ----------------
            emb_key = "embedding" if "embedding" in state_dict else "embeddings"
            embedding = state_dict.get(emb_key, [0.0] * 384)
            embedding = [float(x) for x in embedding]

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
# MODEL (RISK-AWARE ARCHITECTURE)
# ----------------------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.emotion_net = nn.Linear(6, 16)
        # self.risk_net = nn.Linear(1, 8)
        self.risk_net = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32)
        )
        # self.risk_vec_net = nn.Linear(9, 27)
        self.risk_vec_net = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.uncert_net = nn.Linear(1, 4)
        self.embed_net = nn.Linear(384, 128)

        self.classifier = nn.Sequential(
            nn.Linear(16 + 32 + 64 + 4 + 128,256),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
        
            nn.Linear(128,64),
            nn.ReLU(),
        
            nn.Linear(64,7)
        )

    def forward(self, emotion, risk, risk_vec, uncert, embed):
        e = torch.relu(self.emotion_net(emotion))
        r = torch.relu(self.risk_net(risk))
        rv = torch.relu(self.risk_vec_net(risk_vec))
        u = torch.relu(self.uncert_net(uncert))
        emb = torch.relu(self.embed_net(embed))

        x = torch.cat([e, r, rv, u, emb], dim=1)
        return self.classifier(x)




# FILE_1 = "./data/action_label_dataset_llm.csv"
# FILE_2 = "./data/action_label_dataset_llm_2.csv"
# OUTPUT_FILE = "./data/action_label_dataset_llm_combined.csv"

# combine_two_csvs(FILE_1, FILE_2, OUTPUT_FILE)

        


dataset = MentalHealthDataset(CSV_PATH)
labels = dataset.labels.numpy()
indices = np.arange(len(dataset))

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# train_ds, val_ds, test_ds = random_split(
#     dataset,
#     [train_size, val_size, test_size]
# )
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

temp_labels = labels[temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)

from torch.utils.data import Subset

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)

all_features = torch.cat([
    dataset.emotions,
    dataset.risks,
    dataset.risks_vec,
    dataset.uncerts,
    dataset.embeddings
], dim=1).numpy()

scaler = StandardScaler()
scaler.fit(all_features[train_idx])

scaled_features = scaler.transform(all_features)

e_dim = dataset.emotions.shape[1]
r_dim = dataset.risks.shape[1]
rv_dim = dataset.risks_vec.shape[1]
u_dim = dataset.uncerts.shape[1]

dataset.emotions = torch.tensor(scaled_features[:, :e_dim], dtype=torch.float32)
dataset.risks = torch.tensor(scaled_features[:, e_dim:e_dim+r_dim], dtype=torch.float32)
dataset.risks_vec = torch.tensor(scaled_features[:, e_dim+r_dim:e_dim+r_dim+rv_dim], dtype=torch.float32)
dataset.uncerts = torch.tensor(scaled_features[:, e_dim+r_dim+rv_dim:e_dim+r_dim+rv_dim+u_dim], dtype=torch.float32)
dataset.embeddings = torch.tensor(scaled_features[:, e_dim+r_dim+rv_dim+u_dim:], dtype=torch.float32)

# train_ds, val_ds, test_ds = random_split(
#     dataset, 
#     [train_size, val_size, test_size]
# )

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)



# train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Policy().to(device)

# class_counts = [1507, 4036, 216, 331, 114, 2924, 1186]
# total = sum(class_counts)

# weights = torch.tensor(
#     [total / (7 * c) for c in class_counts],
#     dtype=torch.float32
# ).to(device)

df = pd.read_csv(
    CSV_PATH
)

class_counts = (
    df["action_id"]
    .value_counts()
    .sort_index()
)


print(class_counts)

weights = 1.0 / torch.tensor(
    class_counts.values,
    dtype=torch.float32,
    device=device
)

weights = weights / weights.sum() * len(weights)

# criterion = nn.CrossEntropyLoss(weight=weights)
# optimizer = optim.Adam(model.parameters(), lr=3e-4)


# TRAIN
# EPOCHS = 30

# for epoch in range(EPOCHS):
#     model.train()

#     correct, total_samples = 0, 0
#     epoch_loss = 0

#     for e, r, rv, u, emb, y in train_loader:
#         e, r, rv, u, emb, y = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device), y.to(device)

#         logits = model(e, r, rv, u, emb)
#         loss = criterion(logits, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()

#         preds = logits.argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total_samples += y.size(0)

#     acc = correct / total_samples

#     print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f}")


SEEDS = [0, 42, 123, 256, 512]

all_accuracies = []
all_f1 = []

for seed in SEEDS:

    print("="*60)
    print(f"Training with seed {seed}")
    print("="*60)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Policy().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    EPOCHS = 30

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total_samples = 0
        epoch_loss = 0

        for e,r,rv,u,emb,y in train_loader:

            e,r,rv,u,emb,y = (
                e.to(device),
                r.to(device),
                rv.to(device),
                u.to(device),
                emb.to(device),
                y.to(device)
            )

            logits = model(e,r,rv,u,emb)

            loss = criterion(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(1)

            correct += (preds==y).sum().item()

            total_samples += y.size(0)

        acc = correct / total_samples

        print(
            f"Seed {seed} "
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss={epoch_loss/len(train_loader):.4f} "
            f"Acc={acc:.4f}"
        )





# model.eval()
# all_preds = []
# all_trues = []

# with torch.no_grad():
#     for e, r, rv, u, emb, y in val_loader:
#         e, r, rv, u, emb = e.to(device), r.to(device), rv.to(device), u.to(device), emb.to(device)
#         logits = model(e, r, rv, u, emb)
#         preds = logits.argmax(dim=1).cpu().numpy()
        
#         all_preds.extend(preds)
#         all_trues.extend(y.numpy())

# cm = confusion_matrix(all_trues, all_preds)
# print("Validation Confusion Matrix:\n", cm)



# print(classification_report(
#     all_trues,
#     all_preds,
#     target_names=[
#         "empathy",
#         "reflection",
#         "clarifying",
#         "coping",
#         "safety",
#         "professional",
#         "abstain"
#     ]
# ))



    model.eval()
    
    preds = []
    labels = []
    
    with torch.no_grad():
    
        for e,r,rv,u,emb,y in test_loader:
    
            e,r,rv,u,emb = (
                e.to(device),
                r.to(device),
                rv.to(device),
                u.to(device),
                emb.to(device)
            )
    
            logits = model(e,r,rv,u,emb)
    
            pred = logits.argmax(1)
    
            preds.extend(pred.cpu().numpy())
    
            labels.extend(y.numpy())
    
    acc = accuracy_score(labels,preds)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )
    
    all_accuracies.append(acc)
    all_f1.append(f1)
    
    print(classification_report(
        labels,
        preds,
        target_names=[
            "empathy",
            "reflection",
            "clarifying",
            "coping",
            "safety",
            "professional",
            "abstain"
        ],
        zero_division=0
    ))


print("\n")
print("="*70)
print("FINAL RESULTS OVER MULTIPLE RUNS")
print("="*70)

all_accuracies = np.array(all_accuracies)
all_f1 = np.array(all_f1)

print()

print(
    f"Accuracy: {all_accuracies.mean():.4f} ± {all_accuracies.std(ddof=1):.4f}"
)

print(
    f"Macro F1: {all_f1.mean():.4f} ± {all_f1.std(ddof=1):.4f}"
)


from scipy.stats import t

def confidence_interval(x):

    mean = np.mean(x)

    sem = stats.sem(x)

    ci = t.interval(
        confidence=0.95,
        df=len(x)-1,
        loc=mean,
        scale=sem
    )

    return ci

ci_acc = confidence_interval(all_accuracies)
ci_f1 = confidence_interval(all_f1)

print()

print("95% Confidence Interval (Accuracy):")
print(ci_acc)

print()

print("95% Confidence Interval (Macro F1):")
print(ci_f1)

# ----------------------------
# SAVE MODEL
# ----------------------------
# torch.save(model.state_dict(), MODEL_PATH)
# print("Model saved.")