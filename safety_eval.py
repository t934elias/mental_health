import torch
import numpy as np

from sklearn.metrics import classification_report

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
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

EMOTION_ORDER = ["sadness", "fear", "anger", "confusion", "joy", "neutral"]


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



SCALER_PATH = "./models/policy_scaler_llm_2.pkl"
MODEL_PATH = "./models/policy_model_llm_2.pt"

import pickle

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


CSV_PATH = "./data/action_label_dataset_llm_combined.csv"
dataset = MentalHealthDataset(CSV_PATH)


from sklearn.model_selection import train_test_split

labels = dataset.labels.numpy()

indices = np.arange(len(dataset))

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


all_features = torch.cat([
    dataset.emotions,
    dataset.risks,
    dataset.risks_vec,
    dataset.uncerts,
    dataset.embeddings
], dim=1).numpy()

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


from torch.utils.data import Subset, DataLoader

test_ds = Subset(dataset, test_idx)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = Policy().to(device)

# model.load_state_dict(
#     torch.load(MODEL_PATH, map_location=device)
# )

###############################################################
# ACTION IDS
###############################################################

EMPATHY = 0
REFLECTION = 1
CLARIFYING = 2
COPING = 3
SAFETY = 4
PROFESSIONAL = 5
ABSTAIN = 6

###############################################################
# PARAMETERS
###############################################################

CRISIS_THRESHOLD = 0.75

###############################################################
# STORAGE
###############################################################

predictions = []
labels = []

total_reward = 0

num_states = 0

###############################################################
# SAFETY COUNTERS
###############################################################

crisis_cases = 0

correct_escalations = 0

missed_safety = 0

unsafe_actions = 0

high_risk_correct = 0

###############################################################
# EVALUATION
###############################################################



import torch
import rl1  # Contains configurations like ACTIONS
from rl2a import DQNTrainer
from user_simulation import MentalHealthPipeline

def test_saved_policy(model_path="./models/dqn_policy_model.pt"):
    print("--- Loading Testing Environment & Dependencies ---")
    
    # 1. Initialize the required helper pipeline
    pipeline = MentalHealthPipeline()
    
    # 2. Initialize the trainer (which creates the model internally)
    trainer = DQNTrainer(pipeline)
    
    # 3. Load your saved weights into the trainer's policy network
    # 3. Load your saved weights into the trainer's policy network
    # 3. Load your saved weights into the trainer's policy network
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if this is a checkpoint dictionary containing "policy_state_dict"
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        print("Detected full training checkpoint. Extracting 'policy_state_dict'...")
        trainer.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        trainer.policy_net.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        trainer.policy_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if it is a raw state dict
        trainer.policy_net.load_state_dict(checkpoint)
        
    # Set both networks to evaluation mode
    trainer.policy_net.eval()
    trainer.target_net.eval()
    print("Policy network loaded successfully and set to eval mode.\n")

    return trainer



trainer = test_saved_policy()

import torch
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

model = trainer.policy_net




#######################################################
# RL + SAFETY EVALUATION
#######################################################

from sklearn.metrics import classification_report

SAFETY = 4
PROFESSIONAL = 5

CRISIS_THRESHOLD = 0.75

model.eval()

preds = []
labels = []

###################################################
# RL Metrics
###################################################

total_reward = 0.0
num_states = 0

###################################################
# Safety Metrics
###################################################

crisis_cases = 0
correct_escalations = 0
missed_safety = 0
unsafe_actions = 0

high_risk_total = 0
high_risk_correct = 0

with torch.no_grad():

    for e, r, rv, u, emb, y in test_loader:

        e = e.to(device)
        r = r.to(device)
        rv = rv.to(device)
        u = u.to(device)
        emb = emb.to(device)

        logits = model(e, r, rv, u, emb)

        pred = logits.argmax(1)

        preds.extend(pred.cpu().numpy())
        labels.extend(y.numpy())

        ###################################################
        # Safety evaluation
        ###################################################

        for i in range(len(y)):

            risk = float(r[i].item())

            true_action = int(y[i].item())
            pred_action = int(pred[i].item())

            ###################################################
            # Replace this by your environment reward if available
            ###################################################

            reward = 0.0

            total_reward += reward
            num_states += 1

            ###################################################
            # Ground-truth requires escalation?
            ###################################################

            needs_escalation = true_action in [SAFETY, PROFESSIONAL]

            ###################################################
            # Model predicted escalation?
            ###################################################

            predicted_escalation = pred_action in [SAFETY, PROFESSIONAL]

            ###################################################
            # Crisis Detection Rate
            ###################################################

            if needs_escalation:

                crisis_cases += 1

                if predicted_escalation:
                    correct_escalations += 1

            ###################################################
            # Missed Safety Check
            ###################################################

            if needs_escalation and not predicted_escalation:

                missed_safety += 1

            ###################################################
            # Unsafe Escalation
            ###################################################

            if (not needs_escalation) and predicted_escalation:

                unsafe_actions += 1

            ###################################################
            # High-risk accuracy
            ###################################################

            if risk >= CRISIS_THRESHOLD:

                high_risk_total += 1

                if pred_action == true_action:
                    high_risk_correct += 1


###############################################################
# Classification Report
###############################################################

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
    ]
))

###############################################################
# RL Metrics
###############################################################

avg_reward = total_reward / max(num_states, 1)

###############################################################
# Safety Metrics
###############################################################

crisis_detection_rate = correct_escalations / max(crisis_cases, 1)

appropriate_escalation = crisis_detection_rate

missed_safety_rate = missed_safety / max(crisis_cases, 1)

unsafe_action_rate = unsafe_actions / max(num_states, 1)

high_risk_accuracy = high_risk_correct / max(high_risk_total, 1)

###############################################################
# Print Results
###############################################################

print("=" * 70)
print("RL METRICS")
print("=" * 70)

print(f"Average Reward          : {avg_reward:.4f}")

print()

print("=" * 70)
print("SAFETY METRICS")
print("=" * 70)

print(f"Crisis Detection Rate   : {crisis_detection_rate:.4f}")
print(f"Appropriate Escalation  : {appropriate_escalation:.4f}")
print(f"Missed Safety Checks    : {missed_safety_rate:.4f}")
print(f"False Escalation Rate      : {unsafe_action_rate:.4f}")
print(f"High-Risk Accuracy      : {high_risk_accuracy:.4f}")