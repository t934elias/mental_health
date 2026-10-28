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

model = Policy().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

# model.eval()
        

# all_preds = []
# all_labels = []

# model.eval()

# for e, r, rv, u, emb, y in test_loader:

#     batch = len(y)

#     preds = np.random.randint(0, 7, size=batch)

#     all_preds.extend(preds)
#     all_labels.extend(y.numpy())

# print("Random Policy")
# print("Accuracy:", accuracy_score(all_labels, all_preds))

# print(classification_report(
#     all_labels,
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






# def rule_policy(emotion, risk, uncert):

#     sadness = emotion[0]      # adjust index if necessary

#     risk = risk.item()

#     uncert = uncert.item()

#     if risk > 0.75:
#         return 4      # safety

#     elif risk > 0.50:
#         return 5      # professional help

#     elif sadness > 0.60:
#         return 0      # empathy

#     elif uncert > 0.15:
#         return 6      # abstain

#     else:
#         return 1      # reflection


# from sklearn.metrics import accuracy_score

# preds = []
# labels = []

# for e, r, rv, u, emb, y in test_loader:

#     for i in range(len(y)):

#         pred = rule_policy(
#             e[i],
#             r[i],
#             u[i]
#         )

#         preds.append(pred)

#     labels.extend(y.numpy())

# print("Rule-Based Policy")
# print("Accuracy:", accuracy_score(labels, preds))

# print(classification_report(
#     labels,
#     preds,
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

preds_ddqn = []
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

        preds_ddqn.extend(pred.cpu().numpy())

        labels.extend(y.numpy())

print("Our Model")

print(classification_report(
    labels,
    preds_ddqn,
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


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

acc_ddqn = accuracy_score(labels, preds_ddqn)

precision_ddqn, recall_ddqn, f1_ddqn, _ = precision_recall_fscore_support(
    labels,
    preds_ddqn,
    average="macro",
    zero_division=0
)

print(f"Accuracy : {acc_ddqn:.4f}")
print(f"Precision: {precision_ddqn:.4f}")
print(f"Recall   : {recall_ddqn:.4f}")
print(f"Macro F1 : {f1_ddqn:.4f}")







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
        
    # # Set the network to evaluation mode (highly important for MC Dropout/BatchNorm behavior!)
    # trainer.model.eval()
    # print("Model loaded successfully and set to eval mode.\n")
    
    # # --- 4. Interactive Test Cycle ---
    # print("=" * 60)
    # print("Interactive Policy Simulator")
    # print("Type 'exit' to quit.")
    # print("=" * 60)
    
    # user_input = "I can't seem to get out of bed today, everything feels completely pointless."
    # print(f"Initial Patient: {user_input}")
    
    # while True:
    #     # Build the state from user message
    #     state = pipeline.build_state(user_input)
        
    #     # Select the action using the loaded model (disabling random exploration/epsilon)
    #     # Note: Depending on your DQNTrainer implementation, you might need to pass force_exploit=True 
    #     # or similar to bypass epsilon-greedy exploration.
    #     action_id, uncertainty = trainer.select_action(state)
    #     action_label = rl1.ACTIONS[action_id]
        
    #     print("-" * 60)
    #     print(f"🔍 System Uncertainty Estimation: {uncertainty:.4f}")
    #     print(f"🤖 Selected Strategy Action   : {action_label.upper()} (ID: {action_id})")
        
    #     # Generate the text response based on the selected action strategy
    #     bot_response = pipeline.generate_response(user_input, action_label)
    #     print(f"🤖 Bot Response                : \"{bot_response}\"")
    #     print("-" * 60)
        
    #     # Get next input
    #     user_input = input("\nEnter patient reply: ")
    #     if user_input.lower() == 'exit':
    #         break


trainer = test_saved_policy()

import torch
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

model = trainer.policy_net
# 1. Set model to evaluation mode
model.eval()

preds_dqn = []
labels = []

# 2. Collect predictions across the entire test set
with torch.no_grad():
    for e, r, rv, u, emb, y in test_loader:
        # Move inputs to the correct device
        e = e.to(device)
        r = r.to(device)
        rv = rv.to(device)
        u = u.to(device)
        emb = emb.to(device)

        # Forward pass
        logits = model(e, r, rv, u, emb)
        pred = logits.argmax(dim=1)

        # Safely append predictions and ground truth
        preds_dqn.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())  # Added .cpu() to prevent GPU crashes

# --- Evaluation calculations are now OUTSIDE the loop ---
print("\n" + "=" * 50)
print("DQN Model Evaluation Results")
print("=" * 50)

# 3. Print the comprehensive classification report
target_names = [
    "empathy",
    "reflection",
    "clarifying",
    "coping",
    "safety",
    "professional",
    "abstain"
]

print(classification_report(
    labels,
    preds_dqn,
    target_names=target_names,
    zero_division=0
))

# 4. Calculate and print overall metrics
acc_dqn = accuracy_score(labels, preds_dqn)
precision_dqn, recall_dqn, f1_dqn, _ = precision_recall_fscore_support(
    labels,
    preds_dqn,
    average="macro",
    zero_division=0
)

print("-" * 50)
print(f"Accuracy : {acc_dqn:.4f}")
print(f"Precision: {precision_dqn:.4f}")
print(f"Recall   : {recall_dqn:.4f}")
print(f"Macro F1 : {f1_dqn:.4f}")
print("=" * 50)









# trainer = test_saved_policy("./models/threshold_policy_model.pt")

# import torch
# from sklearn.metrics import (
#     classification_report,
#     accuracy_score,
#     precision_recall_fscore_support,
# )

# model = trainer.policy_net
# # 1. Set model to evaluation mode
# model.eval()

# preds = []
# labels = []

# # 2. Collect predictions across the entire test set
# with torch.no_grad():
#     for e, r, rv, u, emb, y in test_loader:
#         # Move inputs to the correct device
#         e = e.to(device)
#         r = r.to(device)
#         rv = rv.to(device)
#         u = u.to(device)
#         emb = emb.to(device)

#         # Forward pass
#         logits = model(e, r, rv, u, emb)
#         pred = logits.argmax(dim=1)

#         # Safely append predictions and ground truth
#         preds.extend(pred.cpu().numpy())
#         labels.extend(y.cpu().numpy())  # Added .cpu() to prevent GPU crashes

# # --- Evaluation calculations are now OUTSIDE the loop ---
# print("\n" + "=" * 50)
# print(" Model Evaluation Results uncertainty threshold 0.9")
# print("=" * 50)

# # 3. Print the comprehensive classification report
# target_names = [
#     "empathy",
#     "reflection",
#     "clarifying",
#     "coping",
#     "safety",
#     "professional",
#     "abstain"
# ]

# print(classification_report(
#     labels,
#     preds,
#     target_names=target_names,
#     zero_division=0
# ))

# # 4. Calculate and print overall metrics
# acc = accuracy_score(labels, preds)
# precision, recall, f1, _ = precision_recall_fscore_support(
#     labels,
#     preds,
#     average="macro",
#     zero_division=0
# )

# print("-" * 50)
# print(f"Accuracy : {acc:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"Macro F1 : {f1:.4f}")
# print("=" * 50)






import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

labels = np.array(labels)
preds_dqn = np.array(preds_dqn)
preds_ddqn = np.array(preds_ddqn)

# Correct / incorrect predictions
dqn_correct = preds_dqn == labels
ddqn_correct = preds_ddqn == labels

# Contingency table
both_correct = np.sum(dqn_correct & ddqn_correct)
dqn_only = np.sum(dqn_correct & ~ddqn_correct)
ddqn_only = np.sum(~dqn_correct & ddqn_correct)
both_wrong = np.sum(~dqn_correct & ~ddqn_correct)

table = [
    [both_correct, dqn_only],
    [ddqn_only, both_wrong]
]

print(table)

result = mcnemar(table, exact=False, correction=True)

print("Statistic:", result.statistic)
print("p-value:", result.pvalue)