import csv
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import torch.nn.functional as F


# CSV_PATH = "./data/state_dataset.csv"

# ACTIONS = {
#     0: "empathy",
#     1: "reflection",
#     2: "clarifying_question",
#     3: "coping_strategy",
#     5: "professional_help",
#     4: "safety_check",
#     6: "abstain"
# }
# action_labels_clean = [v.replace("_", " ") for v in ACTIONS.values()]
# action_labels = list(ACTIONS.values())

# df = pd.read_csv(CSV_PATH)
# dataset = df

# HF_TOKEN = "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"

# print("--- Phase 1: Labeling Dataset with MentalBERT ---")

# # classifier = pipeline("zero-shot-classification", model="uzh-ucl/mental-bert-base-uncased")

# model_name = "mental/mental-bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
# model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)

# output_csv_path = "./data/action_label_dataset.csv"


# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     # with torch.no_grad():
#     #     outputs = model(**inputs)
#     # return outputs.last_hidden_state[0, 0, :]
#     embeddings = outputs.last_hidden_state
#     attention = inputs["attention_mask"]

#     mask = attention.unsqueeze(-1).expand(embeddings.size()).float()

#     sentence_embedding = torch.sum(embeddings * mask, 1) / mask.sum(1)
#     return sentence_embedding

# action_embeddings = torch.stack([get_embedding(label) for label in action_labels])


# with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
#     fieldnames = ['text', 'predicted_action_label', 'action_id', 'state_vector']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()

#     for idx, row in dataset.iterrows():
#         text = row["text"]
        
#         text_emb = get_embedding(text)

#         similarities = F.cosine_similarity(text_emb.unsqueeze(0), action_embeddings)
#         # result = classifier(text, candidate_labels=action_labels)
#         # top_label = result['labels'][0]

#         top_action_index = torch.argmax(similarities).item()
#         top_label = action_labels[top_action_index]
        
#         action_id = [k for k, v in ACTIONS.items() if v == top_label][0]
        
#         # state_vector_list = item["user_state"].tolist()
#         # state_vector_string = json.dumps(state_vector_list)

#         state_dict = row["user_state"]
#         state_vector_string = json.dumps(state_dict)
        
#         writer.writerow({
#             'text': text,
#             'predicted_action_label': top_label,
#             'action_id': action_id,
#             'state_vector': state_vector_string
#         })
        

# print(f"\nSuccess! Labeled dataset saved to '{output_csv_path}'")



import json
from ollama import chat

CSV_PATH = "./data/state_dataset_2.csv"
output_csv_path = "./data/action_label_dataset_llm_2.csv"

ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    5: "professional_help",
    4: "safety_check",
    6: "abstain"
}

# action_labels_clean = [v.replace("_", " ") for v in ACTIONS.values()]
action_labels = list(ACTIONS.values())

df = pd.read_csv(CSV_PATH)
# df = df[:5]

models = ["mistral:latest"]

def build_prompt(text, action_labels):
    return f"""You are a psychologist, and you want to reply to the person with the best action that will benefit them.

Choose exactly ONE intervention.

Available actions:

{action_labels}

Conversation:
{text}

Return ONLY the action name."""

# dataset = [
#         "i am happy", 
#         "i want to die", 
#         "why is life so hard", 
#         "I don't want to live anymore.", 
#         "I feel so lonely and isolated from everyone, nothing matters anymore."
#     ]


with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['text', 'predicted_action_label', 'action_id', 'state_vector']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for idx, row in df.iterrows():
        text = row["text"]
        response = chat(
                model = models[0],
                messages = [
                    {
                        'role': 'user',
                        'content': build_prompt(text, action_labels),
                    },
                ]
            )
        top_label = response['message']['content'].strip().lower().replace(" ", "_")
        
        matching_ids = [k for k, v in ACTIONS.items() if v == top_label]
        action_id = matching_ids[0] if matching_ids else 6 
        
        if not matching_ids:
            top_label = ACTIONS[6]
        

        state_dict = row["user_state"]
        state_vector_string = json.dumps(state_dict)
        
        writer.writerow({
            'text': text,
            'predicted_action_label': top_label,
            'action_id': action_id,
            'state_vector': state_vector_string
        })