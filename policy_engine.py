import torch
import torch.nn as nn
import numpy as np
import pickle

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

from sentence_transformers import SentenceTransformer

from state_builder import build_state


############################################################
# LOAD SCALER
############################################################

with open("./models/policy_scaler_llm_2.pkl", "rb") as f:
    scaler = pickle.load(f)


############################################################
# POLICY MODEL
############################################################

EMOTION_ORDER = [
    "sadness",
    "fear",
    "anger",
    "confusion",
    "joy",
    "neutral"
]


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.emotion_net = nn.Linear(6,16)

        self.risk_net = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,32)
        )

        self.risk_vec_net = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )

        self.uncert_net = nn.Linear(1,4)

        self.embed_net = nn.Linear(384,128)

        self.classifier = nn.Sequential(
            nn.Linear(244,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,7)
        )

    def forward(self,e,r,rv,u,emb):

        e = torch.relu(self.emotion_net(e))
        r = torch.relu(self.risk_net(r))
        rv = torch.relu(self.risk_vec_net(rv))
        u = torch.relu(self.uncert_net(u))
        emb = torch.relu(self.embed_net(emb))

        x = torch.cat([e,r,rv,u,emb],dim=1)

        return self.classifier(x)


############################################################
# ACTIONS
############################################################

ACTIONS = {
    0:"empathy",
    1:"reflection",
    2:"clarifying_question",
    3:"coping_strategy",
    4:"safety_check",
    5:"professional_help",
    6:"abstain"
}

STRATEGIES = {

"empathy":
"""
Respond with empathy.

Validate the user's feelings.

Do not immediately give advice.
""",

"reflection":
"""
Reflect the user's emotions.

Rephrase their thoughts.

Do not ask questions.
""",

"clarifying_question":
"""
Ask ONE open-ended question.

Do not give advice.
""",

"coping_strategy":
"""
Suggest ONE evidence-based coping strategy.

Explain why it may help.
""",

"safety_check":
"""
Prioritize safety.

Check whether the user is currently safe.

If there is immediate danger, encourage contacting emergency services or a trusted person.
""",

"professional_help":
"""
Encourage professional help.

Be supportive.

Normalize therapy or counselling.
""",

"abstain":
"""
You are uncertain.

Ask the user for more information.
"""
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = Policy().to(device)
policy.load_state_dict(
    torch.load("./models/policy_model_llm_2.pt",
               map_location=device)
)
policy.eval()


tokenizers = {

    "emotion":
        AutoTokenizer.from_pretrained("./models/emotion_model"),

    "risk":
        AutoTokenizer.from_pretrained("./models/risk_model")
}


models = {

    "emotion":
        AutoModelForSequenceClassification.from_pretrained("./models/emotion_model"),

    "risk":
        AutoModelForSequenceClassification.from_pretrained("./models/risk_model"),

    "encoder":
        SentenceTransformer("all-MiniLM-L6-v2")
}


###################################################
# RESPONSE LLM
###################################################

llm_name = "Qwen/Qwen2.5-3B-Instruct"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
)



def predict_action(state):

    emotion = np.array([
        state["emotion"].get(k,0.0)
        for k in EMOTION_ORDER
    ]).reshape(1,-1)

    risk = np.array([[state["risk"]]])

    risk_vec = np.array(state["risk_vector"]).reshape(1,-1)

    uncert = np.array([[state["risk_uncertainty"]]])

    embedding = np.array(state["embedding"]).reshape(1,-1)

    features = np.concatenate([
        emotion,
        risk,
        risk_vec,
        uncert,
        embedding
    ],axis=1)

    features = scaler.transform(features)

    e = torch.tensor(features[:,:6],dtype=torch.float32).to(device)
    r = torch.tensor(features[:,6:7],dtype=torch.float32).to(device)
    rv = torch.tensor(features[:,7:16],dtype=torch.float32).to(device)
    u = torch.tensor(features[:,16:17],dtype=torch.float32).to(device)
    emb = torch.tensor(features[:,17:],dtype=torch.float32).to(device)

    with torch.no_grad():

        logits = policy(e,r,rv,u,emb)

        probs = torch.softmax(logits,dim=1)[0]

        pred = torch.argmax(probs).item()

    return ACTIONS[pred], probs



def build_prompt(user_text, action):

    return f"""
You are an empathetic mental-health assistant.

Conversation strategy:

{STRATEGIES[action]}

User:

{user_text}

Write ONE natural response.

Do not mention the strategy.
"""




def generate(user_text, action):

    messages = [
    {
        "role": "system",
        "content":
        f"""
    You are an empathetic mental-health assistant.
    
    {STRATEGIES[action]}
    
    Write ONE natural response.
    
    Do not mention the strategy.
    """
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # inputs = llm_tokenizer(
    #     prompt,
    #     return_tensors="pt"
    # ).to(llm.device)
    inputs = llm_tokenizer(
        text,
        return_tensors="pt"
    ).to(llm.device)

    # outputs = llm.generate(
    #     **inputs,
    #     max_new_tokens=150,
    #     temperature=0.7
    # )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    
    response = llm_tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    # return llm_tokenizer.decode(
    #     outputs[0],
    #     skip_special_tokens=True
    # )
    return response


RISK_THRESHOLD = 0.80
UNCERTAINTY_THRESHOLD = 0.8

def choose_action(state):

    risk = state["risk"]
    uncertainty = state["risk_uncertainty"]

    if uncertainty >= UNCERTAINTY_THRESHOLD:

        return "abstain", None

    if risk >= RISK_THRESHOLD:

        return "safety_check", None


    action, probs = predict_action(state)

    return action, probs



def chat(user_text):

    state = build_state(
        user_text,
        models,
        tokenizers
    )

    action, probs = choose_action(state)

    # prompt = build_prompt(
    #     user_text,
    #     action
    # )

    response = generate(user_text, action)

    return {
        "risk": state["risk"],
        "uncertainty": state["risk_uncertainty"],
        "strategy": action,
        "response": response,
        "probabilities": probs.detach().cpu().numpy() if probs is not None else probs
    }







while True:

    text = input("User: ")

    result = chat(text)

    print(f"\nRisk: {result['risk']:.3f}")
    print(f"Strategy: {result['strategy']}")
    print(f"\nAssistant: {result['response']}\n")