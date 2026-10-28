"""
===========================================================================
Mental Health Reinforcement Learning Training
Part 1 - Imports, Constants, Replay Buffer, Reward Function, Tensor Builder
===========================================================================

This file is compatible with the existing MentalHealthPipeline.

Part 1 contains:
    - Imports
    - Global constants
    - Action definitions
    - Reward function
    - Replay Buffer
    - Tensor extraction
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from state_builder import build_state
from user_simulation import MentalHealthPipeline


# =============================================================================
# Global Constants
# =============================================================================

EMOTION_ORDER = [
    "sadness",
    "fear",
    "anger",
    "confusion",
    "joy",
    "neutral"
]

ACTIONS = {
    0: "empathy",
    1: "reflection",
    2: "clarifying_question",
    3: "coping_strategy",
    4: "safety_check",
    5: "professional_help",
    6: "abstain"
}

STRATEGIES = {

    "empathy":
        "Respond with empathy.\n"
        "Validate the user's feelings.\n"
        "Do not immediately give advice.",

    "reflection":
        "Reflect the user's emotions.\n"
        "Rephrase what they expressed.\n"
        "Do not ask questions.",

    "clarifying_question":
        "Ask ONE open-ended question.\n"
        "Do not give advice.",

    "coping_strategy":
        "Suggest ONE evidence-based coping strategy.\n"
        "Explain briefly why it may help.",

    "safety_check":
        "Prioritize safety.\n"
        "Check whether the user is currently safe.\n"
        "If immediate danger exists, encourage contacting emergency services or a trusted person.",

    "professional_help":
        "Encourage seeking professional help.\n"
        "Normalize therapy.\n"
        "Remain supportive.",

    "abstain":
        "You are uncertain.\n"
        "Ask the user for more information."
}


# =============================================================================
# Hyperparameters
# =============================================================================

BATCH_SIZE = 32
BUFFER_SIZE = 10000

GAMMA = 0.99

LEARNING_RATE = 1e-4

TARGET_UPDATE = 200

NUM_EPISODES = 50#50 # 200 or 400

MAX_STEPS = 8

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

RISK_THRESHOLD = 0.90

UNCERTAINTY_THRESHOLD = 0.90# 0.80

CHECKPOINT_PATH = "./models/policy_model_llm_rl.pt"


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:

    def __init__(self, capacity=BUFFER_SIZE):

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        return random.sample(self.buffer, batch_size)

    def __len__(self):

        return len(self.buffer)


# =============================================================================
# Reward Function
# =============================================================================

def calculate_reward(
    old_state,
    new_state,
    action_id,
    uncertainty,
    previous_action_id,
):
    """
    Reward Function

    Positive:
        - Risk reduction
        - Sadness reduction

    Negative:
        - Ignoring high-risk users
        - High uncertainty
        - Excessive empathy when not needed
    """

    reward = 0.0

    if action_id == 0:#empathy:
        if old_state["emotion"].get("sadness", 0.0) > 0.6:
            reward += 2

    if action_id == 1:#reflection:
        if old_state["emotion"].get("confusion", 0.0) > 0.5:
            reward += 2
    
    if action_id == 2:#clarifying_question:
        if uncertainty > 0.4:
            reward += 2
    
    if action_id == 3:#coping_strategy:
        if old_state["risk"] < 0.5:
            reward += 2
    
    if action_id == 5:#professional_help:
        if old_state["risk"] > 0.8:
            reward += 3
    
    if action_id == 4:#safety_check:
        if old_state["risk"] > 0.8:
            reward += 4
        else:
            reward -= 2

    if action_id == previous_action_id:
        reward -= 1.5


    # if action_id == 5: # professional_help
    #     if old_state["risk"] > 0.8:
    #         reward += 3
    #     elif old_state["risk"] < 0.4:
    #         reward -= 4.0

    # -------------------------------------------------------------------------
    # Risk Reduction
    # -------------------------------------------------------------------------

    risk_delta = old_state["risk"] - new_state["risk"]

    reward += risk_delta * 5.0

    # -------------------------------------------------------------------------
    # Sadness Reduction
    # -------------------------------------------------------------------------

    sadness_delta = (
        old_state["emotion"].get("sadness", 0.0)
        - new_state["emotion"].get("sadness", 0.0)
    )

    reward += sadness_delta * 2.0

    # -------------------------------------------------------------------------
    # Crisis Handling
    # -------------------------------------------------------------------------

    # if old_state["risk"] >= 0.70:

    #     if action_id != 4:
    #         reward -= 10.0

    #     else:
    #         reward += 3.0

    if old_state["risk"] >= 0.70:
        if action_id not in [4, 5]:  
            reward -= 5.0  
        else:
            reward += 2.0

    # -------------------------------------------------------------------------
    # Risk uncertainty penalty
    # -------------------------------------------------------------------------

    # if new_state["risk_uncertainty"] > 0.15:
    #     reward -= 2.0

    # if uncertainty > 0.15:
    #     reward -= 3.0
    uncertainty_delta = uncertainty - new_state.get("uncertainty", uncertainty)
    reward += uncertainty_delta * 2.0

    # -------------------------------------------------------------------------
    # Prevent excessive empathy
    # -------------------------------------------------------------------------

    if (
        action_id == 0 and
        old_state["emotion"].get("sadness", 0.0) < 0.20
    ):
        reward -= 0.5

    # -------------------------------------------------------------------------
    # Successful recovery bonus
    # -------------------------------------------------------------------------

    if new_state["risk"] < 0.20:
        reward += 5.0

    # -------------------------------------------------------------------------
    # Catastrophic deterioration
    # -------------------------------------------------------------------------

    if new_state["risk"] > 0.90:
        reward -= 5.0

    return reward


# =============================================================================
# Terminal State
# =============================================================================

def is_terminal(state):
    """
    Determines whether the episode should terminate.
    """

    if state["risk"] < 0.20:
        return True

    if state["risk"] > 0.95:
        return True

    return False


# =============================================================================
# Tensor Extraction
# =============================================================================

def extract_tensors(
    batch,
    device,
    scaler
):
    """
    Converts replay buffer transitions into scaled PyTorch tensors.
    """

    emotions = []
    risks = []
    risk_vectors = []
    uncertainties = []
    embeddings = []

    actions = []
    rewards = []

    next_emotions = []
    next_risks = []
    next_risk_vectors = []
    next_uncertainties = []
    next_embeddings = []

    dones = []

    for (
        state,
        action,
        reward,
        next_state,
        done
    ) in batch:

        # ---------------- Current State ---------------- #

        emotion = [
            state["emotion"].get(k, 0.0)
            for k in EMOTION_ORDER
        ]

        features = np.concatenate([

            np.array(emotion),

            [state["risk"]],

            state["risk_vector"],

            [state["risk_uncertainty"]],

            state["embedding"]

        ]).reshape(1, -1)

        features = scaler.transform(features)[0]

        emotions.append(features[:6])

        risks.append([features[6]])

        risk_vectors.append(features[7:16])

        uncertainties.append([features[16]])

        embeddings.append(features[17:])

        # ---------------- Action ---------------- #

        actions.append(action)

        rewards.append(reward)

        dones.append(done)

        # ---------------- Next State ---------------- #

        next_emotion = [
            next_state["emotion"].get(k, 0.0)
            for k in EMOTION_ORDER
        ]

        next_features = np.concatenate([

            np.array(next_emotion),

            [next_state["risk"]],

            next_state["risk_vector"],

            [next_state["risk_uncertainty"]],

            next_state["embedding"]

        ]).reshape(1, -1)

        next_features = scaler.transform(next_features)[0]

        next_emotions.append(next_features[:6])

        next_risks.append([next_features[6]])

        next_risk_vectors.append(next_features[7:16])

        next_uncertainties.append([next_features[16]])

        next_embeddings.append(next_features[17:])

    return (

        torch.FloatTensor(emotions).to(device),

        torch.FloatTensor(risks).to(device),

        torch.FloatTensor(risk_vectors).to(device),

        torch.FloatTensor(uncertainties).to(device),

        torch.FloatTensor(embeddings).to(device),

        torch.LongTensor(actions).to(device),

        torch.FloatTensor(rewards).to(device),

        torch.FloatTensor(next_emotions).to(device),

        torch.FloatTensor(next_risks).to(device),

        torch.FloatTensor(next_risk_vectors).to(device),

        torch.FloatTensor(next_uncertainties).to(device),

        torch.FloatTensor(next_embeddings).to(device),

        torch.FloatTensor(dones).to(device)
    )