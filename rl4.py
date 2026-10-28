"""
Part 4 - Evaluation and Testing Utilities

Use after RL training has completed.
"""

import numpy as np
import torch
from rl2a import DQNTrainer
from user_simulation import MentalHealthPipeline, simulate_user_turn
import rl1


def evaluate_agent(pipeline, trainer, prompts, max_steps=5):
    """
    Evaluate the trained policy on a list of initial prompts.
    """
    trainer.eval_mode()

    rewards = []
    risks = []

    for idx, prompt in enumerate(prompts):

        current_message = prompt
        dialogue_history = [f"Patient: {current_message}"]

        episode_reward = 0.0

        previous_action_id=-1

        print("\n" + "="*70)
        print(f"Evaluation Episode {idx+1}")
        print("="*70)

        for step in range(max_steps):

            state = pipeline.build_state(current_message)

            action_id, uncertainty = trainer.evaluate_policy(state)
            action_label = rl1.ACTIONS[action_id]

            response = pipeline.generate_response(
                current_message,
                action_label
            )

            dialogue_history.append(f"Bot: {response}")

            next_message = simulate_user_turn(
                state,
                action_label,
                dialogue_history,
                pipeline
            )

            next_state = pipeline.build_state(next_message)

            reward = rl1.calculate_reward(
                state,
                next_state,
                action_id,
                uncertainty,
                previous_action_id
            )

            episode_reward += reward
            risks.append(next_state["risk"])

            print(f"Step {step+1}")
            print(f"Action : {action_label}")
            print(f"Reward : {reward:.3f}")
            print(f"Risk   : {next_state['risk']:.3f}")
            print("-"*40)

            current_message = next_message
            previous_action_id = action_id

            if rl1.is_terminal(next_state):
                break

        rewards.append(episode_reward)

    print("\n========== FINAL RESULTS ==========")
    print(f"Mean Reward : {np.mean(rewards):.3f}")
    print(f"Std Reward  : {np.std(rewards):.3f}")
    if risks:
        print(f"Mean Risk   : {np.mean(risks):.3f}")

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_risk": float(np.mean(risks)) if risks else 0.0,
    }


def interactive_demo(pipeline, trainer):
    """
    Chat interactively with the trained policy.
    """
    trainer.eval_mode()

    print("\nInteractive Demo")
    print("Type 'quit' to exit.\n")

    history = []

    while True:

        user = input("Patient: ")

        if user.lower() == "quit":
            break

        state = pipeline.build_state(user)

        action_id, uncertainty = trainer.evaluate_policy(state)
        action = rl1.ACTIONS[action_id]

        reply = pipeline.generate_response(user, action)

        print(f"Action      : {action}")
        print(f"Uncertainty : {uncertainty:.3f}")
        print(f"Assistant   : {reply}\n")

        history.append((user, reply))


if __name__ == "__main__":

    pipeline = MentalHealthPipeline()
    trainer = DQNTrainer(pipeline)

    trainer.load_checkpoint("./models/final_policy_model.pt")

    # test_prompts = [
    #     "I don't see the point anymore.",
    #     "I'm anxious all the time.",
    #     "I feel completely alone.",
    #     "Nothing makes me happy anymore."
    # ]

    # results = evaluate_agent(
    #     pipeline,
    #     trainer,
    #     test_prompts
    # )

    # print(results)

    interactive_demo(pipeline, trainer)
