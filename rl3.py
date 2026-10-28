"""
Part 3 - Main RL Training Loop

This file assumes Parts 1 and 2 are already imported.
"""
from rl2a import DQNTrainer
# from train_dqn import DQNTrainer
from user_simulation import MentalHealthPipeline, simulate_user_turn
import rl1
import csv
import os


log_file = "./models/training_logs.csv"
# log_file = "./models/training_logs_dqn.csv"
# log_file = "./models/training_logs_threshold.csv"
file_exists = os.path.isfile(log_file)


def train_rl(
    pipeline,
    trainer,
    num_episodes=rl1.NUM_EPISODES,
    max_steps=rl1.MAX_STEPS,
):
    best_reward = float("-inf")

    for episode in range(num_episodes):

        previous_action_id = -1

        current_message = (
            "I can't seem to get out of bed today, "
            "everything feels completely pointless."
        )

        dialogue_history = [f"Patient: {current_message}"]

        episode_stats = trainer.initialize_episode_stats()

        print("\n" + "=" * 70)
        print(f"Starting Episode {episode+1}/{num_episodes}")
        print("=" * 70)

        for step in range(max_steps):

            state = pipeline.build_state(current_message)

            action_id, uncertainty = trainer.select_action(state)
            action_label = rl1.ACTIONS[action_id]

            dialogue_history.append(f"Bot Action: {action_label}")

            bot_response = pipeline.generate_response(
                current_message,
                action_label
            )

            dialogue_history.append(f"Bot: {bot_response}")

            next_user_message = simulate_user_turn(
                state,
                action_label,
                dialogue_history,
                pipeline
            )

            dialogue_history.append(
                f"Patient: {next_user_message}"
            )

            next_state = pipeline.build_state(
                next_user_message
            )

            reward = rl1.calculate_reward(
                state,
                next_state,
                action_id,
                uncertainty,
                previous_action_id,
            )

            done = rl1.is_terminal(next_state)

            trainer.remember(
                state,
                action_id,
                reward,
                next_state,
                done
            )

            train_stats = trainer.train_step()

            episode_stats = trainer.update_episode_stats(
                episode_stats,
                reward,
                action_id,
                next_state["risk"],
                uncertainty,
                train_stats
            )

            trainer.print_training_status(
                episode + 1,
                step + 1,
                reward,
                train_stats
            )

            current_message = next_user_message
            previous_action_id = action_id

            if done:
                print("\nEpisode terminated early.")
                break

        trainer.summarize_episode(
            episode + 1,
            episode_stats
        )

        best_reward = trainer.save_best_model(
            episode_stats["episode_reward"],
            best_reward
        )

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["episode", "total_reward", "avg_loss", "avg_risk", "avg_uncertainty"])
            
            writer.writerow([
                episode + 1,       
                episode_stats,      
                uncertainty
            ])

        if (episode + 1) % 5 == 0:
            trainer.save_checkpoint(
                episode=episode + 1
            )

    trainer.save_checkpoint(
        filename="./models/final_policy_model_2.pt",
        # filename="./models/threshold_policy_model.pt",
        episode=num_episodes
    )

    print("\nTraining complete.")




if __name__ == "__main__":

    print("Initializing RL Training...")

    pipeline = MentalHealthPipeline()

    trainer = DQNTrainer(pipeline)

    train_rl(
        pipeline,
        trainer,
        num_episodes=rl1.NUM_EPISODES,
        max_steps=rl1.MAX_STEPS
    )


