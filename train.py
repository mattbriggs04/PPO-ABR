import torch
from student.network import ACNet
from student.ppo import PPO, OBS_DIM, QUALITY_LEVELS
import os

MODEL_PATH = "./ppo_model.pt"
def main():
    device = torch.device("cpu")
    torch.manual_seed(22)
    model = ACNet(obs_dim=OBS_DIM, act_dim=QUALITY_LEVELS)

    if os.path.exists(MODEL_PATH):
        print("Model exists, loading it in")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    ppo = PPO(model, device, "./tests/")
    ppo.train(updates_per_iteration=20, total_timesteps=400_000, save_path=MODEL_PATH)

if __name__ == "__main__":
    main()
