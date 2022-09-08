import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu")
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float, device="cpu")
            return self.model(state).numpy()

    def reset(self):
        pass
