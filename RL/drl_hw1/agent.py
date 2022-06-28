import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self, path="agent.pkl"):
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.load(__file__[:-8] + path).to(self._device)

    def act(self, state):
        with torch.no_grad():
            logits = self.model(torch.tensor(state, device=self._device))
            return logits.argmax(-1).cpu().numpy()

    def reset(self):
        pass
