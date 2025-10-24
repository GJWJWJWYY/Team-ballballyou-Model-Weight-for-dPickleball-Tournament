from collections import deque
from pathlib import Path

import numpy as np
import torch

from End_To_End_Model import PixelAgentConfig, PixelPolicyAgent


CROP_X1, CROP_Y1, CROP_X2, CROP_Y2 = 18, 25, 149, 76
CROP_WIDTH = CROP_X2 - CROP_X1
CROP_HEIGHT = CROP_Y2 - CROP_Y1
FRAME_STACK = 10


class TeamX:
    def __init__(self):
        checkpoint = Path(__file__).with_name("ballballyou_left_weight.pt")
        self.device = torch.device("cpu")
        cfg = PixelAgentConfig(obs_shape=(FRAME_STACK * 3, CROP_HEIGHT, CROP_WIDTH), device=self.device)
        self.agent = PixelPolicyAgent(cfg).to(self.device)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.agent.load(str(checkpoint), map_location=self.device)
        self.frames = deque(maxlen=FRAME_STACK)
        self.initialized = False

    @staticmethod
    def _to_uint8_chw(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation)
        if obs.ndim != 3:
            raise ValueError(f"Observation must be 3D, got shape={obs.shape}")
        channels_first = obs.shape[0] % 3 == 0 and obs.shape[1] >= 50 and obs.shape[2] >= 50
        if not channels_first:
            obs = np.transpose(obs, (2, 0, 1))
        if obs.dtype == np.uint8:
            return obs
        arr = obs.astype(np.float32)
        if arr.max() <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    @staticmethod
    def _crop(observation: np.ndarray) -> np.ndarray:
        return observation[:, CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

    def _push_frame(self, frame: np.ndarray):
        if not self.initialized:
            for _ in range(FRAME_STACK):
                self.frames.append(frame.copy())
            self.initialized = True
        else:
            self.frames.append(frame.copy())
            while len(self.frames) < FRAME_STACK:
                self.frames.append(frame.copy())

    def policy(self, observation: np.ndarray, reward: float) -> np.ndarray:
        frame = self._crop(self._to_uint8_chw(observation))
        if frame.shape[0] == FRAME_STACK * 3:
            stacked = frame
        else:
            self._push_frame(frame)
            stacked = np.concatenate(list(self.frames), axis=0)

        action, _, _, _ = self.agent.act(stacked, deterministic=False)
        return action.astype(np.int32)
