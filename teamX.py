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
        # NOTE: Update the checkpoint path above if the deployment directory changes.
        self.device = torch.device("cpu")
        cfg = PixelAgentConfig(obs_shape=(FRAME_STACK * 3, CROP_HEIGHT, CROP_WIDTH), device=self.device)
        self.agent = PixelPolicyAgent(cfg).to(self.device)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.agent.load(str(checkpoint), map_location=self.device)
        self.frames = deque(maxlen=FRAME_STACK)
        self.initialized = False
        self.last_frame: np.ndarray | None = None

    @staticmethod
    def _extract_frame_candidates(observation) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []

        def recurse(obj):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    recurse(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    recurse(value)
            else:
                arr = np.asarray(obj)
                if arr.ndim >= 3:
                    while arr.ndim > 3:
                        arr = arr[0]
                    if arr.ndim == 3:
                        candidates.append(arr)

        recurse(observation)
        return candidates

    @classmethod
    def _to_uint8_chw(cls, observation) -> np.ndarray:
        candidates = cls._extract_frame_candidates(observation)
        if not candidates:
            raise ValueError("No 3D observation found in observation payload.")
        obs = candidates[0]
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
        self.agent.policy.eval()
        try:
            frame = self._crop(self._to_uint8_chw(observation))
            self.last_frame = frame.copy()
        except ValueError as exc:
            if self.last_frame is None:
                raise exc
            frame = self.last_frame.copy()
        if frame.shape[0] == FRAME_STACK * 3:
            stacked = frame
        else:
            self._push_frame(frame)
            stacked = np.concatenate(list(self.frames), axis=0)
        stacked = np.ascontiguousarray(stacked)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device, dtype=torch.float32)
            logits, _, _ = self.agent.policy(obs_tensor)
            actions = [torch.argmax(logit, dim=-1) for logit in logits]
            action_tensor = torch.stack(actions, dim=-1)
        action = np.asarray(action_tensor.squeeze(0).cpu().numpy(), dtype=np.int32)
        return action
