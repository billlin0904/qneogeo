"""Temporary teacher-policy regularisation for StrategyV2 migration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class KickstartTeacherCallback(BaseCallback):
    """Keep a migrated policy near its V1 teacher while new features settle.

    The loss is applied to rollout observations after collection. It only
    matches the teacher's legal-action distribution and decays to zero, so PPO
    remains free to improve once StrategyV2 features become useful.
    """

    def __init__(
        self,
        teacher_model_path: str | Path,
        *,
        legacy_observation_size: int,
        initial_weight: float = 0.10,
        decay_timesteps: int = 2_000_000,
        batch_size: int = 2048,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.teacher_model_path = Path(teacher_model_path)
        self.legacy_observation_size = legacy_observation_size
        self.initial_weight = max(0.0, float(initial_weight))
        self.decay_timesteps = max(1, int(decay_timesteps))
        self.batch_size = max(1, int(batch_size))
        self.teacher_model = None
        self.start_num_timesteps = 0

    def _on_training_start(self) -> None:
        from sb3_contrib import MaskablePPO

        self.teacher_model = MaskablePPO.load(
            str(self.teacher_model_path),
            device=self.model.device,
        )
        teacher_shape = tuple(self.teacher_model.observation_space.shape)
        if teacher_shape != (self.legacy_observation_size,):
            raise ValueError(
                f"Teacher observation shape is {teacher_shape}; expected "
                f"({self.legacy_observation_size},)"
            )
        if self.teacher_model.action_space.n != self.model.action_space.n:
            raise ValueError("Teacher and student action spaces do not match")
        self.teacher_model.policy.set_training_mode(False)
        self.start_num_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _current_weight(self) -> float:
        elapsed = max(0, int(self.model.num_timesteps) - self.start_num_timesteps)
        progress = min(1.0, elapsed / float(self.decay_timesteps))
        return self.initial_weight * (1.0 - progress)

    def _on_rollout_end(self) -> None:
        if self.teacher_model is None:
            return
        weight = self._current_weight()
        self.logger.record("kickstart/weight", weight)
        if weight <= 0.0:
            self.logger.record("kickstart/loss", 0.0)
            return

        import torch

        rollout_buffer = self.model.rollout_buffer
        observations = np.asarray(rollout_buffer.observations).reshape(
            (-1,) + tuple(self.model.observation_space.shape)
        )
        sample_count = min(self.batch_size, observations.shape[0])
        indices = np.random.choice(
            observations.shape[0],
            size=sample_count,
            replace=False,
        )
        student_observations = torch.as_tensor(
            observations[indices],
            dtype=torch.float32,
            device=self.model.device,
        )
        teacher_observations = student_observations[
            :, :self.legacy_observation_size
        ]

        masks: Optional[torch.Tensor] = None
        action_masks = getattr(rollout_buffer, "action_masks", None)
        if action_masks is not None:
            flat_masks = np.asarray(action_masks).reshape(
                (-1, self.model.action_space.n)
            )
            masks = torch.as_tensor(
                flat_masks[indices],
                dtype=torch.bool,
                device=self.model.device,
            )

        with torch.no_grad():
            teacher_distribution = self.teacher_model.policy.get_distribution(
                teacher_observations,
                action_masks=masks,
            )
            teacher_probabilities = teacher_distribution.distribution.probs

        self.model.policy.set_training_mode(True)
        student_distribution = self.model.policy.get_distribution(
            student_observations,
            action_masks=masks,
        )
        student_log_probabilities = torch.log(
            student_distribution.distribution.probs.clamp_min(1.0e-8)
        )
        distillation_loss = -(
            teacher_probabilities * student_log_probabilities
        ).sum(dim=1).mean()
        loss = weight * distillation_loss

        self.model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.policy.parameters(),
            self.model.max_grad_norm,
        )
        self.model.policy.optimizer.step()
        self.logger.record("kickstart/loss", float(distillation_loss.detach().cpu()))
