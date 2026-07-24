"""Auxiliary Oracle imitation for physical-mask tactical curricula."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class OracleCurriculumCallback(BaseCallback):
    """Teach rare tactical actions without narrowing the environment mask.

    Recipe environments publish the next deterministic Oracle action for the
    observation that the policy just consumed.  We keep the normal PPO action
    and full physical mask, then apply a small supervised policy update after
    each rollout.  The weight decays to zero, leaving PPO to decide when the
    demonstrated tactic is useful in unrestricted fights.
    """

    def __init__(
        self,
        *,
        initial_weight: float = 0.10,
        decay_timesteps: int = 600_000,
        batch_size: int = 2048,
        updates_per_rollout: int = 4,
        replay_capacity: int = 50_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.initial_weight = max(0.0, float(initial_weight))
        self.decay_timesteps = max(1, int(decay_timesteps))
        self.batch_size = max(1, int(batch_size))
        self.updates_per_rollout = max(1, int(updates_per_rollout))
        self.observations: Deque[np.ndarray] = deque(maxlen=replay_capacity)
        self.action_masks: Deque[np.ndarray] = deque(maxlen=replay_capacity)
        self.actions: Deque[int] = deque(maxlen=replay_capacity)
        self.tasks: Deque[str] = deque(maxlen=replay_capacity)
        heldout_capacity = max(512, replay_capacity // 10)
        self.heldout_observations: Deque[np.ndarray] = deque(maxlen=heldout_capacity)
        self.heldout_action_masks: Deque[np.ndarray] = deque(maxlen=heldout_capacity)
        self.heldout_actions: Deque[int] = deque(maxlen=heldout_capacity)
        self.heldout_tasks: Deque[str] = deque(maxlen=heldout_capacity)
        self.task_label_seen: dict[str, int] = {}
        self.start_num_timesteps = 0
        self.last_loss = 0.0
        self.last_accuracy = 0.0
        self.last_sample_count = 0
        self.last_heldout_accuracy = 0.0
        self.last_heldout_task_accuracy: dict[str, float] = {}
        self.rollout_label_count = 0
        self.rollout_task_counts: dict[str, int] = {}

    def _on_training_start(self) -> None:
        self.start_num_timesteps = int(self.model.num_timesteps)

    def _current_weight(self) -> float:
        elapsed = max(0, int(self.model.num_timesteps) - self.start_num_timesteps)
        progress = min(1.0, elapsed / float(self.decay_timesteps))
        return self.initial_weight * (1.0 - progress)

    def _on_rollout_start(self) -> None:
        # The previous rollout has already been consumed by PPO at this point.
        # Updating here keeps the next rollout's old log-probabilities aligned
        # with the policy that actually generates its actions.
        self._apply_imitation_updates()
        self.rollout_label_count = 0
        self.rollout_task_counts = {}

    def _on_step(self) -> bool:
        obs_tensor = self.locals.get("obs_tensor")
        masks = self.locals.get("action_masks")
        infos = self.locals.get("infos")
        if obs_tensor is None or masks is None or infos is None:
            return True

        observations = obs_tensor.detach().cpu().numpy()
        masks = np.asarray(masks, dtype=bool)
        for index, info in enumerate(infos):
            action_id = int(info.get("curriculum_oracle_action_before", -1))
            if (
                action_id < 0
                or action_id >= self.model.action_space.n
                or not masks[index, action_id]
                or not bool(
                    info.get("curriculum_oracle_trajectory_valid_before", 0.0)
                )
            ):
                continue
            task_name = str(
                info.get("curriculum_lesson")
                or info.get("curriculum_task", "unknown")
            )
            seen = self.task_label_seen.get(task_name, 0) + 1
            self.task_label_seen[task_name] = seen
            observation = np.array(observations[index], copy=True)
            action_mask = np.array(masks[index], copy=True)
            if seen % 10 == 0:
                self.heldout_observations.append(observation)
                self.heldout_action_masks.append(action_mask)
                self.heldout_actions.append(action_id)
                self.heldout_tasks.append(task_name)
            else:
                self.observations.append(observation)
                self.action_masks.append(action_mask)
                self.actions.append(action_id)
                self.tasks.append(task_name)
            self.rollout_label_count += 1
            self.rollout_task_counts[task_name] = (
                self.rollout_task_counts.get(task_name, 0) + 1
            )
        return True

    def _on_rollout_end(self) -> None:
        weight = self._current_weight()
        self.logger.record("oracle_teacher/weight", weight)
        self.logger.record("oracle_teacher/replay_size", len(self.actions))
        self.logger.record("oracle_teacher/loss", self.last_loss)
        self.logger.record("oracle_teacher/accuracy", self.last_accuracy)
        self.logger.record(
            "oracle_teacher/heldout_accuracy",
            self.last_heldout_accuracy,
        )
        self.logger.record("oracle_teacher/samples", self.last_sample_count)
        self.logger.record(
            "oracle_teacher/labels_collected",
            self.rollout_label_count,
            exclude="stdout",
        )
        task_names = sorted(
            set(self.task_label_seen)
            | {"guard_only", "block_counter", "anti_air", "approach", "hit_confirm"}
        )
        for task_name in task_names:
            self.logger.record(
                f"oracle_teacher/{task_name}_labels_collected",
                self.rollout_task_counts.get(task_name, 0),
                exclude="stdout",
            )
            self.logger.record(
                f"oracle_teacher/{task_name}_heldout_accuracy",
                self.last_heldout_task_accuracy.get(task_name, 0.0),
                exclude="stdout",
            )

    def _on_training_end(self) -> None:
        # Consume examples from the final rollout before the caller saves the
        # final model.  No PPO rollout follows, so this cannot invalidate an
        # on-policy buffer.
        self._apply_imitation_updates()

    def _apply_imitation_updates(self) -> None:
        weight = self._current_weight()
        self.last_loss = 0.0
        self.last_accuracy = 0.0
        self.last_sample_count = 0
        if weight <= 0.0 or not self.actions:
            self._evaluate_heldout()
            return

        import torch

        sample_count = min(self.batch_size, len(self.actions))
        observation_items = list(self.observations)
        mask_items = list(self.action_masks)
        action_items = list(self.actions)
        task_items = list(self.tasks)
        losses: list[float] = []
        accuracies: list[float] = []
        for _ in range(self.updates_per_rollout):
            indices = self._balanced_sample_indices(task_items, sample_count)
            observations = torch.as_tensor(
                np.stack([observation_items[index] for index in indices]),
                dtype=torch.float32,
                device=self.model.device,
            )
            masks = torch.as_tensor(
                np.stack([mask_items[index] for index in indices]),
                dtype=torch.bool,
                device=self.model.device,
            )
            actions = torch.as_tensor(
                np.asarray([action_items[index] for index in indices]),
                dtype=torch.long,
                device=self.model.device,
            )

            self.model.policy.set_training_mode(True)
            distribution = self.model.policy.get_distribution(
                observations,
                action_masks=masks,
            )
            imitation_loss = -distribution.log_prob(actions).mean()
            loss = weight * imitation_loss

            self.model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.policy.parameters(),
                self.model.max_grad_norm,
            )
            self.model.policy.optimizer.step()

            with torch.no_grad():
                predicted = distribution.distribution.probs.argmax(dim=1)
                accuracies.append(
                    float((predicted == actions).float().mean().cpu())
                )
            losses.append(float(imitation_loss.detach().cpu()))

        self.last_loss = float(np.mean(losses))
        self.last_accuracy = float(np.mean(accuracies))
        self.last_sample_count = sample_count
        self._evaluate_heldout()

    @staticmethod
    def _balanced_sample_indices(
        task_items: list[str],
        sample_count: int,
    ) -> np.ndarray:
        groups: dict[str, list[int]] = {}
        for index, task_name in enumerate(task_items):
            groups.setdefault(task_name, []).append(index)
        if not groups:
            return np.empty(0, dtype=np.int64)

        per_task = max(1, sample_count // len(groups))
        selected: list[int] = []
        for indices in groups.values():
            take = min(per_task, len(indices))
            selected.extend(
                np.random.choice(indices, size=take, replace=False).tolist()
            )
        if len(selected) < sample_count:
            remaining = list(set(range(len(task_items))) - set(selected))
            take = min(sample_count - len(selected), len(remaining))
            if take:
                selected.extend(
                    np.random.choice(remaining, size=take, replace=False).tolist()
                )
        return np.asarray(selected, dtype=np.int64)

    def _evaluate_heldout(self) -> None:
        self.last_heldout_accuracy = 0.0
        self.last_heldout_task_accuracy = {}
        if not self.heldout_actions:
            return

        import torch

        observations = torch.as_tensor(
            np.stack(list(self.heldout_observations)),
            dtype=torch.float32,
            device=self.model.device,
        )
        masks = torch.as_tensor(
            np.stack(list(self.heldout_action_masks)),
            dtype=torch.bool,
            device=self.model.device,
        )
        actions = np.asarray(list(self.heldout_actions), dtype=np.int64)
        tasks = list(self.heldout_tasks)
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                observations,
                action_masks=masks,
            )
            predicted = distribution.distribution.probs.argmax(dim=1).cpu().numpy()
        correct = predicted == actions
        self.last_heldout_accuracy = float(correct.mean())
        for task_name in set(tasks):
            task_indices = np.asarray(
                [index for index, item in enumerate(tasks) if item == task_name],
                dtype=np.int64,
            )
            self.last_heldout_task_accuracy[task_name] = float(
                correct[task_indices].mean()
            )
