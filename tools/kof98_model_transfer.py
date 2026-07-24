"""Checkpoint compatibility helpers for versioned KOF98 observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kof98_observation import (
    OBSERVATION_V2_SIZE,
    OBSERVATION_V3_REPURPOSED_INDICES,
)


@dataclass(frozen=True)
class TransferReport:
    copied_tensors: tuple[str, ...]
    expanded_tensors: tuple[str, ...]
    transformed_tensors: tuple[str, ...] = ()


def transplant_policy_observation_inputs(
    source_model: Any,
    target_model: Any,
    *,
    legacy_observation_size: int,
) -> TransferReport:
    """Copy a policy into a wider observation space without changing V1 output.

    Equal-shaped tensors are copied verbatim. The policy and value input
    layers are widened by zero-initialising every new observation column and
    copying the legacy columns. Therefore a V2 observation whose appended
    features are zero produces the exact same logits and value as the V1
    checkpoint at migration time.
    """

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    if set(source_state) != set(target_state):
        missing = sorted(set(source_state).symmetric_difference(target_state))
        raise ValueError(f"Policy architectures do not match: {missing}")

    copied: list[str] = []
    expanded: list[str] = []
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        if source_tensor.shape == target_tensor.shape:
            target_state[name] = source_tensor.detach().to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            ).clone()
            copied.append(name)
            continue

        can_expand_input = (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] == legacy_observation_size
            and target_tensor.shape[1] > source_tensor.shape[1]
            and name in {
                "mlp_extractor.policy_net.0.weight",
                "mlp_extractor.value_net.0.weight",
            }
        )
        if not can_expand_input:
            raise ValueError(
                f"Cannot migrate policy tensor {name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )

        widened = target_tensor.detach().clone()
        widened.zero_()
        widened[:, :legacy_observation_size].copy_(
            source_tensor.detach().to(
                device=widened.device,
                dtype=widened.dtype,
            )
        )
        target_state[name] = widened
        expanded.append(name)

    target_model.policy.load_state_dict(target_state, strict=True)
    return TransferReport(tuple(copied), tuple(expanded))


def transplant_v2_policy_to_v3(
    source_model: Any,
    target_model: Any,
) -> TransferReport:
    """Rebind V2's 32 neutral inputs to V3 without changing policy output.

    V2 index 133 is the constant opponent-action NONE bit. V3 repurposes that
    whole one-hot (plus two zero scalars), so its learned constant contribution
    is folded into each first-layer bias before all 32 new columns are zeroed.
    """

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    if set(source_state) != set(target_state):
        missing = sorted(set(source_state).symmetric_difference(target_state))
        raise ValueError(f"Policy architectures do not match: {missing}")

    copied: list[str] = []
    transformed: list[str] = []
    input_layers = (
        (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias",
        ),
        (
            "mlp_extractor.value_net.0.weight",
            "mlp_extractor.value_net.0.bias",
        ),
    )
    input_names = {name for pair in input_layers for name in pair}
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"V2/V3 policy tensor shape changed for {name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
        if name in input_names:
            continue
        target_state[name] = source_tensor.detach().to(
            device=target_tensor.device,
            dtype=target_tensor.dtype,
        ).clone()
        copied.append(name)

    old_none_index = max(OBSERVATION_V3_REPURPOSED_INDICES)
    for weight_name, bias_name in input_layers:
        source_weight = source_state[weight_name]
        source_bias = source_state[bias_name]
        target_weight = target_state[weight_name]
        if (
            source_weight.ndim != 2
            or source_weight.shape[1] != OBSERVATION_V2_SIZE
        ):
            raise ValueError(
                f"Unexpected V2 input layer {weight_name}: "
                f"{tuple(source_weight.shape)}"
            )

        migrated_weight = source_weight.detach().to(
            device=target_weight.device,
            dtype=target_weight.dtype,
        ).clone()
        migrated_bias = source_bias.detach().to(
            device=target_state[bias_name].device,
            dtype=target_state[bias_name].dtype,
        ).clone()
        migrated_bias.add_(migrated_weight[:, old_none_index])
        migrated_weight[:, list(OBSERVATION_V3_REPURPOSED_INDICES)] = 0
        target_state[weight_name] = migrated_weight
        target_state[bias_name] = migrated_bias
        transformed.extend((weight_name, bias_name))

    target_model.policy.load_state_dict(target_state, strict=True)
    return TransferReport(
        tuple(copied),
        (),
        tuple(transformed),
    )


def assert_legacy_policy_equivalence(
    source_model: Any,
    target_model: Any,
    *,
    legacy_observation_size: int,
    target_observation_size: int,
    tolerance: float = 1.0e-6,
) -> None:
    """Fail fast if migration changed policy logits or values for V1 input."""

    import torch

    source_device = source_model.policy.device
    target_device = target_model.policy.device
    legacy = torch.linspace(
        -1.0,
        1.0,
        legacy_observation_size,
        dtype=torch.float32,
        device=source_device,
    ).unsqueeze(0)
    widened = torch.zeros(
        (1, target_observation_size),
        dtype=torch.float32,
        device=target_device,
    )
    widened[:, :legacy_observation_size] = legacy.to(target_device)

    with torch.no_grad():
        source_distribution = source_model.policy.get_distribution(legacy)
        target_distribution = target_model.policy.get_distribution(widened)
        source_probs = source_distribution.distribution.probs
        target_probs = target_distribution.distribution.probs
        source_value = source_model.policy.predict_values(legacy)
        target_value = target_model.policy.predict_values(widened)

    if not torch.allclose(
        source_probs.cpu(),
        target_probs.cpu(),
        atol=tolerance,
        rtol=0.0,
    ):
        difference = torch.max(
            torch.abs(source_probs.cpu() - target_probs.cpu())
        ).item()
        raise AssertionError(
            f"Migrated action probabilities changed (max diff {difference})"
        )
    if not torch.allclose(
        source_value.cpu(),
        target_value.cpu(),
        atol=tolerance,
        rtol=0.0,
    ):
        difference = torch.max(
            torch.abs(source_value.cpu() - target_value.cpu())
        ).item()
        raise AssertionError(f"Migrated value changed (max diff {difference})")


def assert_v2_v3_policy_equivalence(
    source_model: Any,
    target_model: Any,
    *,
    sample_count: int = 10_000,
    batch_size: int = 512,
    tolerance: float = 2.0e-4,
    seed: int = 98,
) -> None:
    """Validate V2->V3 migration on neutral V3 event features.

    The transformation is algebraically exact. Float32 GEMM is not bitwise
    associative, however: moving V2's constant NONE-column contribution into
    the bias changes accumulation order. The tolerance covers the measured
    CPU/GPU float32 GEMM error of the mature V3-C value network (about
    1.6e-4), while deterministic actions must still match for every sample.
    """

    import torch

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    max_logits_error = 0.0
    max_value_error = 0.0
    action_matches = 0
    checked = 0
    old_none_index = max(OBSERVATION_V3_REPURPOSED_INDICES)
    repurposed = list(OBSERVATION_V3_REPURPOSED_INDICES)

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    for weight_name, bias_name in (
        (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias",
        ),
        (
            "mlp_extractor.value_net.0.weight",
            "mlp_extractor.value_net.0.bias",
        ),
    ):
        source_weight = source_state[weight_name].detach().cpu()
        target_weight = target_state[weight_name].detach().cpu()
        if torch.count_nonzero(target_weight[:, repurposed]).item() != 0:
            raise AssertionError(
                f"V3 semantic columns are not zero in {weight_name}"
            )
        preserved_columns = [
            index
            for index in range(OBSERVATION_V2_SIZE)
            if index not in OBSERVATION_V3_REPURPOSED_INDICES
        ]
        if not torch.equal(
            source_weight[:, preserved_columns],
            target_weight[:, preserved_columns],
        ):
            raise AssertionError(
                f"Legacy observation weights changed in {weight_name}"
            )
        expected_bias = source_state[bias_name].detach().cpu().clone()
        expected_bias.add_(source_weight[:, old_none_index])
        if not torch.equal(
            expected_bias,
            target_state[bias_name].detach().cpu(),
        ):
            raise AssertionError(f"Constant NONE bias fold failed in {bias_name}")

    while checked < sample_count:
        current_batch = min(batch_size, sample_count - checked)
        source_observation = torch.randn(
            (current_batch, OBSERVATION_V2_SIZE),
            generator=generator,
            dtype=torch.float32,
        )
        source_observation[:, repurposed] = 0
        source_observation[:, old_none_index] = 1
        target_observation = source_observation.clone()
        target_observation[:, repurposed] = 0

        source_input = source_observation.to(source_model.policy.device)
        target_input = target_observation.to(target_model.policy.device)
        with torch.no_grad():
            source_features = source_model.policy.extract_features(source_input)
            target_features = target_model.policy.extract_features(target_input)
            source_pi, source_vf = source_model.policy.mlp_extractor(
                source_features
            )
            target_pi, target_vf = target_model.policy.mlp_extractor(
                target_features
            )
            # Compare the policy head's actual unnormalised logits. PyTorch's
            # Categorical.logits property is log-softmax-normalised and is not
            # the value emitted by SB3's action_net.
            source_logits = source_model.policy.action_net(source_pi).cpu()
            target_logits = target_model.policy.action_net(target_pi).cpu()
            source_values = source_model.policy.value_net(source_vf).cpu()
            target_values = target_model.policy.value_net(target_vf).cpu()

        max_logits_error = max(
            max_logits_error,
            float(torch.max(torch.abs(source_logits - target_logits)).item()),
        )
        max_value_error = max(
            max_value_error,
            float(torch.max(torch.abs(source_values - target_values)).item()),
        )
        action_matches += int(
            torch.sum(
                torch.argmax(source_logits, dim=1)
                == torch.argmax(target_logits, dim=1)
            ).item()
        )
        checked += current_batch

    if action_matches != sample_count:
        raise AssertionError(
            "V2/V3 deterministic actions changed: "
            f"{action_matches}/{sample_count} matched"
        )
    if max_logits_error >= tolerance:
        raise AssertionError(
            "V2/V3 logits changed "
            f"(max error {max_logits_error}, tolerance {tolerance})"
        )
    if max_value_error >= tolerance:
        raise AssertionError(
            "V2/V3 value changed "
            f"(max error {max_value_error}, tolerance {tolerance})"
        )
