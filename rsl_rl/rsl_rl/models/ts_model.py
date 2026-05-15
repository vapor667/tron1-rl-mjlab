# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-student model with dual encoders for privileged training and deployment."""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation


class Encoder(nn.Module):
    """Encoder network with orthogonal initialization and L2-normalized output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        latent_dim: int = 32,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        activation_mod = resolve_nn_activation(activation)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))  # type: ignore[arg-type]
        layers.append(activation_mod)

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], latent_dim))
                nn.init.orthogonal_(layers[-1].weight, 0.01)  # type: ignore[arg-type]
                nn.init.constant_(layers[-1].bias, 0.0)  # type: ignore[arg-type]
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))  # type: ignore[arg-type]
                nn.init.constant_(layers[-1].bias, 0.0)  # type: ignore[arg-type]
                layers.append(activation_mod)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return nn.functional.normalize(latent, p=2, dim=-1)


class TSModel(nn.Module):
    """Teacher-student model with dual encoders for privileged training and student deployment.

    During teacher mode (training), the actor uses the privileged encoder (fed privileged
    critic observations) to compute a latent representation. During student mode (deployment),
    the proprioceptive encoder (fed observation history OR raw actor obs) is used instead.

    The obs TensorDict is expected to contain the following keys (configurable):
    - ``raw_obs_key`` (default ``"policy"``): actor observations.
    - ``history_obs_key`` (default ``"history"``): obs history for proprioceptive encoder.
    - ``privileged_obs_key`` (default ``"critic"``): privileged observations for privileged encoder.
    - ``commands_key`` (default ``"commands"``): command/goal vector.

    Compatibility:
    If ``history_obs_key=None``, the student/proprioceptive encoder falls back to encoding
    ``raw_obs_key`` instead of history.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,  # accepted for interface compatibility, not used
        obs_set: str,  # accepted for interface compatibility, not used
        output_dim: int,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        encoder_latent_dim: int = 32,
        raw_obs_key: str = "actor",
        history_obs_key: str | None = "history",
        privileged_obs_key: str = "critic",
        commands_key: str | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        distribution_cfg: dict | None = None,
        **kwargs,  # absorb unused RslRlModelCfg fields
    ) -> None:
        super().__init__()

        self.raw_obs_key = raw_obs_key
        self.history_obs_key = history_obs_key
        self.privileged_obs_key = privileged_obs_key
        self.commands_key = commands_key
        self._student_mode: bool = False
        
        # 核心兼容逻辑：如果有 history 就用 history，没有就 fallback 到 raw_obs (actor)
        self.student_obs_key = history_obs_key if history_obs_key is not None else raw_obs_key

        # Compute input dimensions from the obs TensorDict
        raw_obs_dim: int = obs[raw_obs_key].shape[-1]
        privileged_obs_dim: int = obs[privileged_obs_key].shape[-1]
        commands_dim: int = obs[commands_key].shape[-1] if commands_key is not None else 0
        student_obs_dim: int = obs[self.student_obs_key].shape[-1]

        # Store for export helpers
        self._raw_obs_dim = raw_obs_dim
        self._commands_dim = commands_dim
        self._encoder_latent_dim = encoder_latent_dim
        self._student_obs_dim = student_obs_dim

        # Privileged encoder (teacher) — always created
        self.privileged_encoder = Encoder(
            input_dim=privileged_obs_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=encoder_latent_dim,
            activation=activation,
        )

        # Proprioceptive encoder (student) — always created, dynamically sizes to history or actor obs
        self.proprioceptive_encoder = Encoder(
            input_dim=student_obs_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=encoder_latent_dim,
            activation=activation,
        )

        # Distribution (for stochastic actor; None for deterministic critic)
        mlp_output_dim: int
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim  # type: ignore
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP head: input is (encoder_latent_dim + raw_obs_dim + commands_dim)
        mlp_input_dim = encoder_latent_dim + raw_obs_dim + commands_dim
        self.mlp = MLP(mlp_input_dim, mlp_output_dim, hidden_dims, activation)

        # Distribution-specific MLP weight init
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        print(f"PrivilegedEncoder: {self.privileged_encoder}")
        print(f"ProprioceptiveEncoder (encoding '{self.student_obs_key}'): {self.proprioceptive_encoder}")
        print(f"MLP: {self.mlp}")

    # ------------------------------------------------------------------ #
    # MLPModel-compatible interface                                      #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if self._student_mode:
            encoder_latent = self.proprioceptive_encoder(obs[self.student_obs_key])
        else:
            encoder_latent = self.privileged_encoder(obs[self.privileged_obs_key])
            
        parts = [encoder_latent, obs[self.raw_obs_key]]
        if self.commands_key is not None:
            parts.append(obs[self.commands_key])
        return torch.cat(parts, dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore

    # ------------------------------------------------------------------ #
    # Teacher-student specific methods                                   #
    # ------------------------------------------------------------------ #

    def use_student_mode(self) -> None:
        """Switch to student (proprioceptive encoder) mode."""
        self._student_mode = True

    def use_teacher_mode(self) -> None:
        """Switch to teacher (privileged encoder) mode."""
        self._student_mode = False

    def proprio_encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode student observations (history or actor obs) with the proprioceptive encoder."""
        return self.proprioceptive_encoder(obs[self.student_obs_key])

    def privileged_encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode privileged obs with the privileged encoder."""
        return self.privileged_encoder(obs[self.privileged_obs_key])

    def act_inference_student(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic forward pass using the proprioceptive (student) encoder."""
        prev = self._student_mode
        self._student_mode = True
        result = self.forward(obs, stochastic_output=False)
        self._student_mode = prev
        return result

    def act_inference_teacher(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic forward pass using the privileged (teacher) encoder."""
        prev = self._student_mode
        self._student_mode = False
        result = self.forward(obs, stochastic_output=False)
        self._student_mode = prev
        return result

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #

    def as_jit(self) -> nn.Module:
        """Return a JIT-scriptable student inference model."""
        return _TorchTSStudentModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-exportable student inference model."""
        return _OnnxTSStudentModel(self, verbose)


class _TorchTSStudentModel(nn.Module):
    """JIT-exportable student inference model.

    Takes three separate tensors (obs_history/student_obs, raw_obs, commands) and returns actions.
    """

    def __init__(self, model: TSModel) -> None:
        super().__init__()
        self.proprio_encoder = copy.deepcopy(model.proprioceptive_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output: nn.Module = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        obs_history: torch.Tensor,  # Note: this represents raw_obs if history is disabled
        raw_obs: torch.Tensor,
        commands: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.proprio_encoder(obs_history)
        x = torch.cat([latent, raw_obs, commands], dim=-1)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxTSStudentModel(nn.Module):
    """ONNX-exportable student inference model."""

    is_recurrent: bool = False

    def __init__(self, model: TSModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.proprio_encoder = copy.deepcopy(model.proprioceptive_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output: nn.Module = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
            
        self._student_obs_size = model._student_obs_dim
        self._raw_obs_size = model._raw_obs_dim
        self._commands_size = model._commands_dim

    def forward(
        self,
        obs_history: torch.Tensor,  # Note: this represents raw_obs if history is disabled
        raw_obs: torch.Tensor,
        commands: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.proprio_encoder(obs_history)
        x = torch.cat([latent, raw_obs, commands], dim=-1)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self._student_obs_size),
            torch.zeros(1, self._raw_obs_size),
            torch.zeros(1, self._commands_size),
        )

    @property
    def input_names(self) -> list[str]:
        # Keep "obs_history" name to not break downstream C++ deployment code,
        # but conceptually it represents whatever the student encoder takes.
        return ["obs_history", "obs", "commands"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]