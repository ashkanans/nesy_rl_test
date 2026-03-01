from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class TokenField:
    name: str
    index: int
    description: str
    expected_cardinality: int | None = None


@dataclass(frozen=True)
class TokenSchemaDefinition:
    schema_id: str
    env_name: str
    width: int
    dtype: str
    fields: tuple[TokenField, ...]

    def field_index(self, name: str) -> int:
        for field in self.fields:
            if field.name == name:
                return int(field.index)
        raise KeyError(f"Unknown field '{name}' for schema '{self.schema_id}'")

    def field_names(self) -> list[str]:
        return [field.name for field in self.fields]


SCHEMA_REGISTRY: Dict[str, TokenSchemaDefinition] = {
    "cb_v1": TokenSchemaDefinition(
        schema_id="cb_v1",
        env_name="cb",
        width=4,
        dtype="int64",
        fields=(
            TokenField("state", 0, "Discrete environment state token."),
            TokenField("action", 1, "Discrete action token."),
            TokenField("reward", 2, "Reward-token slot (currently placeholder=0).", 1),
            TokenField("aux", 3, "Auxiliary slot (currently placeholder=0).", 1),
        ),
    ),
    "nrm_nav_v1": TokenSchemaDefinition(
        schema_id="nrm_nav_v1",
        env_name="nrm_nav",
        width=4,
        dtype="int64",
        fields=(
            TokenField("state", 0, "Discrete environment state token."),
            TokenField("action", 1, "Discrete action token."),
            TokenField("reward", 2, "Reward-token slot (currently placeholder=0).", 1),
            TokenField("safety_cost", 3, "Binary safety cost token.", 2),
        ),
    ),
    "frozenlake_v1": TokenSchemaDefinition(
        schema_id="frozenlake_v1",
        env_name="frozenlake",
        width=4,
        dtype="int64",
        fields=(
            TokenField("state", 0, "Discrete environment state token."),
            TokenField("action", 1, "Discrete action token."),
            TokenField("reward", 2, "Reward-token slot (currently placeholder=0).", 1),
            TokenField("safety_cost", 3, "Binary hole-cost token.", 2),
        ),
    ),
    "antmaze_v1": TokenSchemaDefinition(
        schema_id="antmaze_v1",
        env_name="antmaze",
        width=4,
        dtype="int64",
        fields=(
            TokenField("state", 0, "Discretized state token."),
            TokenField("action", 1, "Discretized action token."),
            TokenField("reward", 2, "Reward-token slot (placeholder by default).", 1),
            TokenField("safety_cost", 3, "Binary safety cost token.", 2),
        ),
    ),
    "dsrl_v1": TokenSchemaDefinition(
        schema_id="dsrl_v1",
        env_name="dsrl",
        width=4,
        dtype="int64",
        fields=(
            TokenField("state", 0, "Discretized state token."),
            TokenField("action", 1, "Discretized action token."),
            TokenField("reward", 2, "Goal-indicator reward token (0/1).", 2),
            TokenField("safety_cost", 3, "Binary safety cost token.", 2),
        ),
    ),
}

ENV_TO_SCHEMA_ID = {
    "cb": "cb_v1",
    "nrm_nav": "nrm_nav_v1",
    "frozenlake": "frozenlake_v1",
    "antmaze": "antmaze_v1",
    "dsrl": "dsrl_v1",
}


def get_schema_for_env(env_name: str) -> TokenSchemaDefinition:
    if env_name not in ENV_TO_SCHEMA_ID:
        raise ValueError(f"Unsupported env '{env_name}' for token schema.")
    return SCHEMA_REGISTRY[ENV_TO_SCHEMA_ID[env_name]]


def get_num_bins_per_dim(schema: TokenSchemaDefinition, observation_bins: int, action_bins: int) -> list[int]:
    bins = [1] * schema.width
    bins[schema.field_index("state")] = int(observation_bins)
    bins[schema.field_index("action")] = int(action_bins)
    for field in schema.fields:
        if field.expected_cardinality is not None:
            bins[field.index] = int(field.expected_cardinality)
    return bins


def get_end_token_id(schema: TokenSchemaDefinition, observation_bins: int, action_bins: int) -> int:
    return max(get_num_bins_per_dim(schema, observation_bins, action_bins))


def build_end_row(schema: TokenSchemaDefinition, end_token_id: int) -> np.ndarray:
    row = np.zeros(schema.width, dtype=np.int64)
    row[schema.field_index("state")] = int(end_token_id)
    return row


def make_transition_row(
    schema: TokenSchemaDefinition,
    state: int,
    action: int,
    reward_token: int = 0,
    safety_cost: int = 0,
) -> np.ndarray:
    row = np.zeros(schema.width, dtype=np.int64)
    row[schema.field_index("state")] = int(state)
    row[schema.field_index("action")] = int(action)
    if "reward" in schema.field_names():
        row[schema.field_index("reward")] = int(reward_token)
    if "safety_cost" in schema.field_names():
        row[schema.field_index("safety_cost")] = int(safety_cost)
    return row


def _validate_cardinality(row: np.ndarray, field: TokenField) -> None:
    if field.expected_cardinality is None:
        return
    value = int(row[field.index])
    if value < 0 or value >= int(field.expected_cardinality):
        raise ValueError(
            f"Field '{field.name}' value {value} out of range [0, {field.expected_cardinality - 1}]"
        )


def validate_transition_row(
    row: np.ndarray,
    schema: TokenSchemaDefinition,
    observation_space_n: int | None = None,
    action_space_n: int | None = None,
) -> None:
    if row.shape != (schema.width,):
        raise ValueError(f"Expected row shape ({schema.width},), got {row.shape}")
    if not np.issubdtype(row.dtype, np.integer):
        raise ValueError(f"Expected integer row dtype for schema '{schema.schema_id}', got {row.dtype}")

    sidx = schema.field_index("state")
    aidx = schema.field_index("action")
    state_val = int(row[sidx])
    action_val = int(row[aidx])

    if observation_space_n is not None and (state_val < 0 or state_val >= int(observation_space_n)):
        raise ValueError(
            f"State token {state_val} out of range [0, {int(observation_space_n) - 1}]"
        )
    if action_space_n is not None and (action_val < 0 or action_val >= int(action_space_n)):
        raise ValueError(
            f"Action token {action_val} out of range [0, {int(action_space_n) - 1}]"
        )

    for field in schema.fields:
        _validate_cardinality(row, field)


def validate_episode_tokens(
    episode_tokens: np.ndarray,
    schema: TokenSchemaDefinition,
    end_token_id: int,
    observation_space_n: int | None = None,
    action_space_n: int | None = None,
) -> None:
    if episode_tokens.ndim != 2:
        raise ValueError("episode_tokens must be 2D.")
    if episode_tokens.shape[1] != schema.width:
        raise ValueError(
            f"episode_tokens width mismatch for schema '{schema.schema_id}': "
            f"expected {schema.width}, got {episode_tokens.shape[1]}"
        )
    if not np.issubdtype(episode_tokens.dtype, np.integer):
        raise ValueError("episode_tokens must use integer dtype.")

    state_idx = schema.field_index("state")
    end_rows = np.where(episode_tokens[:, state_idx] == int(end_token_id))[0]
    if len(end_rows) != 1:
        raise ValueError(f"Expected exactly one END row, found {len(end_rows)}")
    if int(end_rows[0]) != int(episode_tokens.shape[0] - 1):
        raise ValueError("END row must be the final row of each full episode.")

    for row in episode_tokens[:-1]:
        validate_transition_row(
            row=row,
            schema=schema,
            observation_space_n=observation_space_n,
            action_space_n=action_space_n,
        )

    expected_end = build_end_row(schema=schema, end_token_id=end_token_id)
    if not np.array_equal(episode_tokens[-1], expected_end):
        raise ValueError(
            f"END row mismatch for schema '{schema.schema_id}'. "
            f"Expected {expected_end.tolist()}, got {episode_tokens[-1].tolist()}"
        )


def summarize_schema(schema: TokenSchemaDefinition) -> dict:
    return {
        "schema_id": schema.schema_id,
        "env_name": schema.env_name,
        "width": schema.width,
        "dtype": schema.dtype,
        "fields": [
            {
                "name": f.name,
                "index": int(f.index),
                "description": f.description,
                "expected_cardinality": f.expected_cardinality,
            }
            for f in schema.fields
        ],
    }


def iter_supported_envs() -> Iterable[str]:
    return tuple(ENV_TO_SCHEMA_ID.keys())
