"""
Model selector for weighted multi-model request distribution.

Provides the ModelSelector class that encapsulates model selection logic
for OpenAIHTTPBackend, enabling requests to be distributed across multiple
models (e.g. LoRA adapters) according to a configurable weighted distribution.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import Field, PrivateAttr

from guidellm.schemas.base import StandardBaseModel

__all__ = ["ModelSelector"]


class ModelSelector(StandardBaseModel):
    """
    Weighted multi-model selector for OpenAIHTTPBackend.

    Encapsulates model selection config and runtime state. When attached to a
    backend, each call to ``default_model()`` returns a model chosen according
    to the configured distribution rather than a fixed model name.

    Default construction (no args) means single-model passthrough — the backend
    behaves exactly as before.

    :param models: Explicit list of model names to distribute across. When
        ``None``, models are discovered dynamically from the server's
        ``/v1/models`` endpoint.
    :param n_models: Number of models to use when discovering dynamically.
        ``-1`` means use all available models. ``0`` disables multi-model mode
        (single-model passthrough). Ignored when ``models`` is provided.
    :param distribution: Per-slot weights as a list of floats, positionally
        aligned to the resolved model list. Models beyond the length of this
        list share the remaining probability equally. Omit for uniform
        distribution across all resolved models.

    Example::

        # Explicit models, weighted
        selector = ModelSelector(
            models=["base", "lora-a", "lora-b"],
            distribution=[0.80, 0.12, 0.05],
        )

        # Dynamic: all deployed models, top-heavy distribution
        selector = ModelSelector(n_models=-1, distribution=[0.70, 0.20])

        # Dynamic: top 3 models, uniform distribution
        selector = ModelSelector(n_models=3)
    """

    models: list[str] | None = Field(
        default=None,
        description=(
            "Explicit model names to distribute across. "
            "None = discover from server /v1/models."
        ),
    )
    n_models: int = Field(
        default=0,
        description=(
            "Number of models to use when discovering dynamically. "
            "-1 = all available, 0 = single-model mode (disabled)."
        ),
    )
    distribution: list[float] | None = Field(
        default=None,
        description=(
            "Per-slot weights aligned to the resolved model list. "
            "Extra models share the remaining probability evenly. "
            "Omit for uniform distribution."
        ),
    )
    type: Literal["weighted", "round_robin"] = Field(
        default="weighted",
        description=(
            "Selection strategy. 'weighted' uses random.choices with the computed "
            "weight distribution. 'round_robin' cycles through resolved models "
            "deterministically using a stateful counter."
        ),
    )
    loras_only: bool = Field(
        default=False,
        description=(
            "When True, exclude base models from the dynamically discovered model "
            "list. Base models are identified by containing a '/' in their name "
            "(HuggingFace org/model format). Has no effect when 'models' is "
            "provided explicitly."
        ),
    )

    # Runtime state — populated by resolve(), not part of the schema
    _resolved: list[str] = PrivateAttr(default_factory=list)
    _weights: list[float] = PrivateAttr(default_factory=list)
    _counter: int = PrivateAttr(default=0)

    @property
    def is_multi(self) -> bool:
        """True when multi-model mode is active."""
        return self.models is not None or self.n_models != 0

    def resolve(self, available: list[str]) -> None:
        """
        Populate the resolved model list and compute weights.

        Called once per ``process_startup()`` after the server's available
        models are known.

        :param available: Models returned by the server's ``/v1/models``
            endpoint. Used only when ``models`` is ``None``.
        """
        pool = available
        if self.loras_only:
            pool = [m for m in available if "/" not in m]

        if self.models is not None:
            self._resolved = list(self.models)
        elif self.n_models == -1:
            self._resolved = list(pool)
        else:
            self._resolved = list(pool[: self.n_models])

        self._weights = _compute_weights(self._resolved, self.distribution or [])

    def select(self) -> str:
        """
        Pick a model according to the configured selection strategy.

        ``weighted``: draws randomly using the computed weight distribution.
        ``round_robin``: cycles through resolved models in order using a
        stateful counter (deterministic, evenly distributed over time).

        :return: A model name, or ``""`` if no models have been resolved yet.
        """
        if not self._resolved:
            return ""
        if self.type == "round_robin":
            model = self._resolved[self._counter % len(self._resolved)]
            self._counter += 1
            return model
        return random.choices(self._resolved, weights=self._weights, k=1)[0]

    @property
    def resolved_models(self) -> list[str]:
        """The resolved model list after ``resolve()`` has been called."""
        return list(self._resolved)

    @property
    def resolved_weights(self) -> list[float]:
        """The computed weight list after ``resolve()`` has been called."""
        return list(self._weights)


def _compute_weights(models: list[str], distribution: list[float]) -> list[float]:
    """
    Build a full weight list from a partial distribution spec.

    Slots covered by ``distribution`` use those weights directly. Any remaining
    slots share the leftover probability equally (uniform split).

    :param models: The resolved model list.
    :param distribution: Partial per-slot weights.
    :return: A weight list of the same length as ``models``.
    """
    n = len(models)
    if not n:
        return []
    explicit = distribution[:n]
    assigned = sum(explicit)
    n_remaining = n - len(explicit)
    remainder_each = max(0.0, 1.0 - assigned) / n_remaining if n_remaining else 0.0
    return list(explicit) + [remainder_each] * n_remaining
