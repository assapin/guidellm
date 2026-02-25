"""
Unit tests for ModelSelector and its integration with OpenAIHTTPBackend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from guidellm.backends.openai.http import OpenAIHTTPBackend
from guidellm.backends.openai.selector import ModelSelector
from tests.unit.testing_utils import async_timeout

AVAILABLE = ["model-a", "model-b", "model-c"]


class TestModelSelector:
    """Test cases for ModelSelector."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(
        params=[
            {"models": ["a", "b", "c"]},
            {"n_models": -1},
            {"n_models": 2},
            {"models": ["x", "y"], "distribution": [0.7, 0.3]},
        ],
        ids=["explicit_models", "n_models_all", "n_models_slice", "weighted"],
    )
    def valid_instances(self, request):
        """Fixture providing valid ModelSelector instances."""
        return ModelSelector(**request.param), request.param

    # ------------------------------------------------------------------
    # Smoke tests
    # ------------------------------------------------------------------

    @pytest.mark.smoke
    def test_class_signatures(self):
        """Test ModelSelector has expected attributes and methods."""
        assert hasattr(ModelSelector, "resolve")
        assert hasattr(ModelSelector, "select")
        assert hasattr(ModelSelector, "is_multi")
        assert hasattr(ModelSelector, "resolved_models")
        assert hasattr(ModelSelector, "resolved_weights")

    @pytest.mark.smoke
    def test_initialization(self, valid_instances):
        """Test ModelSelector initialization with valid parameters."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ModelSelector)
        if "models" in constructor_args:
            assert instance.models == constructor_args["models"]
        if "n_models" in constructor_args:
            assert instance.n_models == constructor_args["n_models"]
        if "distribution" in constructor_args:
            assert instance.distribution == constructor_args["distribution"]

    @pytest.mark.smoke
    def test_initialization_defaults(self):
        """Test ModelSelector default field values."""
        sel = ModelSelector()
        assert sel.models is None
        assert sel.n_models == 0
        assert sel.distribution is None
        assert sel.resolved_models == []
        assert sel.resolved_weights == []

    @pytest.mark.smoke
    def test_is_multi_explicit_models(self):
        """Test is_multi is True when explicit models are provided."""
        assert ModelSelector(models=["a", "b"]).is_multi is True

    @pytest.mark.smoke
    def test_is_multi_n_models_nonzero(self):
        """Test is_multi is True when n_models is non-zero."""
        assert ModelSelector(n_models=2).is_multi is True
        assert ModelSelector(n_models=-1).is_multi is True

    @pytest.mark.smoke
    def test_is_multi_default_false(self):
        """Test is_multi is False with default construction."""
        assert ModelSelector().is_multi is False
        assert ModelSelector(n_models=0).is_multi is False

    # ------------------------------------------------------------------
    # Sanity tests — resolve()
    # ------------------------------------------------------------------

    @pytest.mark.sanity
    def test_resolve_explicit_models_ignores_available(self):
        """Test that explicit models list takes precedence over available."""
        sel = ModelSelector(models=["x", "y"])
        sel.resolve(AVAILABLE)
        assert sel.resolved_models == ["x", "y"]

    @pytest.mark.sanity
    def test_resolve_n_models_minus_one_uses_all(self):
        """Test n_models=-1 resolves to all available models."""
        sel = ModelSelector(n_models=-1)
        sel.resolve(AVAILABLE)
        assert sel.resolved_models == AVAILABLE

    @pytest.mark.sanity
    def test_resolve_n_models_slices_available(self):
        """Test n_models slices the available list."""
        sel = ModelSelector(n_models=2)
        sel.resolve(AVAILABLE)
        assert sel.resolved_models == ["model-a", "model-b"]

    @pytest.mark.sanity
    def test_resolve_n_models_larger_than_available_uses_all(self):
        """Test n_models larger than available clips to all available."""
        sel = ModelSelector(n_models=100)
        sel.resolve(AVAILABLE)
        assert sel.resolved_models == AVAILABLE

    @pytest.mark.sanity
    def test_resolve_empty_available(self):
        """Test resolve with empty available list produces empty state."""
        sel = ModelSelector(n_models=-1)
        sel.resolve([])
        assert sel.resolved_models == []
        assert sel.resolved_weights == []

    # ------------------------------------------------------------------
    # Sanity tests — weight computation
    # ------------------------------------------------------------------

    @pytest.mark.sanity
    def test_weights_uniform_when_no_distribution(self):
        """Test uniform weights when no distribution is specified."""
        sel = ModelSelector(models=["a", "b", "c"])
        sel.resolve([])
        weights = sel.resolved_weights
        assert len(weights) == 3
        assert all(abs(w - weights[0]) < 1e-9 for w in weights)

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("distribution", "expected"),
        [
            ([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]),
            ([0.6], [0.6, 0.2, 0.2]),
            ([1.1, 0.0], [1.1, 0.0, 0.0]),
        ],
        ids=["full_explicit", "partial_fills_remainder", "overflow_clamps_to_zero"],
    )
    def test_weights_distribution_variants(self, distribution, expected):
        """Test weight computation for various distribution inputs."""
        sel = ModelSelector(models=["a", "b", "c"], distribution=distribution)
        sel.resolve([])
        assert sel.resolved_weights == pytest.approx(expected)

    @pytest.mark.sanity
    def test_weights_distribution_longer_than_models_truncated(self):
        """Test that extra distribution values beyond model count are ignored."""
        sel = ModelSelector(models=["a", "b"], distribution=[0.5, 0.3, 0.99])
        sel.resolve([])
        assert len(sel.resolved_weights) == 2
        assert sel.resolved_weights == pytest.approx([0.5, 0.3])

    @pytest.mark.sanity
    def test_weights_single_model(self):
        """Test weight for a single model."""
        sel = ModelSelector(models=["only"])
        sel.resolve([])
        assert sel.resolved_weights == pytest.approx([1.0])

    # ------------------------------------------------------------------
    # Sanity tests — select()
    # ------------------------------------------------------------------

    @pytest.mark.sanity
    def test_select_before_resolve_returns_empty_string(self):
        """Test select returns empty string when no models have been resolved."""
        sel = ModelSelector(models=["a", "b"])
        assert sel.select() == ""

    @pytest.mark.sanity
    def test_select_returns_model_from_resolved_list(self):
        """Test select returns a model from the resolved list."""
        sel = ModelSelector(models=["a", "b", "c"])
        sel.resolve([])
        for _ in range(20):
            assert sel.select() in ["a", "b", "c"]

    @pytest.mark.sanity
    def test_select_single_model_always_returns_it(self):
        """Test select always returns the only resolved model."""
        sel = ModelSelector(models=["only"])
        sel.resolve([])
        assert all(sel.select() == "only" for _ in range(10))

    # ------------------------------------------------------------------
    # Regression tests
    # ------------------------------------------------------------------

    @pytest.mark.regression
    def test_select_heavily_weighted_model_dominates(self):
        """Test that a heavily weighted model is selected far more often."""
        sel = ModelSelector(models=["dominant", "rare"], distribution=[0.999, 0.001])
        sel.resolve([])
        results = [sel.select() for _ in range(200)]
        assert results.count("dominant") > results.count("rare")

    @pytest.mark.regression
    def test_model_validate_from_dict(self):
        """Test ModelSelector can be constructed via model_validate from dict."""
        sel = ModelSelector.model_validate(
            {"models": ["a", "b"], "distribution": [0.7, 0.3]}
        )
        sel.resolve([])
        assert sel.resolved_models == ["a", "b"]
        assert sel.resolved_weights == pytest.approx([0.7, 0.3])

    @pytest.mark.regression
    def test_resolve_called_twice_overwrites_state(self):
        """Test that calling resolve a second time replaces prior state."""
        sel = ModelSelector(n_models=-1)
        sel.resolve(["first"])
        assert sel.resolved_models == ["first"]
        sel.resolve(["second-a", "second-b"])
        assert sel.resolved_models == ["second-a", "second-b"]


class TestOpenAIHTTPBackendModelSelector:
    """Test cases for ModelSelector integration with OpenAIHTTPBackend."""

    @pytest.mark.smoke
    def test_model_selector_accepts_dict(self):
        """Test backend accepts model_selector as a plain dict."""
        backend = OpenAIHTTPBackend(
            target="http://test",
            model_selector={"models": ["a", "b"], "distribution": [0.6, 0.4]},
        )
        assert isinstance(backend.model_selector, ModelSelector)

    @pytest.mark.smoke
    def test_model_selector_accepts_instance(self):
        """Test backend accepts model_selector as a ModelSelector instance."""
        sel = ModelSelector(models=["x", "y"])
        backend = OpenAIHTTPBackend(target="http://test", model_selector=sel)
        assert backend.model_selector is sel

    @pytest.mark.smoke
    def test_model_selector_none_by_default(self):
        """Test model_selector is None when not provided."""
        backend = OpenAIHTTPBackend(target="http://test", model="my-model")
        assert backend.model_selector is None

    @pytest.mark.sanity
    def test_multi_mode_clears_cached_model(self):
        """Test that enabling multi-model mode clears the cached model name."""
        backend = OpenAIHTTPBackend(
            target="http://test",
            model="base-model",
            model_selector={"models": ["a", "b"]},
        )
        assert backend.model == ""

    @pytest.mark.sanity
    def test_single_model_mode_preserves_model(self):
        """Test that no selector leaves the model field intact."""
        backend = OpenAIHTTPBackend(target="http://test", model="my-model")
        assert backend.model == "my-model"

    @pytest.mark.sanity
    def test_info_includes_selector_in_multi_mode(self):
        """Test backend.info exposes selector state when in multi-model mode."""
        backend = OpenAIHTTPBackend(
            target="http://test",
            model_selector={"models": ["a", "b"], "distribution": [0.7, 0.3]},
        )
        backend.model_selector.resolve([])
        info = backend.info
        assert "model_selector" in info
        assert info["model_selector"]["models"] == ["a", "b"]
        assert info["model_selector"]["weights"] == pytest.approx([0.7, 0.3])

    @pytest.mark.sanity
    def test_info_excludes_selector_without_multi_mode(self):
        """Test backend.info omits model_selector when not in multi-model mode."""
        backend = OpenAIHTTPBackend(target="http://test", model="my-model")
        assert "model_selector" not in backend.info

    @pytest.mark.smoke
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_default_model_delegates_to_selector(self):
        """Test default_model returns a model from the selector when configured."""
        sel = ModelSelector(models=["a", "b", "c"])
        sel.resolve([])
        backend = OpenAIHTTPBackend(target="http://test", model_selector=sel)
        result = await backend.default_model()
        assert result in ["a", "b", "c"]

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_startup_resolves_dynamic_selector(self):
        """Test process_startup resolves a dynamic (n_models) selector."""
        backend = OpenAIHTTPBackend(
            target="http://test",
            model_selector={"n_models": -1},
            validate_backend=False,
        )
        with patch.object(
            backend, "available_models", return_value=["m1", "m2", "m3"]
        ):
            await backend.process_startup()

        assert backend.model_selector.resolved_models == ["m1", "m2", "m3"]
        await backend.process_shutdown()

    @pytest.mark.regression
    @pytest.mark.asyncio
    @async_timeout(10.0)
    async def test_process_startup_without_selector_does_not_call_available_models(
        self,
    ):
        """Test process_startup without a selector does not query available models."""
        backend = OpenAIHTTPBackend(
            target="http://test", model="fixed", validate_backend=False
        )
        with patch.object(
            backend, "available_models", side_effect=AssertionError("should not call")
        ):
            await backend.process_startup()

        assert backend.model_selector is None
        await backend.process_shutdown()
