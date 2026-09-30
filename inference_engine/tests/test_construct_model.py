"""Finding a model distribution's factory by its ref, and calling it."""

import sys
from dataclasses import fields
from pathlib import Path

import pytest

from doubles import CallLog, ScriptedModel, installed_distribution, installed_factory
from robin_contracts.cards import AudioGeometry, ModelCard, ModelRef, RunnerResampled, model_ref
from robin_contracts.embedding_transforms import Identity
from robin_contracts.protocols import ModelContext
from robin_inference_engine import errors
from robin_inference_engine.construct_model import (
    ENTRY_POINT_GROUP,
    construct_model,
    installed_models,
)

CARD = ModelCard(
    model_name="test-model",
    model_version="1",
    runtime="none",
    window_duration=3.0,
    sample_rate=16000,
    min_detection_threshold=0.0,
    window_overlap=0.0,
    score_domain="probability",
    audio=AudioGeometry(
        downmix="mean", resampler=RunnerResampled(algorithm="soxr_hq"), pad="drop"
    ),
    backend="none",
    embedding_transform=Identity(),
    dtype="float32",
)
REF = model_ref(CARD)
OTHER_REF = ModelRef(name="absent-model", version="9", digest=REF.digest)

def build_context(scratch_dir: Path) -> ModelContext:
    return ModelContext(
        card=CARD,
        registry=None,
        files={"weights": scratch_dir / "weights.bin"},
        settings={"top_k": None, "gain": 1.5},
        resources={"batch_size": 8},
        scratch_dir=scratch_dir,
        emit_embeddings=False,
        log=print,
    )


def build_model(calls: CallLog) -> ScriptedModel:
    return ScriptedModel(script=(), calls=calls)


def refusal(**kwargs) -> errors.EngineError:
    with pytest.raises(errors.EngineError) as caught:
        construct_model(**kwargs)
    assert caught.value.stage == errors.CONSTRUCT_MODEL
    return caught.value


def test_the_group_is_the_one_model_distributions_register_under():
    assert ENTRY_POINT_GROUP == "robin.models"


def test_listing_names_an_installed_model_without_importing_it(tmp_path):
    module = "robin_test_listing_only"
    with installed_distribution(
        tmp_path,
        name="robin-test-listing",
        entry_points={REF.id: f"{module}:build"},
        modules={module: "raise RuntimeError('imported by a listing')\n"},
    ):
        listed = installed_models()

        assert REF.id in listed
        assert module not in sys.modules


def test_listing_is_sorted_and_names_each_model_once(tmp_path):
    with installed_distribution(
        tmp_path,
        name="robin-test-one",
        entry_points={"zeta/1": "zeta_module:build", "alpha/1": "alpha_module:build"},
        modules={},
    ), installed_distribution(
        tmp_path,
        name="robin-test-two",
        entry_points={"alpha/1": "other_alpha_module:build"},
        modules={},
    ):
        listed = installed_models()

    assert list(listed) == sorted(set(listed))
    assert {"alpha/1", "zeta/1"} <= set(listed)


def test_a_registered_factory_constructs_the_model(tmp_path):
    model = build_model([])
    with installed_factory(tmp_path, REF.id, lambda context: model):
        built = construct_model(ref=REF, context=build_context(tmp_path))

    assert built is model


def test_the_factory_receives_the_context_the_caller_built(tmp_path):
    received: list[ModelContext] = []

    def factory(context: ModelContext) -> ScriptedModel:
        received.append(context)
        return build_model([])

    context = build_context(tmp_path)
    with installed_factory(tmp_path, REF.id, factory):
        construct_model(ref=REF, context=context)

    assert len(received) == 1
    for field in fields(ModelContext):
        assert getattr(received[0], field.name) is getattr(context, field.name), field.name


def test_constructing_imports_only_the_requested_model(tmp_path):
    other = "robin_test_not_requested"
    with installed_distribution(
        tmp_path,
        name="robin-test-other",
        entry_points={OTHER_REF.id: f"{other}:build"},
        modules={other: "raise RuntimeError('imported without being requested')\n"},
    ), installed_factory(tmp_path, REF.id, lambda context: build_model([])):
        construct_model(ref=REF, context=build_context(tmp_path))

        assert other not in sys.modules


def test_an_unknown_ref_is_refused_naming_what_is_installed(tmp_path):
    with installed_factory(tmp_path, REF.id, lambda context: build_model([])):
        error = refusal(ref=OTHER_REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_NOT_INSTALLED
    assert OTHER_REF.id in error.detail
    assert REF.id in error.detail


def test_two_distributions_registering_one_ref_are_refused_naming_both(tmp_path):
    with installed_factory(
        tmp_path, REF.id, lambda context: build_model([]), distribution="robin-fork-b"
    ) as second, installed_factory(
        tmp_path, REF.id, lambda context: build_model([]), distribution="robin-fork-a"
    ) as first:
        error = refusal(ref=REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_REGISTERED_TWICE
    assert REF.id in error.detail
    assert error.detail.index("robin-fork-a 0") < error.detail.index("robin-fork-b 0")
    assert f"{first}:build" in error.detail
    assert f"{second}:build" in error.detail


def test_an_entry_point_naming_a_missing_attribute_is_refused(tmp_path):
    module = "robin_test_no_factory"
    with installed_distribution(
        tmp_path,
        name="robin-test-no-factory",
        entry_points={REF.id: f"{module}:build"},
        modules={module: "NOT_A_FACTORY = None\n"},
    ):
        error = refusal(ref=REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_ENTRY_POINT_UNLOADABLE
    assert REF.id in error.detail
    assert "AttributeError" in error.detail


def test_an_entry_point_naming_a_missing_module_is_refused(tmp_path):
    with installed_distribution(
        tmp_path,
        name="robin-test-no-module",
        entry_points={REF.id: "robin_test_module_that_is_not_there:build"},
        modules={},
    ):
        error = refusal(ref=REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_ENTRY_POINT_UNLOADABLE
    assert "ModuleNotFoundError" in error.detail


def test_a_factory_that_raises_is_refused_carrying_the_exception(tmp_path):
    def factory(context: ModelContext) -> ScriptedModel:
        raise FileNotFoundError("weights.bin is not a model")

    with installed_factory(tmp_path, REF.id, factory):
        error = refusal(ref=REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_CONSTRUCTION_FAILED
    assert "FileNotFoundError" in error.detail
    assert "weights.bin is not a model" in error.detail


def test_a_factory_returning_a_plain_object_is_refused(tmp_path):
    with installed_factory(tmp_path, REF.id, lambda context: object()):
        error = refusal(ref=REF, context=build_context(tmp_path))

    assert error.code == errors.MODEL_PROTOCOL_UNSATISFIED
    assert REF.id in error.detail


def test_an_installed_test_distribution_is_gone_once_its_context_exits(tmp_path):
    with installed_factory(tmp_path, REF.id, lambda context: build_model([])):
        assert REF.id in installed_models()

    assert REF.id not in installed_models()
    assert list(tmp_path.iterdir()) == []
