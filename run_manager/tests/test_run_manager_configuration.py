"""The inference configuration: its file, its checks against the model, and its digest.

These tests use no database. The callers look up the models.
"""

from pathlib import Path

import pytest

from robin_contracts.cards import card_digest, model_ref
from robin_contracts.work import AudioInput, InferenceWork, PinnedModel, RecordingRef
from robin_run_manager.configuration import (
    Configuration,
    UnresolvedConfiguration,
    get_configuration_digest,
    get_embedding_input,
    read_configuration,
    resolve_configuration,
)
from robin_run_manager.errors import INVALID_CONFIGURATION, INVALID_REQUEST, Refused

from model_fixtures import read_pinned_model

FIXTURES = Path(__file__).parent / "fixtures"
OWL = read_pinned_model("owl-v4")
PERCH = read_pinned_model("perch-v8")
HEAD = read_pinned_model("songbird-head-v1")
AUDIO = AudioInput()

# The digest of configuration-owl-v4.yaml. It must not change: runs and presets store it.
GOLDEN_DIGEST = "sha256:v1:e5227f7649736eee7a8cfcad2d8a59b37de4e261617c49f66de62c4756ba0b5b"

EXAMPLE_FILE = """\
model: owl/v4
input:
  kind: audio
settings:
  window_overlap: 0.0
outputs:
  - kind: scores
    contract_id: robin.scores.parquet/1
    retention: full
"""


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "configuration.yaml"
    path.write_text(text)
    return path


def owl_file(**changes) -> UnresolvedConfiguration:
    fields = read_configuration(FIXTURES / "configuration-owl-v4.yaml").model_dump()
    return UnresolvedConfiguration.model_validate(fields | changes)


def perch_configuration():
    file = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
    return resolve_configuration(file, PERCH, AUDIO)


def get_audio_digest(file: UnresolvedConfiguration, model: PinnedModel) -> str:
    return get_configuration_digest(resolve_configuration(file, model, AUDIO))


def head_file(backbone_digest: str, **changes) -> UnresolvedConfiguration:
    fields = {
        "model": "songbird-head/v1",
        "input": {"kind": "embedding_artifact", "backbone_configuration": backbone_digest},
        "outputs": [{"kind": "scores", "contract_id": "robin.scores.parquet/1", "retention": "full"}],
    }
    return UnresolvedConfiguration.model_validate(fields | changes)


def refused_code(call) -> str:
    with pytest.raises(Refused) as refused:
        call()
    assert refused.value.reason == INVALID_CONFIGURATION
    return refused.value.details.get("code")


# ---------------------------------------------------------------------------
# The file.
# ---------------------------------------------------------------------------


def test_the_documented_example_reads_and_resolves(tmp_path):
    spec = resolve_configuration(read_configuration(write(tmp_path, EXAMPLE_FILE)), OWL, AUDIO)

    assert spec.outputs[0].contract_id == "robin.scores.parquet/1"
    assert spec.model.name == "owl" and spec.model.version == "v4"
    assert spec.model.card_digest == card_digest(OWL.card)
    assert dict(spec.settings) == {"window_overlap": 0.0}


def test_the_golden_file_gives_the_golden_digest():
    file = read_configuration(FIXTURES / "configuration-owl-v4.yaml")
    spec = resolve_configuration(file, OWL, AUDIO)

    assert get_configuration_digest(spec) == GOLDEN_DIGEST


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param("    contract_id: robin.scores.parquet/1\n", "", id="missing"),
        pytest.param("robin.scores.parquet/1", "robin.scores.parquet/2", id="unknown"),
    ],
)
def test_a_missing_or_unknown_contract_id_is_invalid(tmp_path, old, new):
    path = write(tmp_path, EXAMPLE_FILE.replace(old, new))

    with pytest.raises(Refused) as refused:
        read_configuration(path)

    assert refused.value.reason == INVALID_CONFIGURATION
    assert any("contract_id" in p["field"] for p in refused.value.details["problems"])


HEADER = "model: owl/v4\ninput: {kind: audio}\n"
SCORES = "  - {kind: scores, contract_id: robin.scores.parquet/1, retention: full}\n"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(HEADER + "outputs: []\n", id="no_outputs"),
        pytest.param(HEADER + "outputs:\n" + SCORES + SCORES, id="a_repeated_kind"),
        pytest.param(
            HEADER + "outputs:\n"
            "  - kind: detections\n"
            "    contract_id: robin.detections.parquet/1\n"
            "    policy: {kind: threshold, min_score: 0.5}\n",
            id="detections_without_scores",
        ),
        pytest.param(
            HEADER + "settings: {window_overlap: .nan}\noutputs:\n" + SCORES, id="a_nan_setting"
        ),
        pytest.param(
            HEADER + "settings: {window_overlap: .inf}\noutputs:\n" + SCORES,
            id="an_infinite_setting",
        ),
        pytest.param(HEADER, id="a_missing_field"),
        pytest.param(EXAMPLE_FILE + "extra: 1\n", id="an_unknown_field"),
    ],
)
def test_a_file_that_breaks_a_work_rule_is_invalid(tmp_path, text):
    with pytest.raises(Refused) as refused:
        read_configuration(write(tmp_path, text))

    assert refused.value.reason == INVALID_CONFIGURATION


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("model: [unclosed", id="not_yaml"),
        pytest.param(EXAMPLE_FILE + "model: perch/v8\n", id="a_repeated_key"),
    ],
)
def test_a_file_that_is_not_yaml_is_an_invalid_request(tmp_path, text):
    with pytest.raises(Refused) as refused:
        read_configuration(write(tmp_path, text))

    assert refused.value.reason == INVALID_REQUEST


def test_a_missing_file_is_an_invalid_request(tmp_path):
    with pytest.raises(Refused) as refused:
        read_configuration(tmp_path / "absent.yaml")

    assert refused.value.reason == INVALID_REQUEST


# ---------------------------------------------------------------------------
# The digest.
# ---------------------------------------------------------------------------


def test_another_key_order_gives_the_same_digest(tmp_path):
    reordered = """\
outputs:
  - retention: full
    contract_id: robin.scores.parquet/1
    kind: scores
settings:
  window_overlap: 0.0
input:
  kind: audio
model: owl/v4
"""
    file = read_configuration(write(tmp_path, reordered))

    assert get_audio_digest(file, OWL) == get_audio_digest(owl_file(), OWL)


def test_another_output_order_gives_the_same_digest():
    scores = {"kind": "scores", "contract_id": "robin.scores.parquet/1", "retention": "full"}
    embeddings = {"kind": "embeddings", "contract_id": "robin.embeddings.parquet/1"}
    fields = {"model": "perch/v8", "input": {"kind": "audio"}}
    one = UnresolvedConfiguration.model_validate(fields | {"outputs": [scores, embeddings]})
    other = UnresolvedConfiguration.model_validate(fields | {"outputs": [embeddings, scores]})

    assert get_audio_digest(one, PERCH) == get_audio_digest(other, PERCH)


def test_a_moved_model_file_gives_the_same_digest():
    files = dict(OWL.files)
    files["weights"] = files["weights"].model_copy(update={"uri": "s3://elsewhere/w.h5"})
    moved = OWL.model_copy(update={"files": files})

    assert get_audio_digest(owl_file(), moved) == get_audio_digest(owl_file(), OWL)


def test_another_file_digest_gives_another_digest():
    files = dict(OWL.files)
    files["weights"] = files["weights"].model_copy(update={"digest": "sha256:" + "9" * 64})
    changed = OWL.model_copy(update={"files": files})

    assert get_audio_digest(owl_file(), changed) != get_audio_digest(owl_file(), OWL)


def test_a_configuration_holds_no_file_locations():
    spec = resolve_configuration(owl_file(), OWL, AUDIO)

    assert "s3://" not in str(spec.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# The checks against the model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        pytest.param({"settings": {"gain": 1.0}}, "setting_undeclared", id="setting_undeclared"),
        pytest.param({"settings": {"window_overlap": 1}}, "setting_type_mismatch", id="type"),
        pytest.param({"settings": {"window_overlap": 12.0}}, "window_overlap_invalid", id="overlap"),
        pytest.param(
            {"outputs": [{"kind": "embeddings", "contract_id": "robin.embeddings.parquet/1"}]},
            "embeddings_not_emitted",
            id="embeddings_not_emitted",
        ),
        pytest.param(
            {"outputs": [{"kind": "scores", "contract_id": "robin.scores.parquet/1",
                          "retention": "thresholded", "min_score": 1.5}]},
            "min_score_out_of_domain",
            id="min_score_out_of_domain",
        ),
        pytest.param(
            {"input": {"kind": "embedding_artifact", "backbone_configuration": "sha256:v1:" + "0" * 64}},
            "input_kind_disagrees_with_card",
            id="input_kind",
        ),
    ],
)
def test_a_shared_check_refuses_with_its_code(changes, code):
    work_input = get_embedding_input(perch_configuration(), PERCH) if "input" in changes else AUDIO

    assert refused_code(lambda: resolve_configuration(owl_file(**changes), OWL, work_input)) == code


def test_a_registry_the_card_does_not_name_is_refused():
    files = dict(OWL.files)
    files["taxa_registry"] = files["taxa_registry"].model_copy(update={"digest": "sha256:" + "8" * 64})
    model = OWL.model_copy(update={"files": files})

    code = refused_code(lambda: resolve_configuration(owl_file(), model, AUDIO))

    assert code == "registry_disagrees_with_card"


def test_a_head_resolves_from_its_backbone_configuration():
    backbone = perch_configuration()

    work_input = get_embedding_input(backbone, PERCH)
    spec = resolve_configuration(head_file(get_configuration_digest(backbone)), HEAD, work_input)

    assert spec.input.backbone_configuration == get_configuration_digest(backbone)


def test_a_backbone_configuration_without_embeddings_is_refused():
    scores_only = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
    scores_only = UnresolvedConfiguration.model_validate(
        scores_only.model_dump() | {"outputs": [scores_only.model_dump()["outputs"][0]]}
    )
    backbone = resolve_configuration(scores_only, PERCH, AUDIO)

    code = refused_code(lambda: get_embedding_input(backbone, PERCH))

    assert code == "backbone_requests_no_embeddings"


def make_other_backbone(**card_changes) -> tuple[Configuration, PinnedModel]:
    """A perch-like backbone model with a changed card, and its configuration."""
    card = PERCH.card.model_copy(update=card_changes)
    model = PERCH.model_copy(update={"card": card})
    file = read_configuration(FIXTURES / "configuration-perch-v8.yaml")
    file = UnresolvedConfiguration.model_validate(file.model_dump() | {"model": model_ref(card).id})
    return resolve_configuration(file, model, AUDIO), model


@pytest.mark.parametrize(
    "card_changes",
    [
        pytest.param({"model_name": "birdnet"}, id="another_name"),
        pytest.param({"sample_rate": 48000}, id="another_card_digest"),
    ],
)
def test_a_backbone_that_is_not_the_heads_is_refused(card_changes):
    backbone, model = make_other_backbone(**card_changes)
    work_input = get_embedding_input(backbone, model)

    file = head_file(get_configuration_digest(backbone))

    code = refused_code(lambda: resolve_configuration(file, HEAD, work_input))

    assert code == "head_backbone_disagrees"


def test_a_head_of_another_width_than_its_backbone_is_refused():
    spec, model = make_other_backbone(embedding_dim=1024)
    head = HEAD.model_copy(
        update={"card": HEAD.card.model_copy(update={"backbone": model_ref(model.card)})}
    )

    code = refused_code(
        lambda: resolve_configuration(
            head_file(get_configuration_digest(spec)), head, get_embedding_input(spec, model)
        )
    )

    assert code == "head_backbone_disagrees"


def test_a_head_with_settings_is_refused():
    backbone = perch_configuration()
    file = head_file(get_configuration_digest(backbone), settings={"window_overlap": 1.0})

    work_input = get_embedding_input(backbone, PERCH)

    code = refused_code(lambda: resolve_configuration(file, HEAD, work_input))

    assert code == "setting_undeclared"


# ---------------------------------------------------------------------------
# A resolved configuration always makes a valid work.
# ---------------------------------------------------------------------------


def valid_configurations():
    backbone = perch_configuration()
    return [
        (owl_file(), OWL, AUDIO),
        (read_configuration(FIXTURES / "configuration-perch-v8.yaml"), PERCH, AUDIO),
        (
            read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml"),
            HEAD,
            get_embedding_input(backbone, PERCH),
        ),
    ]


def test_the_head_fixture_names_the_perch_fixture():
    head = read_configuration(FIXTURES / "configuration-songbird-head-v1.yaml")

    assert head.input.backbone_configuration == get_configuration_digest(perch_configuration())


@pytest.mark.parametrize("index", range(3), ids=["owl", "perch", "head"])
def test_each_valid_configuration_builds_an_inference_work(index):
    file, model, input = valid_configurations()[index]
    spec = resolve_configuration(file, model, input)
    embeddings = None
    if input.kind == "embedding_artifact":
        embeddings = {"uri": "s3://b/42.parquet", "checksum": "sha256:" + "0" * 64}

    work = InferenceWork(
        schema_version="robin.inference-work/1",
        recordings=(
            RecordingRef(namespace="soundhub", value="42", audio_uri="s3://b/42.wav",
                         embeddings=embeddings),
        ),
        model=PinnedModel.model_validate(model.model_dump()),
        input=input,
        settings=spec.settings,
        resources={},
        outputs=spec.outputs,
    )

    assert work.outputs == spec.outputs
