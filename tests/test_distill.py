"""Hermetic tests for the A1-bis distillation label math + loss wiring (no network/GPU).

The teacher pass (embedder mining + reranker scoring) and the distill training loop are
integration layers exercised later on a GPU; here we lock the *pure* label math, the dataset
shaping (incl. the ragged-label drop guard), and the loss wiring — mirroring how
tests/test_hard_negatives.py monkeypatches sentence_transformers so nothing heavy runs.
"""

import pytest

from lexfr_embed.config import settings


def test_apply_activation():
    """sigmoid(0)==0.5, monotonic, bounded (0,1); identity passthrough; tanh in (-1,1)."""
    from lexfr_embed.distill import apply_activation

    assert apply_activation(0.0, "sigmoid") == pytest.approx(0.5)
    assert apply_activation(-1.0, "sigmoid") < apply_activation(0.0, "sigmoid") < apply_activation(1.0, "sigmoid")
    for x in (-10.0, -1.0, 0.0, 1.0, 10.0):
        assert 0.0 < apply_activation(x, "sigmoid") < 1.0

    assert apply_activation(3.7, "identity") == 3.7
    assert apply_activation(-2.1, "identity") == -2.1

    for x in (-10.0, 0.0, 10.0):
        assert -1.0 < apply_activation(x, "tanh") < 1.0
    assert apply_activation(0.0, "tanh") == pytest.approx(0.0)

    with pytest.raises(ValueError):
        apply_activation(0.0, "softmax")  # unknown kind must fail loudly


def test_teacher_label_row_shape_and_values():
    """pos=0.9, negs=[0.2,0.1]: identity -> [0.9,0.2,0.1] (len K+1, pos first); sigmoid -> act of each."""
    from lexfr_embed.distill import apply_activation, teacher_label_row

    row = teacher_label_row(0.9, [0.2, 0.1], activation="identity")
    assert row == [0.9, 0.2, 0.1]
    assert len(row) == 3  # K + 1

    sig = teacher_label_row(0.9, [0.2, 0.1], activation="sigmoid")
    assert sig == [apply_activation(v, "sigmoid") for v in (0.9, 0.2, 0.1)]
    assert sig[0] == apply_activation(0.9, "sigmoid")  # pos-first ordering MarginMSE expects


def test_build_distill_dataset_columns():
    """Fake cached records -> Dataset with columns anchor, positive, negative_1..K, label; len(label)==K+1."""
    from lexfr_embed.distill import build_distill_dataset

    k = settings.distill_num_negatives
    records = [
        {
            "anchor": f"q{i}",
            "positive": f"p{i}",
            "negatives": [f"n{i}_{j}" for j in range(k)],
            "label": [0.9] + [0.1] * k,
        }
        for i in range(2)
    ]
    ds = build_distill_dataset(records)
    expected = {"anchor", "positive", "label"} | {f"negative_{j}" for j in range(1, k + 1)}
    assert set(ds.column_names) == expected
    assert len(ds) == 2  # row count preserved
    assert len(ds[0]["label"]) == k + 1


def test_build_distill_dataset_drops_incomplete_rows():
    """A record with < K negatives is dropped (guards the ragged-label silent bug), keeping the rest."""
    from lexfr_embed.distill import build_distill_dataset

    k = settings.distill_num_negatives
    if k < 1:
        pytest.skip("drop guard is only meaningful for K >= 1")
    complete = {
        "anchor": "q1",
        "positive": "p1",
        "negatives": [f"n{j}" for j in range(k)],
        "label": [0.9] + [0.1] * k,
    }
    incomplete = {  # one fewer negative than required -> must be dropped
        "anchor": "q2",
        "positive": "p2",
        "negatives": [f"m{j}" for j in range(k - 1)],
        "label": [0.9] + [0.1] * (k - 1),
    }
    ds = build_distill_dataset([complete, incomplete])
    assert len(ds) == 1  # incomplete dropped -> the drop is counted as a length difference
    assert ds[0]["anchor"] == "q1"


def test_make_distill_loss_uses_marginmse_cosine_and_matryoshka(monkeypatch):
    """MarginMSE built with similarity_fct = util.pairwise_cos_sim; wrapped in Matryoshka (with
    build_matryoshka_dims) iff distill_matryoshka. Mirrors test_hard_negatives' monkeypatch style."""
    import sentence_transformers.sentence_transformer.losses as st_losses
    import sentence_transformers.util as st_util

    from lexfr_embed import distill
    from lexfr_embed.train import build_matryoshka_dims

    captured: dict = {}

    class FakeMarginMSE:
        def __init__(self, model, similarity_fct=None):
            captured["margin_model"] = model
            captured["similarity_fct"] = similarity_fct

    class FakeMatryoshka:
        def __init__(self, model, loss, matryoshka_dims=None):
            captured["matryoshka_model"] = model
            captured["inner_loss"] = loss
            captured["matryoshka_dims"] = matryoshka_dims

    sentinel_cos = object()
    monkeypatch.setattr(st_util, "pairwise_cos_sim", sentinel_cos)
    monkeypatch.setattr(st_losses, "MarginMSELoss", FakeMarginMSE)
    monkeypatch.setattr(st_losses, "MatryoshkaLoss", FakeMatryoshka)

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 1024

    model = FakeModel()

    monkeypatch.setattr(settings, "distill_student_sim", "cos")
    monkeypatch.setattr(settings, "distill_matryoshka", True)
    loss = distill.make_distill_loss(model)
    assert isinstance(loss, FakeMatryoshka)
    assert isinstance(captured["inner_loss"], FakeMarginMSE)
    assert captured["similarity_fct"] is sentinel_cos
    assert captured["matryoshka_dims"] == build_matryoshka_dims(1024, settings.matryoshka_dims)

    captured.clear()
    monkeypatch.setattr(settings, "distill_matryoshka", False)
    loss_flat = distill.make_distill_loss(model)
    assert isinstance(loss_flat, FakeMarginMSE)  # NOT wrapped when the flag is off
    assert "matryoshka_dims" not in captured


def test_mine_teacher_candidates_is_embedder_only_no_filter(monkeypatch):
    """Executable guard against SILENT-BUG #1: the teacher-candidate miner forwards
    output_format='n-tuple', num_negatives=K, sampling_strategy='top', and NEVER a cross_encoder
    or relative_margin/max_score (those would silently re-score with the embedder, not the teacher)."""
    import sentence_transformers.util as st_util

    from lexfr_embed.data import hard_negatives as hn

    captured: dict = {}

    def fake_mine(dataset, model, **kwargs):
        captured.update(kwargs)
        return "NTUPLES"

    monkeypatch.setattr(st_util, "mine_hard_negatives", fake_mine)
    out = hn.mine_teacher_candidates(
        [{"anchor": "q", "positive": "p"}], model=object(), num_negatives=settings.distill_num_negatives
    )
    assert out == "NTUPLES"
    assert captured["output_format"] == "n-tuple"
    assert captured["num_negatives"] == settings.distill_num_negatives
    assert captured["sampling_strategy"] == "top"
    assert captured["use_faiss"] is True
    assert "cross_encoder" not in captured
    assert "relative_margin" not in captured
    assert "max_score" not in captured
