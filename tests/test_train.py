"""Hermetic tests for the pure training helpers (no network/torch).

The two-stage train_embedder itself is exercised by the marked smoke run (MiniLM, CPU).
"""

import pytest

from lexfr_embed.config import settings
from lexfr_embed.train import build_matryoshka_dims, stage_training_args


def test_matryoshka_dims_filtered_to_model_dim():
    assert build_matryoshka_dims(1024, [1024, 512, 256, 128, 64]) == [1024, 512, 256, 128, 64]
    assert build_matryoshka_dims(512, [1024, 512, 256, 128, 64]) == [512, 256, 128, 64]


def test_matryoshka_dims_fall_back_when_none_fit():
    assert build_matryoshka_dims(384, [1024]) == [384]


def test_matryoshka_dims_always_include_full_model_dim():
    # 384 is not in `wanted`; it must still be included so the full embedding is trained (ST warns otherwise)
    assert build_matryoshka_dims(384, [1024, 512, 256, 128, 64]) == [384, 256, 128, 64]


def test_stage2_halves_lr_and_uses_stage2_epochs():
    a1 = stage_training_args(1, lora=True, out_dir="x")
    a2 = stage_training_args(2, lora=True, out_dir="x")
    assert a2["learning_rate"] == 0.5 * a1["learning_rate"]
    assert a1["num_train_epochs"] == settings.epochs_stage1
    assert a2["num_train_epochs"] == settings.epochs_stage2
    assert a1["bf16"] is True


def test_lr_depends_on_lora_flag():
    assert stage_training_args(1, lora=True, out_dir="x")["learning_rate"] == settings.lr_lora
    assert stage_training_args(1, lora=False, out_dir="x")["learning_rate"] == settings.lr_full_ft


@pytest.mark.smoke
def test_two_stage_smoke_minilm_cpu(tmp_path):
    """Full two-stage path on MiniLM (CPU) + varied toy pairs: proves wiring, mining, and SAVE.

    The loss+mining stack (MNRL⊂Matryoshka -> mine 1 hard neg -> Stage-2) had never run in this
    repo; this is the daylight de-risk before the multi-hour BGE-M3 run.
    """
    from lexfr_embed.train import train_embedder

    topics = [
        ("resiliation du bail par le locataire", "le locataire peut resilier le bail sous reserve d un preavis"),
        ("delai de prescription en matiere civile", "les actions personnelles se prescrivent par cinq ans"),
        ("indemnite de licenciement", "l indemnite de licenciement depend de l anciennete du salarie"),
        ("responsabilite du fait des produits defectueux", "le producteur repond du dommage cause par un defaut"),
        ("creation d une societe a responsabilite limitee", "la sarl est constituee par une ou plusieurs personnes"),
        ("droit de retractation du consommateur", "le consommateur dispose de quatorze jours pour se retracter"),
    ]
    pairs = [
        {"anchor": f"{q} (cas {i})", "positive": f"{a} — article {i}", "code": "civil"}
        for i in range(4)
        for q, a in topics
    ]
    out = tmp_path / "smoke"
    train_embedder(base_model_key="smoke", use_lora=False, train_pairs=pairs, out_dir=str(out), max_steps=1)
    assert (out / "stage1").exists()
    assert (out / "final").exists()
