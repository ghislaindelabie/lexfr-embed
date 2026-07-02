"""Hermetic unit tests for the train/eval leakage-control spine (no network/torch).

Backs: hard-exclude of eval gold from training, order-independent partition hashes
(written to results/partition_hashes.json before any mining), and the NC/SA whitelist.
"""

from lexfr_embed.data.leakage import (
    canonical_id,
    filter_pairs_by_ids,
    hard_exclude,
    hash_partition,
    is_barred_source,
)


def test_canonical_id_case_and_whitespace_insensitive():
    assert canonical_id("CODE CIVIL", "1240") == canonical_id("code civil", "1240")
    assert canonical_id("Code civil", 1240) == canonical_id("  code   civil ", "1240")


def test_canonical_id_idempotent():
    once = canonical_id("Code civil", "1240")
    code, num = once.split("|")
    assert canonical_id(code, num) == once


def test_hard_exclude_yields_disjoint_training_set():
    train = {"code civil|1240", "code civil|1241", "code du travail|l1234-9"}
    ev = {"code civil|1240"}
    clean = hard_exclude(train, ev)
    assert clean.isdisjoint(ev)
    assert "code civil|1240" not in clean
    assert "code civil|1241" in clean


def test_hash_partition_order_independent_and_dedup():
    assert hash_partition(["b|2", "a|1", "c|3"]) == hash_partition(["c|3", "a|1", "b|2", "a|1"])


def test_hash_partition_changes_when_an_id_moves():
    assert hash_partition(["a|1", "b|2"]) != hash_partition(["a|1", "b|3"])


def test_is_barred_source_flags_bsard_and_nc_sa():
    assert is_barred_source("bsard")
    assert is_barred_source("maastrichtlawtech/bsard")
    assert not is_barred_source("louisbrulenaudet/legalkit")


def test_filter_pairs_by_ids_drops_barred():
    pairs = [
        {"anchor": "q1", "positive": "a", "id": "code civil|1240"},
        {"anchor": "q2", "positive": "b", "id": "code civil|1241"},
    ]
    kept = filter_pairs_by_ids(pairs, barred_ids={"code civil|1240"}, id_key="id")
    assert len(kept) == 1
    assert kept[0]["id"] == "code civil|1241"
