"""Data tests: pure Python (no torch). Checks against the released data run when it is present."""
import pytest

from moebase import paths
from moebase.data import logic


def rel(a: str, b: str) -> str:
    return logic.compute_relation(logic.satisfying_worlds(logic.parse(a.split())),
                                  logic.satisfying_worlds(logic.parse(b.split())))


def test_parser_roundtrip_shapes():
    t = logic.parse("( not ( e ( or ( not c ) ) ) )".split())
    assert t == ("not", ("e", "or", ("not", "c")))
    assert logic.n_ops(t) == 3 and logic.depth(t) == 3
    assert logic.parse(["d"]) == "d"
    with pytest.raises(ValueError):
        logic.parse("( a ( xor b ) )".split())


@pytest.mark.parametrize("a,b,label", [
    ("a", "( not a )", "^"),                       # negation
    ("a", "( a ( or b ) )", "<"),                  # forward entailment
    ("( a ( or b ) )", "a", ">"),                  # reverse entailment
    ("( a ( and b ) )", "( b ( and a ) )", "="),   # equivalence (operand order irrelevant semantically)
    ("( a ( and b ) )", "( not a )", "|"),         # alternation
    ("a", "( ( not a ) ( or b ) )", "v"),          # cover
    ("a", "b", "#"),                               # independence
])
def test_relation_semantics(a, b, label):
    assert rel(a, b) == label


def test_length_bin_is_per_clause_max():
    ex = logic.Example("#", "( a ( and b ) )".split(), "( not ( c ( or d ) ) )".split(), "x", 0, 1, 2)
    assert ex.length_bin == 2  # max(1, 2), not 3


def test_systematicity_patterns_respect_operand_order():
    pats = logic.SYSTEMATICITY_PATTERNS
    import re
    hit = "( b ( and ( not a ) ) )"
    miss = "( ( not a ) ( and b ) )"
    assert any(re.search(p, hit) for p in pats["A"])
    assert not any(re.search(p, miss) for p in pats["A"])
    assert any(re.search(p, "( b ( or ( not c ) ) )") for p in pats["C"])
    assert not any(re.search(p, "( b ( or ( not c ) ) )") for p in pats["B"])


raw = paths.logic_raw_dir()
needs_raw = pytest.mark.skipif(not (raw / "train0").exists(), reason="run scripts/fetch_upstream.py first")


@needs_raw
def test_released_splits_match_ordered_memory_table2():
    data = logic.load_raw(raw)
    pool = [e for n in range(7) for e in data[f"train{n}"]]
    sizes = {"full": len(pool), **{k: sum(k not in e.tags for e in pool) for k in "ABC"}}
    assert sizes == logic.PUBLISHED_TRAIN_SIZES


@needs_raw
def test_released_labels_and_bins_sample():
    data = logic.load_raw(raw)
    for name in ("train3", "test9", "test12"):
        for e in data[name][:300]:
            ok, got = logic.verify_label(e)
            assert ok, (name, e.line, e.label, got)
            assert not logic.is_degenerate(e)
            assert e.length_bin == int(name.lstrip("traines"))


@needs_raw
def test_no_exact_leakage_and_patterns_excluded():
    data = logic.load_raw(raw)
    splits = logic.build_splits(data)
    rep = logic.audit(data, splits, check_labels=False)
    for k, v in rep["leakage"].items():
        if isinstance(v, dict):
            assert v["exact"] == 0, k
        else:
            assert v == 0, k


def test_interfaces_share_vocab():
    v = logic.Vocab()
    ex = logic.Example("<", ["a"], "( a ( or b ) )".split(), "x", 0)
    ids, y = logic.encode_classifier(ex, v)
    assert ids[0] == v.cls and y == logic.LABELS.index("<")
    src, prev, tgt, _ = logic.encode_seq2seq(ex, v)
    assert src[-1] == v.eos and prev[0] == v.eos and tgt == [v.stoi["REL<"], v.eos]
    ids, pos, _ = logic.encode_causal(ex, v)
    assert ids[pos] == v.stoi["REL<"] and ids[pos - 1] == v.rel
