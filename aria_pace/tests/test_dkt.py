import functools

import numpy as np
import torch

from aria_pace import dkt as D
from aria_pace.labels import first_mastery
from aria_pace.loader import Sequence


def seq(c, kc="1", uid=1):
    c = np.array(c, dtype=np.int8)
    return Sequence(uid, uid, kc, c, len(c), first_mastery(c), np.zeros(len(c), dtype=np.int8))


def constant_model(p, n_kc=3, hidden=8):
    """A DKT whose prediction is the fixed probability p whatever the history (all weights zero, output bias = logit p)."""
    m = D.DKT(n_kc, hidden)
    with torch.no_grad():
        for prm in m.parameters():
            prm.zero_()
        m.out.bias.fill_(float(np.log(p / (1 - p))))
    return m.eval()


@functools.lru_cache(maxsize=None)
def closed_form(p, steps, run):
    """P(no three-in-a-row completes in `steps` more responses) when the current run of correct responses is `run`."""
    if steps == 0:
        return 1.0
    ok = 0.0 if run + 1 >= 3 else closed_form(p, steps - 1, run + 1)
    return p * ok + (1 - p) * closed_form(p, steps - 1, 0)


def test_vocab_threshold_and_unknown():
    train = [seq([0, 1], kc="a") for _ in range(10)] + [seq([0, 1], kc="b") for _ in range(9)]
    v = D.build_vocab(train)
    assert v == {"a": 1}                      # 'b' has 9 < 10 sequences -> UNK (id 0 via .get)
    assert v.get("b", 0) == 0 and v.get("never-seen", 0) == 0


def test_enumeration_matches_the_closed_form_and_probabilities_sum_to_one():
    for p in (0.2, 0.55, 0.9):
        model = constant_model(p)
        for k in (3, 5, 8):
            for tail in ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]):
                prefix = [0] * (k - 3) + tail           # trailing run of 0, 0 / 1 / 1 correct
                run = 0
                for x in prefix[::-1]:
                    if x == 1: run += 1
                    else: break
                q = seq(prefix + [0] * 10)
                alive, mastered = D.risk_scores(model, {"1": 1}, [q], k, return_mastered=True)
                want = closed_form(p, 10 - k, run)
                assert abs(alive[0] - want) < 1e-5, (p, k, prefix, alive[0], want)
                assert abs(alive[0] + mastered[0] - 1.0) < 1e-6


def test_score_uses_only_the_first_k_responses():
    torch.manual_seed(0)
    model = D.DKT(3, 16).eval()                      # random weights: history matters
    vocab = {"1": 1, "2": 2}
    a = seq([0, 1, 0, 0, 1] + [1, 1, 1, 1, 1, 1], kc="2")
    b = seq([0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0], kc="2")
    c = seq([1, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0], kc="2")
    ra, rb, rc = D.risk_scores(model, vocab, [a, b, c], 5)
    assert abs(ra - rb) < 1e-9                       # same first five responses, different futures
    assert abs(ra - rc) > 1e-6                       # different history -> different score (the LSTM reads it)
    assert 0.0 <= ra <= 1.0


def test_training_learns_and_is_deterministic():
    rng = np.random.RandomState(1)
    def make(n):
        out = []
        for i in range(n):
            skill = rng.rand()                         # students differ in how fast they learn
            c = [int(rng.rand() < min(0.9, 0.25 + skill * 0.1 * t)) for t in range(12)]
            out.append(seq(c, kc="1" if i % 2 else "2", uid=i))
        return out
    tr, va = make(400), make(120)
    vocab = D.build_vocab(tr, min_kc=1)
    chance = D.next_response_logloss(D.DKT(len(vocab) + 1, 16).eval(), va, vocab)
    m1, i1 = D.train_one(tr, va, vocab, 16, max_epochs=6, patience=6)
    m2, i2 = D.train_one(tr, va, vocab, 16, max_epochs=6, patience=6)
    assert i1["val_next_response_logloss"] < chance - 0.02           # it learned something
    assert i1["val_curve"] == i2["val_curve"]                       # same seed -> identical training
    qs = [q for q in va if q.n >= 5 and q.m > 5][:20]
    assert np.array_equal(D.risk_scores(m1, vocab, qs, 5), D.risk_scores(m2, vocab, qs, 5))
