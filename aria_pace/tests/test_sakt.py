import numpy as np
import torch

from aria_pace import dkt as D
from aria_pace import sakt as S
from aria_pace.tests.test_dkt import closed_form, seq


def constant_sakt(p, n_kc=3, d=8):
    m = S.SAKT(n_kc, d)
    with torch.no_grad():
        for prm in m.parameters():
            prm.zero_()
        m.out.bias.fill_(float(np.log(p / (1 - p))))
    return m.eval()


def test_enumeration_matches_the_closed_form_and_sums_to_one():
    for p in (0.2, 0.55, 0.9):
        model = constant_sakt(p)
        for k in (3, 5, 8):
            for tail in ([0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]):
                prefix = [0] * (k - 3) + tail
                run = 0
                for x in prefix[::-1]:
                    if x == 1: run += 1
                    else: break
                alive, mastered = S.risk_scores(model, {"1": 1}, [seq(prefix + [0] * 10)], k, return_mastered=True)
                assert abs(alive[0] - closed_form(p, 10 - k, run)) < 1e-5, (p, k, prefix)
                assert abs(alive[0] + mastered[0] - 1.0) < 1e-6


def test_causal_attention_future_tokens_never_change_earlier_predictions():
    torch.manual_seed(0)
    model = S.SAKT(3, 16).eval()
    tok_a = torch.tensor([[2, 3, 2, 2, 3, 3, 2]])
    tok_b = torch.tensor([[2, 3, 2, 2, 3, 2, 3]])           # differs only in the last two interaction tokens
    kc = torch.tensor([1])
    la, lb = model(tok_a, kc), model(tok_b, kc)               # (1, 6): entry j predicts step j+2
    assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-6)   # predictions that only see steps 1..5 are identical
    assert not torch.allclose(la[:, 5], lb[:, 5], atol=1e-6)


def test_batch_forward_agrees_with_the_single_query_path_used_for_scoring():
    torch.manual_seed(1)
    model = S.SAKT(3, 16).eval()
    tok = torch.tensor([[2, 3, 3, 2, 3, 2]])
    kc = torch.tensor([1])
    full = model(tok, kc)                                      # entry j: logit that step j+2 is correct
    for t in range(1, 6):
        single = model.predict_next(tok[:, :t], kc)
        assert abs(float(full[0, t - 1]) - float(single[0])) < 1e-5, t


def test_score_uses_only_the_first_k_responses_and_training_is_deterministic():
    torch.manual_seed(0)
    model = S.SAKT(3, 16).eval()
    vocab = {"1": 1, "2": 2}
    a = seq([0, 1, 0, 0, 1] + [1, 1, 1, 1, 1, 1], kc="2")
    b = seq([0, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0], kc="2")
    c = seq([1, 1, 0, 0, 1] + [0, 0, 0, 0, 0, 0], kc="2")
    ra, rb, rc = S.risk_scores(model, vocab, [a, b, c], 5)
    assert abs(ra - rb) < 1e-9 and abs(ra - rc) > 1e-6

    rng = np.random.RandomState(1)
    def make(n):
        out = []
        for i in range(n):
            s = rng.rand()
            out.append(seq([int(rng.rand() < min(0.9, 0.25 + s * 0.1 * t)) for t in range(12)], kc="1" if i % 2 else "2", uid=i))
        return out
    tr, va = make(400), make(120)
    vocab = D.build_vocab(tr, min_kc=1)
    chance = S.next_response_logloss(S.SAKT(len(vocab) + 1, 16).eval(), va, vocab)
    m1, i1 = S.train_one(tr, va, vocab, 16, max_epochs=6, patience=6)
    m2, i2 = S.train_one(tr, va, vocab, 16, max_epochs=6, patience=6)
    assert i1["val_next_response_logloss"] < chance - 0.02
    assert i1["val_curve"] == i2["val_curve"]
    qs = [q for q in va if q.n >= 5 and q.m > 5][:20]
    assert np.array_equal(S.risk_scores(m1, vocab, qs, 5), S.risk_scores(m2, vocab, qs, 5))
