"""ARIA-Pace inside the live demo: a per-session BKT tracker, a per-turn risk score, and the speak / watch rule.

Everything numeric is read from frozen files in aria_pace/results/ (nothing is tuned or hard-coded here):
    frozen_v2.json            primary_risk_score = "bkt"; sha256 of frozen_val.json (both files are checked against their .sha256)
    frozen_val.json           per-profile thresholds tau (configs.main.frozen.taus.bkt), Platt calibration of the BKT score
    baselines_val.json        pooled BKT parameters (selection.bkt_pooled_params): the demo has no KC, so the pooled fit is used
    policy.py (PROFILES, DEC) the cost weights (w_I, w_D) behind each profile, and the decision points DEC = arange(3, 10) (SPEC section 7:
                              help is decided after opportunity i = 3..9), both read by AST (importing it would pull in sklearn, +3.7 s)

The gate applies only at attempts K_MIN..K_MAX = 3..9 (attempt = number of verified answers so far, this one included). Before it
(attempts 1-2) and after it (10+) the old rule holds: speak on a verified wrong step. The frozen policy was never fitted outside 3..9.

The tracker is a pure-Python port of aria_pace.baselines.BKT (filter + exact "no mastery by opportunity 10" DP); tests/test_demo_pace.py
checks it against aria_pace's own BKT.score and aria_pace.policy.platt_apply on every 0/1 sequence of length 1..8.

Profiles are cost weights (how much an unnecessary interruption costs relative to delayed help), not learner types.
Nothing here measures learning, and nothing says anything about any clinical or demographic group.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

RESULTS = Path(__file__).resolve().parent.parent / "aria_pace" / "results"
POLICY_SRC = Path(__file__).resolve().parent.parent / "aria_pace" / "policy.py"

PROFILE_ORDER = ("quiet", "balanced", "supportive")
PROFILE_LABELS = {"quiet": "Quiet", "balanced": "Balanced", "supportive": "Supportive"}
DEFAULT_PROFILE = "balanced"

# SymPy verdicts (agent/math_check.py) -> BKT observation. restate / none: no update.
RIGHT = frozenset({"correct", "partial", "step"})
WRONG = frozenset({"wrong"})

HORIZON = 10          # aria_pace.constants.T_DETERMINE: determination point, in opportunities
MASTERY_RUN = 3       # aria_pace.constants.MASTERY_RUN: three correct in a row = mastered
EPS = 1e-6            # aria_pace.policy.EPS (Platt clipping)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked(path: Path, sha_file: Path, expect: Optional[str] = None) -> dict:
    got = _sha(path)
    want = expect or sha_file.read_text().strip()
    if got != want:
        raise RuntimeError(f"ARIA-Pace: {path.name} does not match its frozen sha256 ({got[:12]} != {want[:12]}); refusing to start")
    return json.loads(path.read_text())


def _profile_weights(src: Path) -> dict:
    """PROFILES = {...} from aria_pace/policy.py, read as a literal (no import)."""
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROFILES" for t in node.targets):
            return {k: tuple(float(x) for x in v) for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("ARIA-Pace: PROFILES not found in aria_pace/policy.py")


def _decision_window(src: Path) -> tuple:
    """(K_MIN, K_MAX) from `DEC = np.arange(a, b)` in aria_pace/policy.py: decision points a .. b-1."""
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DEC" for t in node.targets):
            lo, hi = (ast.literal_eval(a) for a in node.value.args)
            return int(lo), int(hi) - 1
    raise RuntimeError("ARIA-Pace: DEC not found in aria_pace/policy.py")


@dataclass(frozen=True)
class PaceConfig:
    k_min: int            # first attempt at which the gate applies (frozen policy: 3)
    k_max: int            # last attempt at which the gate applies (frozen policy: 9)
    taus: dict            # profile -> tau on the Platt-calibrated BKT risk
    weights: dict         # profile -> (w_I, w_D)
    platt: tuple          # (intercept, slope)
    params: dict          # pooled BKT: prior / learn / guess / slip
    sources: dict         # file -> sha256 (shown in tests / logs, never student text)

    def profiles(self) -> list:
        return [{"key": k, "label": PROFILE_LABELS[k], "tau": self.taus[k], "w_I": self.weights[k][0], "w_D": self.weights[k][1]}
                for k in PROFILE_ORDER]


@lru_cache(maxsize=4)
def load_config(results: Path = RESULTS, policy_src: Path = POLICY_SRC) -> PaceConfig:
    v2 = _checked(results / "frozen_v2.json", results / "frozen_v2.sha256")
    a = v2["cross_dataset_a"]
    if a["primary_risk_score"] != "bkt":
        raise RuntimeError(f"ARIA-Pace: frozen primary_risk_score is {a['primary_risk_score']!r}, expected 'bkt'")
    val = _checked(results / "frozen_val.json", results / "frozen_val.sha256", expect=a["frozen_val_sha256"])
    taus = val["configs"]["main"]["frozen"]["taus"]["bkt"]
    platt = val["platt"]["bkt"]
    params = json.loads((results / "baselines_val.json").read_text())["selection"]["bkt_pooled_params"]
    weights = _profile_weights(policy_src)
    if set(taus) != set(PROFILE_ORDER) or set(weights) != set(PROFILE_ORDER):
        raise RuntimeError("ARIA-Pace: frozen profiles are not quiet / balanced / supportive")
    k_min, k_max = _decision_window(policy_src)
    return PaceConfig(
        k_min=k_min, k_max=k_max, taus={k: float(taus[k]) for k in PROFILE_ORDER}, weights={k: weights[k] for k in PROFILE_ORDER},
        platt=(float(platt["intercept"]), float(platt["slope"])), params={k: float(params[k]) for k in ("prior", "learn", "guess", "slip")},
        sources={"frozen_v2.json": _sha(results / "frozen_v2.json"), "frozen_val.json": _sha(results / "frozen_val.json")})


# ------------------------------------------------------------------------------------ the tracker
def raw_risk(obs: list, par: dict) -> float:
    """P(no mastery in opportunities len(obs)+1 .. HORIZON | obs): BKT forward filter + exact DP (aria_pace.baselines.BKT.score).
    A student whose last MASTERY_RUN answers are all right has mastered: risk 0. After HORIZON observations the window is empty."""
    if not obs:
        raise ValueError("raw_risk needs at least one observation")
    run = 0
    for x in reversed(obs):
        if x != 1:
            break
        run += 1
    if run >= MASTERY_RUN:
        return 0.0
    pk = par["prior"]
    for x in obs:
        e1 = (1 - par["slip"]) if x == 1 else par["slip"]
        e0 = par["guess"] if x == 1 else (1 - par["guess"])
        pk = pk * e1 / (pk * e1 + (1 - pk) * e0)
        pk_post = pk
        pk = pk + (1 - pk) * par["learn"]
    d = [[0.0] * MASTERY_RUN for _ in range(2)]            # d[latent known?][current correct run]
    d[1][run], d[0][run] = pk_post, 1 - pk_post
    for _ in range(len(obs) + 1, HORIZON + 1):
        nd = [[0.0] * MASTERY_RUN for _ in range(2)]
        for ls in (0, 1):
            for r in range(MASTERY_RUN):
                m = d[ls][r]
                if m == 0:
                    continue
                for ls2, pt in ((1, 1.0 if ls == 1 else par["learn"]), (0, 0.0 if ls == 1 else 1 - par["learn"])):
                    if pt == 0:
                        continue
                    pc = (1 - par["slip"]) if ls2 == 1 else par["guess"]
                    if r + 1 < MASTERY_RUN:
                        nd[ls2][r + 1] += m * pt * pc
                    nd[ls2][0] += m * pt * (1 - pc)
        d = nd
    return float(sum(map(sum, d)))


def calibrate(raw: float, platt: tuple) -> float:
    """aria_pace.policy.platt_apply for one score."""
    s = min(max(raw, EPS), 1 - EPS)
    z = platt[0] + platt[1] * math.log(s / (1 - s))
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class PaceDecision:
    profile: str
    tau: float
    risk: Optional[float]           # calibrated risk after this turn's update; None before any verified step
    n_obs: int
    updated: bool                   # did this turn's verdict update the tracker
    verified_wrong: bool
    in_window: bool                 # k_min <= attempts <= k_max: the gate applies
    speak: bool                     # verified wrong step AND (risk >= tau inside the window, or outside it: the old rule)
    reason: str                     # short identifier, safe for the log


class PaceTracker:
    """One per session. update(status) from the SymPy verdict; risk() is the Platt-calibrated BKT risk.
    decide(): inside attempts K_MIN..K_MAX, speak on a verified wrong step only if risk >= tau; outside, speak on every one."""

    def __init__(self, cfg: PaceConfig):
        self.cfg = cfg
        self.obs: list = []

    def reset(self):
        self.obs = []

    def update(self, status: str) -> bool:
        if status in RIGHT:
            self.obs.append(1)
        elif status in WRONG:
            self.obs.append(0)
        else:
            return False
        return True

    def risk(self) -> Optional[float]:
        if not self.obs:
            return None
        return calibrate(raw_risk(self.obs, self.cfg.params), self.cfg.platt)

    def decide(self, status: str, profile: str) -> PaceDecision:
        """Update from this turn's verdict, then apply the rule (see the class doc)."""
        updated = self.update(status)
        return self.snapshot(profile, status, updated)

    def snapshot(self, profile: str, status: str = "none", updated: bool = False) -> PaceDecision:
        tau, risk = self.cfg.taus[profile], self.risk()
        n = len(self.obs)
        wrong = status in WRONG and risk is not None
        inside = self.cfg.k_min <= n <= self.cfg.k_max
        speak = wrong and (risk >= tau if inside else True)         # outside 3..9 the old rule: always speak on a wrong step
        if wrong:
            reason = ("wrong_risk_ge_tau" if speak else "wrong_risk_lt_tau") if inside else (
                "wrong_before_window" if n < self.cfg.k_min else "wrong_after_window")
        else:
            reason = "right_step" if status in RIGHT else "no_verdict" if status == "none" else "restate"
        return PaceDecision(profile, tau, risk, n, updated, wrong, inside, speak, reason)
