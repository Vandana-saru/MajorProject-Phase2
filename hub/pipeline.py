"""
PRAHARI — the reference pipeline. One window in, a full verdict out.

    window -> features -> class + probabilities   (Layer 1)
                       -> severity                (Layer 2)
                       -> Signal Trust Index
           -> matched restoration                 (Layer 3)
           -> heart rate + gate decision          (Layer 4)

This is the module the dashboard, the replay tool and the evaluation scripts
all import. If you find yourself re-implementing any of this somewhere else,
import from here instead — divergence between the dashboard's pipeline and the
evaluation's pipeline is how a team ends up publishing numbers it cannot
reproduce on stage.
"""

import os

import numpy as np
import joblib

from .contracts import CLASSES, Verdict, signal_trust_index, FS
from .features import extract
from . import restore as restore_mod
from . import hr as hr_mod
from .gate import DEFAULT_POLICY, apply_gate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "prahari_model.joblib")


class Prahari:
    def __init__(self, model_path=DEFAULT_MODEL, policy=DEFAULT_POLICY,
                 calibration=None):
        bundle = joblib.load(model_path)
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.severity_models = bundle.get("severity", {})
        self.classes = list(bundle.get("classes", self.clf.classes_))
        self.policy = policy
        self.calibration = calibration

    # ------------------------------------------------------------- layers 1+2
    def diagnose(self, window):
        """window (preprocessed) -> Verdict"""
        f = extract(window).reshape(1, -1)
        fs_ = self.scaler.transform(f)

        proba_raw = self.clf.predict_proba(fs_)[0]
        cls = str(self.clf.classes_[int(np.argmax(proba_raw))])

        # re-order probabilities into the frozen CLASSES order
        proba = np.zeros(len(CLASSES), dtype=np.float32)
        for p, c in zip(proba_raw, self.clf.classes_):
            proba[CLASSES.index(str(c))] = p

        if cls == "Clean" or cls not in self.severity_models:
            severity = 0.0
        else:
            severity = float(np.clip(
                self.severity_models[cls].predict(fs_)[0], 0.0, 1.0))

        return Verdict(cls, proba, severity, signal_trust_index(proba, severity))

    # --------------------------------------------------------------- layer 3
    def restore(self, window, verdict, **kw):
        return restore_mod.restore(window, verdict, **kw)

    # --------------------------------------------------------------- layer 4
    def heart_rate(self, window):
        return hr_mod.estimate(window)

    # ------------------------------------------------------------ full chain
    def process(self, window, do_restore=True):
        """
        Returns a dict with everything the dashboard and the logger need.
        `hr_raw` is what an ungated device would have shown; `hr_shown` is
        what Prahari actually releases. The difference between those two
        columns, accumulated over a session, is the project's whole argument.
        """
        window = np.asarray(window, dtype=np.float64)
        v = self.diagnose(window)

        restored = self.restore(window, v) if (do_restore and v.cls != "Clean") else window
        hr_raw, agree_raw = self.heart_rate(window)
        hr_restored, _ = self.heart_rate(restored)

        # gate on the restored signal when restoration ran, since that is what
        # the device would actually be reading from
        hr_candidate = hr_restored if do_restore else hr_raw
        decision, hr_shown = apply_gate(hr_candidate, v.sti, self.policy)

        return dict(
            verdict=v,
            window=window,
            restored=restored,
            hr_raw=hr_raw,
            hr_restored=hr_restored,
            hr_shown=hr_shown,
            hr_agreement=agree_raw,
            decision=decision,
            **v.to_dict(),
        )
