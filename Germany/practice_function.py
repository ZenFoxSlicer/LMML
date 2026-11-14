# GERMANY'S TRAINING BOX: a box we are allowed to open for practice.
# This file contains its known rules so we can check whether our investigator works.
# The real exam uses a different box. Our main.py does not import this file.
#
# Read f() below to see the training box's behaviour:
# - Start with 2 on the screen.
# - Add the separate changes caused by knobs 1, 3, and 5.
# - Add a shared effect from knobs 3 and 5.
# - Add 1.5 when knobs 1 and 4 together reach the switch limit.
# - Knob 2 does nothing in this training box.
#
# TRUTH is the examiner's answer sheet. evaluate() returns a rounded screen reading.
# Upload only main.py, not this known training formula.

"""
Practice black box for "Recover the Black Box" — PUBLIC.

`python3 evaluation_script.py main.py --eval-type train` runs your main.py
against this function and scores it, so you can test your strategy as often
as you like. Same family as the graded function — main effects, one pairwise
interaction, one hard step — but different variables, coefficients and
boundary. A main.py that hardcodes anything from this file will fail the
real run; one that discovers the structure will not.
"""
from typing import Dict, List, Sequence

VAR_NAMES: List[str] = ["x1", "x2", "x3", "x4", "x5"]

# The evaluator uses this dictionary to check the solver's structural claims.
TRUTH: Dict[str, object] = {
    "intercept": 2.0,
    "linear": {"x1": 1.5, "x2": 0.0, "x3": 0.4, "x4": 0.0, "x5": -0.6},
    "interaction": {
        "variables": ("x3", "x5"),
        "form": "centered_product",          # c * (x3 - 0.5) * (x5 - 0.5)
        "coefficient": -4.0,
        "center": 0.5,
    },
    "discontinuity": {
        "gate_variables": ("x1", "x4"),
        "boundary": {"x1": 1.0, "x2": 0.0, "x3": 0.0, "x4": 1.0, "x5": 0.0},
        "threshold": 1.7,                    # active where x1 + x4 >= 1.7
        "jump": 1.5,
    },
    "relevant": ["x1", "x3", "x4", "x5"],
    "irrelevant": ["x2"],
}


# Tell the evaluator whether a point lies on the side with the extra jump.
def gate_active(x: Sequence[float]) -> bool:
    return (x[0] + x[3]) >= 1.7


def f(x: Sequence[float]) -> float:
    # Unpack the five dial settings. _x2 is unused: that dial is decorative here.
    x1, _x2, x3, x4, x5 = x
    # Start at 2.0, then add the separate effects of x1, x3, and x5.
    y = 2.0 + 1.5 * x1 + 0.4 * x3 - 0.6 * x5
    # These two knobs work together. If either is at 0.5, the product is zero.
    # Like testing a brightness dial with the lamp off, a middle-only test hides it.
    y += -4.0 * (x3 - 0.5) * (x5 - 0.5)
    # Once x1 + x4 reaches 1.7, add 1.5 all at once: this is the jump.
    if (x1 + x4) >= 1.7:
        y += 1.5
    return y


# Round readings to six decimal places, matching what the oracle reveals.
def evaluate(x: Sequence[float]) -> float:
    return round(f(x), 6)


# Return the same practice formula as text for the evaluation report.
def expression() -> str:
    return ("2.0 + 1.5*x1 + 0.4*x3 - 0.6*x5 - 4.0*(x3-0.5)*(x5-0.5) "
            "+ (1.5 if x1 + x4 >= 1.7 else 0.0)")
