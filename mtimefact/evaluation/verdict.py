"""Paper A.3.4: exact binary verdict accuracy, no model judge required."""


def normalize_label(value):
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "supported", "supports", "real", "真实", "真", "支持"}:
            return True
        if text in {"false", "refuted", "refutes", "fake", "虚假", "假", "反驳"}:
            return False
    raise ValueError("Expected an explicit binary label; free-form text/unknown/NEI is not a binary verdict")


def evaluate(gold, prediction, judge=None, options=None):
    label = normalize_label(gold.get("label"))
    result = {"id": gold["id"], "stage": "verdict", "gold_label": label,
              "predicted_label": None, "score": 0.0, "status": "missing_output"}
    if prediction is None:
        return result
    try:
        predicted = normalize_label(prediction.get("predicted_label"))
    except ValueError as exc:
        return dict(result, status="invalid_output", reason=str(exc), raw_predicted_label=prediction.get("predicted_label"))
    result.update(predicted_label=predicted, score=float(label == predicted), status="ok")
    return result
