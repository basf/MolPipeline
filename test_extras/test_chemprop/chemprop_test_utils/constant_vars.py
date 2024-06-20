"""Variables that are used in multiple tests."""

# These are model parameters which are copied by value, but are too complex to check for equality.
# Thus, for these model parameters, only the type is checked.
NO_IDENTITY_CHECK = [
    "model__agg",
    "model__message_passing",
    "model",
    "model__predictor",
    "model__predictor__criterion",
    "model__predictor__output_transform",
]
