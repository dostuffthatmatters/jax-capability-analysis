def assert_similar_result(
    label: str,
    expected: float | int,
    actual: float | int,
    precision: int = 2,
) -> None:
    assert abs((actual / expected) - 1) < 1 / pow(10, precision), (
        f"{label} is not similar to expected value: "
        + f"expected {expected}, got {actual}"
    )
