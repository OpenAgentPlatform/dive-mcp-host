def filter_empty_dict(inpt: list) -> list:
    """Filter out empty dict values in list."""
    return list(filter(lambda x: x != {}, inpt))
