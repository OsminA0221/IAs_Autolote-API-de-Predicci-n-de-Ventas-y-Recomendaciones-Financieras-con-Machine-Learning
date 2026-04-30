from datetime import datetime


def validate_date(date_str):
    """Parse a YYYY-MM-DD string into a date."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError("Formato de fecha invalido. Usa YYYY-MM-DD.")


def validate_positive_int(value, name):
    """
    Ensure the value is a positive integer.
    Returns the integer (converted) or raises ValueError.
    """
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} debe ser un entero positivo.")
    if value_int <= 0:
        raise ValueError(f"{name} debe ser un entero positivo.")
    return value_int
