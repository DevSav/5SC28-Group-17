from src.config import MAX_VOLTAGE, MIN_VOLTAGE


def limit_voltage(voltage):
    """Keep the voltage inside the allowed range."""
    if voltage > MAX_VOLTAGE:
        return MAX_VOLTAGE
    if voltage < MIN_VOLTAGE:
        return MIN_VOLTAGE
    return voltage

