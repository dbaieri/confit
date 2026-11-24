import confort


@confort.command(help="Add two numbers")
def add_numbers(a: int, b: int = 2) -> int:
    return a + b
