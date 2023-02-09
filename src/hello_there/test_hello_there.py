from src.hello_there import hello_there as ht

def test_hello_there_exists():
    # Check that it exists
    res = ht.greeting("Hello there")
    assert res

def test_hello_there_returns_correct():
    # Check that it returns "General Kenobi" if provided with correct input
    res = ht.greeting("Hello there")
    assert res == "General Kenobi"

def test_hello_there_returns_None_on_invalid_input():
    # Check that it returns error if given anything else
    res = ht.greeting(1)
    assert res != "General Kenobi"