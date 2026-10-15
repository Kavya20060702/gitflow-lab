from app import greet, welcome


def test_greet():
    assert greet("Kavya") == "Hello, Kavya!"


def test_welcome():
    assert welcome("Kavya") == "Welcome to Gitflow, Kavya!"
