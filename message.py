class Message:
    def __init__(self, author, issue: int, magnitude: int):
        self.author = author # Agent
        self.issue = issue
        self.magnitude = magnitude