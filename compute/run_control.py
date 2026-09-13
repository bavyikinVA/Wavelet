"""Cooperative cancellation: a control signal must bypass backend fallback."""
from threading import Event


class RunCancelled(BaseException):
    """Not an algorithm failure: never retry the same cancelled work on CPU."""


class RunControl:
    def __init__(self):
        self.event = Event()

    def reset(self):
        self.event.clear()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise RunCancelled('Расчёт отменён пользователем')
