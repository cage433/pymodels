from datetime import date, timedelta


class Day(date):
    def __init__(self, y: int, m: int, d: int):
        date.__init__(y, m, d)

    def __add__(self, other):
        daysdelta = timedelta(other) if type(other) == int else other
        d = date.__add__(self, daysdelta)
        return Day(d.year, d.month, d.day)

    def __sub__(self, other):
        if type(other) == int:
            daysdelta = timedelta(other)
            d = date.__sub__(self, daysdelta)
            return Day(d.year, d.month, d.day)
        else:
            return date.__sub__(self, other)

    def time_since(self, d: date) -> float:
        diff = date.__sub__(self, d)
        return diff.days / 365.25
