from random import Random
from enum import Enum
from typing import TypeVar, Type, List, Iterable, cast

TEnum = TypeVar("TEnum", bound=Enum)


class PimpedRandom(Random):
    def enum_choice(self, enum_cls: Type[TEnum]) -> TEnum:
        members: List[TEnum] = list(cast(Iterable[TEnum], enum_cls))
        return self.choice(members)
