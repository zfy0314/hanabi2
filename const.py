from enum import IntEnum
from itertools import product
from typing import Dict, List, Union

import numpy as np


class AugmentedIntEnum(IntEnum):
    @classmethod
    def get(cls, value: Union[str, int]) -> IntEnum:
        return cls(value) if isinstance(value, int) else getattr(cls, value)

    @classmethod
    def create(cls, values: Dict[Union[str, int], float]) -> np.ndarray:
        vec = np.zeros((len(cls),))
        for field, value in values.items():
            vec[cls.get(field).value] = value
        return vec

    @classmethod
    def create2d(cls, values2d: List[Dict[Union[str, int], float]]) -> np.ndarray:
        vec = np.zeros((len(values2d), len(cls)))
        for i, values in enumerate(values2d):
            for field, value in values.items():
                vec[i, cls.get(field).value] = value
        return vec

    @classmethod
    def onehot(cls, field: Union[str, int]) -> np.ndarray:
        vec = np.zeros((len(cls),))
        vec[cls.get(field).value] = 1
        return vec

    @classmethod
    def binary(cls, fields: List[Union[str, int]]) -> np.ndarray:
        vec = np.zeros((len(cls),))
        for field in fields:
            vec[cls.get(field).value] = 1
        return vec

    @classmethod
    def binary2d(cls, fields2d: List[List[str]]) -> np.ndarray:
        vec = np.zeros(len(fields2d), len(cls))
        for i, fields in enumerate(fields2d):
            for field in fields:
                vec[i, cls.get(field).value] = 1
        return vec

    @classmethod
    def to_fields(cls, vec: np.ndarray) -> List[str]:
        return [cls(x).name for x in vec.nonzero()[0]]

    @classmethod
    def to_dict(cls, vec: np.ndarray) -> Dict[str, float]:
        return {attr.name: vec[attr] for attr in cls if vec[attr] > 1e-12}


Suit = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5]

Color = IntEnum("Color", ["blue", "green", "red", "white", "yellow"], start=0)
Rank = IntEnum("Rank", ["one", "two", "three", "four", "five"], start=1)
Action = AugmentedIntEnum(
    "Action",
    [
        "{}_{}".format(*cfg)
        for cfg in product(["play", "discard", "hint_color"], range(5))
    ]
    + ["hint_rank_{}".format(rank) for rank in Rank],
    start=0,
)
# [deprecated] hint_color_1 => hint the color of partner's 1st card, not hint blue

GameStateField = AugmentedIntEnum(
    "GameStateField",
    # board (5 * 6 = 30)
    ["board_{}_{}".format(c.name, i) for c, i in product(Color, range(6))]
    # hint tokens (6)
    + ["hint_8", "hint_some", "hint_3", "hint_2", "hint_1", "hint_0"]
    # strike tokens (3)
    + ["strike_3", "strike_2", "strike_1"]
    # partner hand basic (5 * (4 + 10) = 70)
    + [
        "partner_hand_{}_{}".format(*cfg)
        for cfg in product(
            range(5),
            ["playable", "useless", "critical", "five"]
            + ["is_{}".format(c.name) for c in Color]
            + ["is_{}".format(r.value) for r in Rank],
        )
    ]
    # partner hand extra (2 * 10 + 5 = 25)
    + [
        "partner_hand_same_{}_{}_{}".format(*cfg)
        for cfg in product(["color", "rank"], range(5), range(5))
        if cfg[1] < cfg[2]
    ]
    + ["partner_hand_removed_{}".format(i) for i in range(5)]
    # knowledge basic (2 * 5 * 11 = 110)
    + [
        "{}_knowledge_{}_{}".format(*cfg)
        for cfg in product(
            ["partner", "self"],
            range(5),
            [
                "definitely_playable",
                "definitely_not_playable",
                "definitely_useless",
                "definitely_not_useless",
                "definitely_critical",
                "definitely_not_critical",
                "five",
                "just_hinted",
                "known_color",
                "known_rank",
                "fully_known",
            ],
        )
    ],
    # # knowledge extra (2 * 10 = 20)
    # + [
    #     "{}_knowledge_same_{}_{}_{}".format(*cfg)
    #     for cfg in product(["partner", "self"], ["color", "rank"], range(5), range(5))
    #     if cfg[2] < cfg[3]
    # ],
    start=0,
)
GoalStateField = AugmentedIntEnum(
    "GoalStateField",
    # play/discard own card (2 * 5 = 10)
    ["{}_{}".format(*cfg) for cfg in product(["play", "discard"], range(5))]
    # strategic hints (3 * 5 = 15)
    + [
        "hint_{}_{}".format(*cfg)
        for cfg in product(["playable", "useless", "critical"], range(5))
    ],
    start=0,
)
