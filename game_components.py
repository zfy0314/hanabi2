import random
import warnings
from collections import UserList, namedtuple
from copy import deepcopy
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from const import Color, GameStateField, Rank, Suit
from player import Player, Spectator

GameSummary = namedtuple(
    "GameSummary", ["score", "turns", "hints", "strikes", "actions"]
)


class Card:
    def __init__(self, color: Union[Color, int], rank: Union[Rank, int]):
        self.color = color if isinstance(color, Color) else Color(color)
        self.rank = rank if isinstance(rank, Rank) else Rank(rank)
        self.possibilities = list(product(Color, Suit))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __iter__(self):
        yield self.color
        yield self.rank

    @property
    def all_possibilities(self):
        return self.possibilities

    @property
    def known_color(self):
        return all(x[0] == self.color for x in self.possibilities)

    @property
    def known_rank(self):
        return all(x[1] == self.rank for x in self.possibilities)

    @property
    def fully_known(self):
        return self.known_color and self.known_rank

    def remove_possibilities(self, trash: List[Tuple[int, int]]):
        for card in trash:
            try:
                self.possibilities.remove(tuple(card))
            except ValueError:
                pass

    def hint_color(self, color: Color):
        if color == self.color:
            self.possibilities = [x for x in self.possibilities if x[0] == color]
        else:
            self.possibilities = [x for x in self.possibilities if x[0] != color]

    def hint_rank(self, rank: Rank):
        if rank == self.rank:
            self.possibilities = [x for x in self.possibilities if x[1] == rank]
        else:
            self.possibilities = [x for x in self.possibilities if x[1] != rank]

    def decrement(self, card):
        try:
            self.possibilities.remove(tuple(card))
        except ValueError:
            pass


class Hand(UserList):
    just_hinted = set()

    def retrieve(self, index: int) -> Card:
        card = self.data[index]
        self.data = self.data[:index] + self.data[index + 1 :]
        return card

    def decrement(self, card: Card):
        for card2 in self.data:
            card2.decrement(card)

    def hint_color(self, color: Union[int, Color]):
        if not isinstance(color, Color):
            color = Color(color)
        self.just_hinted = set()
        for i, card in enumerate(self.data):
            card.hint_color(color)
            if card.color == color:
                self.just_hinted.add(i)

    def hint_rank(self, rank: Union[int, Rank]):
        if not isinstance(rank, Rank):
            rank = Rank(rank)
        self.just_hinted = set()
        for i, card in enumerate(self.data):
            card.hint_rank(rank)
            if card.rank == rank:
                self.just_hinted.add(i)

    def reset_hints(self):
        self.just_hinted = set()

    def card_count_partner(self, partner_hand: "Hand") -> "Hand":

        updated_hand = deepcopy(self)
        for card in partner_hand:
            updated_hand.decrement(card)
        return updated_hand


class Deck:
    def __init__(self, deck: Optional[List[Tuple[int, int]]] = None, seed: int = 42):
        if deck is None:
            self.deck = [Card(*cfg) for cfg in product(Color, Suit)]
            random.seed(seed)
            random.shuffle(self.deck)
        else:
            self.deck = [Card(*cfg) for cfg in deck]
        self.pointer = 0

    def draw_hands(self) -> Tuple[Hand, Hand]:
        hand1 = Hand([self.draw_card() for _ in range(5)])
        hand2 = Hand([self.draw_card() for _ in range(5)])
        return (hand1, hand2)

    def draw_card(self) -> Card:
        card = self.deck[self.pointer]
        self.pointer += 1
        return card

    @property
    def isempty(self) -> bool:
        return self.pointer >= len(self.deck)

    def __len__(self) -> int:
        return len(self.deck) - self.pointer


class Trash:
    def __init__(self):
        self.trash = []

    def __iter__(self):
        for item in self.trash:
            yield item

    def add(self, card: Card):
        self.trash.append((card.color, card.rank))

    def has_all(self, card: Tuple[int, int]) -> bool:
        return self.trash.count(card) == Suit.count(card[1])

    def missing_one(self, card: Tuple[int, int]) -> bool:
        return self.trash.count(card) == Suit.count(card[1]) - 1


class Board:
    def __init__(self):
        self.board = {color: 0 for color in Color}
        self.get_color = self.board.get

    def play(self, card: Card) -> bool:
        if self.board[card.color] == card.rank - 1:
            self.board[card.color] += 1
            return True
        else:
            return False

    @property
    def score(self):
        return sum(self.board.values())


class Game:
    hint_mapping = {
        0: "hint_0",
        1: "hint_1",
        2: "hint_2",
        3: "hint_3",
        8: "hint_8",
        "default": "hint_some",
    }

    def __init__(
        self,
        player1: Player,
        player2: Player,
        spectators: Dict[Spectator, int] = dict(),
        config: Dict[str, dict] = {"deck": {"seed": 42}},
    ):
        self.players = [player1, player2]
        self.spectators = spectators
        self.reset(config)

    def reset(self, config: Dict[str, dict] = {}):
        self.turns = 0
        self.hints = 8
        self.strikes = 3
        self.current_player_id = 0
        self.turns_after_empty_deck = 0
        self.deck = Deck(**config.get("deck", {}))
        self.board = Board()
        self.trash = Trash()
        self.actions = []
        self.hands = self.deck.draw_hands()

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_id]

    @property
    def current_hand(self) -> Hand:
        return self.hands[self.current_player_id]

    @property
    def partner_hand(self) -> Hand:
        return self.hands[1 - self.current_player_id]

    def is_playable(self, card: Union[Card, Tuple[int, int]], **kwargs) -> bool:
        color, rank = tuple(card)
        return self.board.get_color(color) == rank - 1

    def is_useless(
        self,
        card: Union[Card, Tuple[int, int]],
        alternatives: List[tuple] = [],
        **kwargs
    ) -> bool:
        color, rank = tuple(card)
        return (
            any(self.trash.has_all((color, r)) for r in range(1, rank))
            or self.board.get_color(color) >= rank
            or alternatives.count(card) >= 2
        )

    def is_critical(self, card: Union[Card, Tuple[int, int]], **kwargs) -> bool:
        color, rank = tuple(card)
        return (not self.is_useless(card)) and self.trash.missing_one((color, rank))

    def is_five(self, card: Union[Card, Tuple[int, int]], **kwargs) -> bool:
        color, rank = tuple(card)
        return rank == 5

    @property
    def game_state(self) -> np.ndarray:

        fields = []

        # board
        for color in Color:
            fields.append("board_{}_{}".format(color.name, self.board.get_color(color)))

        # hint & strike tokens
        fields.append(self.hint_mapping.get(self.hints, self.hint_mapping["default"]))
        if self.strikes != 0:
            fields.append("strike_{}".format(self.strikes))

        # partner hand
        partner_hand = "partner_hand_{}_{}"
        for i, card in enumerate(self.partner_hand):
            # basic
            for attr in ["playable", "useless", "critical", "five"]:
                if getattr(self, "is_{}".format(attr))(
                    card, alternatives=self.partner_hand
                ):
                    fields.append(partner_hand.format(i, attr))
            fields.append(partner_hand.format(i, "is_{}".format(card.color.name)))
            fields.append(partner_hand.format(i, "is_{}".format(card.rank.value)))
            # extra
            for j, card2 in enumerate(self.partner_hand[:i]):
                if card2.color == card.color:
                    fields.append("partner_hand_same_color_{}_{}".format(j, i))
                if card2.rank == card.rank:
                    fields.append("partner_hand_same_rank_{}_{}".format(j, i))
        if self.actions != [] and self.actions[-1][1][0] in {"p", "d"}:
            fields.append("partner_hand_removed_{}".format(self.actions[-1][1][-1]))

        # partner & self knowledge
        for name, hand in [
            ("partner", self.partner_hand),
            ("self", self.current_hand.card_count_partner(self.partner_hand)),
        ]:
            for i, card in enumerate(hand):
                field_name = "{}_knowledge_{}_".format(name, i)
                for attr in ["playable", "useless", "critical"]:
                    alternatives = [card for card in hand if card.fully_known]
                    func = lambda x: getattr(self, "is_{}".format(attr))(
                        x, alternatives=alternatives
                    )
                    if all(func(x) for x in card.all_possibilities):
                        fields.append(field_name + "definitely_" + attr)
                    elif all(not func(x) for x in card.all_possibilities):
                        fields.append(field_name + "definitely_not_" + attr)
                if all(self.is_five(x) for x in card.all_possibilities):
                    fields.append(field_name + "five")
                for attr in ["known_color", "known_rank", "fully_known"]:
                    if getattr(card, attr):
                        fields.append(field_name + attr)
                """
                field_name = "{}_knowledge_same".format(name)
                for j, card2 in enumerate(hand[:i]):
                    if all(
                        card.known_color, card2.known_color, card2.color == card.color
                    ):
                        fields.append(
                            "_".join(field_name, "same_color", str(j), str(i))
                        )
                    if all(card.known_rank, card2.known_rank, card2.rank == card.rank):
                        fields.append("_".join(field_name, "same_rank", str(j), str(i)))
                """
            for i in hand.just_hinted:
                fields.append("{}_knowledge_{}_just_hinted".format(name, i))

        return GameStateField.binary(fields).astype(np.uint16)

    def turn(self) -> bool:

        if self.strikes == 0:
            warnings.warn("run out of strikes")
            return False
        if self.turns_after_empty_deck >= 2:
            warnings.warn("run out of cards")
            return False
        elif self.deck.isempty:
            self.turns_after_empty_deck += 1

        old_state = self.game_state

        cnt = 10

        while cnt >= 0:
            action = self.current_player.get_action(
                self if hasattr(self.current_player, "raw_state") else old_state
            )
            cnt -= 1

            if action.name.startswith("hint") and self.hints == 0:
                warnings.warn(
                    "{} tries to hint when there are no hint tokens left".format(
                        self.current_player.name
                    )
                )
                continue
            if action.name.startswith("discard") and self.hints == 8:
                warnings.warn(
                    "{} tries to discard when there are 8 hint tokens".format(
                        self.current_player.name
                    )
                )
                continue
            if action.name in {"discard_4", "play_4"} and len(self.current_hand) == 4:
                warnings.warn(
                    "{} tries to act on a non-existing card".format(
                        self.current_player.name
                    )
                )
                continue
            # TODO only hint existing cards
            break

        if cnt < 0:
            warnings.warn("No legal actions found after 10 attempts")
            return False

        self.act(action)

        self.turns += 1
        self.players[0].inform(
            self if hasattr(self.players[0], "raw_state") else self.game_state
        )
        for spectator, player_id in self.spectators.items():
            if player_id == self.current_player_id:
                spectator.observe_action(old_state, action, self.game_state)
        self.current_hand.reset_hints()
        self.current_player_id = 1 - self.current_player_id
        self.players[1].inform(
            self if hasattr(self.players[1], "raw_state") else self.game_state
        )
        return True

    def act(self, action: int) -> None:

        index = action % 5
        if action.name.startswith("play") or action.name.startswith("discard"):
            # play and discard
            card = self.current_hand.retrieve(index)
            if action.name.startswith("play"):
                if self.board.play(card):
                    self.hints = min(8, self.hints + (card.rank == 5))
                else:
                    self.trash.add(card)
                    self.strikes -= 1
            else:
                self.hints += 1
                self.trash.add(card)
            self.current_hand.decrement(card)
            self.partner_hand.decrement(card)
            if not self.deck.isempty:
                new_card = self.deck.draw_card()
                new_card.remove_possibilities(self.trash)
                for color in Color:
                    new_card.remove_possibilities(
                        [(color, i) for i in range(1, self.board.get_color(color) + 1)]
                    )
                new_card.remove_possibilities(
                    [tuple(card) for card in self.current_hand if card.fully_known]
                )
                self.current_hand.append(new_card)
            action_name, index = action.name.split("_")
            self.actions.append(
                (
                    self.current_player_id,
                    "{} {} {} at {}".format(
                        action_name, card.color.name, card.rank.name, index
                    ),
                )
            )
        else:
            # hint color / rank
            self.hints -= 1
            # card = self.partner_hand[index]
            # attr = getattr(card, action.name[5:-2])
            attr = int(action.name[-1])
            action_name = action.name[:-2]
            getattr(self.partner_hand, action_name)(attr)
            self.actions.append(
                (
                    self.current_player_id,
                    action.name.replace("_", " "),
                )
            )

        return None

    def run(self) -> GameSummary:
        while self.turn():
            pass
        return GameSummary(
            self.score, self.turns, self.hints, self.strikes, self.actions
        )

    @property
    def score(self):
        return self.board.score

    def export(self) -> dict:
        # TODO
        return {
            "players": [self.players[0].name, self.players[1].name],
            "deck": [tuple(x) for x in self.deck],
            "actions": self.actions,
        }
