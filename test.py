from pprint import pprint

from game_components import Game
from human_player import HumanPlayer
from rule_player import RuleBasedPlayer

G = Game(
    RuleBasedPlayer.from_json("../human_rules/AC01010.json"),
    RuleBasedPlayer.from_json("../human_rules/AC01010.json"),
    {},
)
pprint(G.run())
