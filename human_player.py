from time import sleep

from const import Action, Color
from player import Player

"""
====================================================================================
Player 0
------------------------------------------------------------------------------------
 # hints:   [5 / 8]        # strikes: [2 / 3]       # Deck: [30 / 50]
 Last action: dicard 4
------------------------------------------------------------------------------------
 Partner's hand and knowledge                          (card direction: old <-- new)

+--------------+ +--------------+ +--------------+ +--------------+ +--------------+
| Card #0  *** | | Card #1  *** | | Card #2  *** | | Card #3  *** | | Card #4  *** |
| Actual:      | | Actual:      | | Actual:      | | Actual:      | | Actual:      |
|   <yellow 5> | |   <yellow 5> | |   <yellow 5> | |   <yellow 5> | |   <yellow 5> |
| Knowledge:   | | Knowledge:   | | Knowledge:   | | Knowledge:   | | Knowledge:   |
|    B G R W Y | |    B G R W Y | |    B G R W Y | |    B G R W Y | |    B G R W Y |
| 1: * * * * * | | 1: * * * * * | | 1: * * * * * | | 1: * * * * * | | 1: * * * * * |
| 2: * * * * * | | 2: * * * * * | | 2: * * * * * | | 2: * * * * * | | 2: * * * * * |
| 3: * * * * * | | 3: * * * * * | | 3: * * * * * | | 3: * * * * * | | 3: * * * * * |
| 4: * * * * * | | 4: * * * * * | | 4: * * * * * | | 4: * * * * * | | 4: * * * * * |
| 5: * * * * * | | 5: * * * * * | | 5: * * * * * | | 5: * * * * * | | 5: * * * * * |
+--------------+ +--------------+ +--------------+ +--------------+ +--------------+

------------------------------------------------------------------------------------
 Board:

  ----------------------------------------
  blue |  green |    red |  white | yellow
     0 |      1 |      2 |      1 |      4

------------------------------------------------------------------------------------
self hand and knowledge                                (card direction: old <-- new)

+--------------+ +--------------+ +--------------+ +--------------+ +--------------+
| Card #0  *** | | Card #1  *** | | Card #2  *** | | Card #3  *** | | Card #4  *** |
| Actual:      | | Actual:      | | Actual:      | | Actual:      | | Actual:      |
|   <yellow 5> | |   <yellow 5> | |   <yellow 5> | |   <yellow 5> | |   <yellow 5> |
| Knowledge:   | | Knowledge:   | | Knowledge:   | | Knowledge:   | | Knowledge:   |
|    B G R R Y | |    B G R W Y | |    B G R W Y | |    B G R W Y | |    B G R W Y |
| 1: * * * * * | | 1: * * * * * | | 1: * * * * * | | 1: * * * * * | | 1: * * * * * |
| 2: * * * * * | | 2: * * * * * | | 2: * * * * * | | 2: * * * * * | | 2: * * * * * |
| 3: * * * * * | | 3: * * * * * | | 3: * * * * * | | 3: * * * * * | | 3: * * * * * |
| 4: * * * * * | | 4: * * * * * | | 4: * * * * * | | 4: * * * * * | | 4: * * * * * |
| 5: * * * * * | | 5: * * * * * | | 5: * * * * * | | 5: * * * * * | | 5: * * * * * |
+--------------+ +--------------+ +--------------+ +--------------+ +--------------+

====================================================================================
Sample commands:
    play 0 | Play 0 | P0 | P 0 | 11
    hint_color 4 | HC4 | HC 4 | hint color 4 | 5
------------------------------------------------------------------------------------
Your Action:
"""


interface = "\n".join(
    [
        "",
        "",
        "=" * 84,
        " Player {player_id}",
        "-" * 84,
        "        ".join(
            [
                " # hints:   [{hints} / 8]",
                "# strikes: [{strikes} / 3]",
                "# deck: [{deck} / 50]",
            ]
        ),
        " Last action: {last_action}",
        "-" * 84,
        " Partner's hand and knowledge" + " " * 26 + "(card direction: old <-- new)",
        "",
        "+--------------+ " * 5,
        " ".join(
            [
                "| Card #" + str(i) + "  {partner_just_hinted_" + str(i) + "} |"
                for i in range(5)
            ]
        ),
        "| Actual:      | " * 5,
        " ".join(
            [
                "|  < {partner_color_"
                + str(i)
                + ":<6} {partner_rank_"
                + str(i)
                + "}> |"
                for i in range(5)
            ]
        ),
        "| Knowledge:   | " * 5,
        "|    B G R W Y | " * 5,
        " ".join(
            [
                "| 1: {partner_B1_"
                + str(i)
                + "} {partner_G1_"
                + str(i)
                + "} {partner_R1_"
                + str(i)
                + "} {partner_W1_"
                + str(i)
                + "} {partner_Y1_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 2: {partner_B2_"
                + str(i)
                + "} {partner_G2_"
                + str(i)
                + "} {partner_R2_"
                + str(i)
                + "} {partner_W2_"
                + str(i)
                + "} {partner_Y2_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 3: {partner_B3_"
                + str(i)
                + "} {partner_G3_"
                + str(i)
                + "} {partner_R3_"
                + str(i)
                + "} {partner_W3_"
                + str(i)
                + "} {partner_Y3_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 4: {partner_B4_"
                + str(i)
                + "} {partner_G4_"
                + str(i)
                + "} {partner_R4_"
                + str(i)
                + "} {partner_W4_"
                + str(i)
                + "} {partner_Y4_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 5: {partner_B5_"
                + str(i)
                + "} {partner_G5_"
                + str(i)
                + "} {partner_R5_"
                + str(i)
                + "} {partner_W5_"
                + str(i)
                + "} {partner_Y5_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        "+--------------+ " * 5,
        "",
        "-" * 84,
        " Board:                                      | Trash:",
        "                total: {score:2d}                    |   blue: {trash_blue}",
        "  ----------------------------------------   |  green: {trash_green}",
        "  blue |  green |    red |  white | yellow   |    red: {trash_red}",
        " | ".join(["{board_" + color.name + ":>6}" for color in Color])
        + "   |  white: {trash_white}",
        "                                             | yellow: {trash_yellow}",
        "-" * 84,
        " self's hand and knowledge" + " " * 29 + "(card direction: old <-- new)",
        "",
        "+--------------+ " * 5,
        " ".join(
            [
                "| Card #" + str(i) + "  {self_just_hinted_" + str(i) + "} |"
                for i in range(5)
            ]
        ),
        "| Actual:      | " * 5,
        " ".join(
            [
                "|  < {self_color_" + str(i) + ":<6} {self_rank_" + str(i) + "}> |"
                for i in range(5)
            ]
        ),
        "| Knowledge:   | " * 5,
        "|    B G R W Y | " * 5,
        " ".join(
            [
                "| 1: {self_B1_"
                + str(i)
                + "} {self_G1_"
                + str(i)
                + "} {self_R1_"
                + str(i)
                + "} {self_W1_"
                + str(i)
                + "} {self_Y1_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 2: {self_B2_"
                + str(i)
                + "} {self_G2_"
                + str(i)
                + "} {self_R2_"
                + str(i)
                + "} {self_W2_"
                + str(i)
                + "} {self_Y2_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 3: {self_B3_"
                + str(i)
                + "} {self_G3_"
                + str(i)
                + "} {self_R3_"
                + str(i)
                + "} {self_W3_"
                + str(i)
                + "} {self_Y3_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 4: {self_B4_"
                + str(i)
                + "} {self_G4_"
                + str(i)
                + "} {self_R4_"
                + str(i)
                + "} {self_W4_"
                + str(i)
                + "} {self_Y4_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        " ".join(
            [
                "| 5: {self_B5_"
                + str(i)
                + "} {self_G5_"
                + str(i)
                + "} {self_R5_"
                + str(i)
                + "} {self_W5_"
                + str(i)
                + "} {self_Y5_"
                + str(i)
                + "} |"
                for i in range(5)
            ]
        ),
        "+--------------+ " * 5,
        "",
        "=" * 84,
        " Sample commands:",
        "    play 0 | Play 0 | P0 | P 0 | P_0 | 11",
        "    hint_color 4 | HC4 | HC 4 | hint color 4",
        "-" * 84,
    ]
)


def show_game_obj(game_obj):
    info = dict(
        player_id=game_obj.current_player_id,
        hints=game_obj.hints,
        strikes=game_obj.strikes,
        score=game_obj.score,
        deck=len(game_obj.deck),
        last_action=game_obj.actions[-1][1] if game_obj.actions != [] else "None",
    )
    for color in Color:
        info["board_{}".format(color.name)] = game_obj.board.get_color(color)
        info["trash_{}".format(color.name)] = " ".join(
            sorted(str(x[1].value) for x in game_obj.trash.trash if x[0] == color)
        )
    partner_hand = game_obj.partner_hand
    current_hand = game_obj.current_hand.card_count_partner(partner_hand)
    for i in range(5):
        if i in partner_hand.just_hinted:
            info["partner_just_hinted_{}".format(i)] = "***"
        else:
            info["partner_just_hinted_{}".format(i)] = "   "
        if i in current_hand.just_hinted:
            info["self_just_hinted_{}".format(i)] = "***"
        else:
            info["self_just_hinted_{}".format(i)] = "   "
        try:
            info["partner_color_{}".format(i)] = partner_hand[i].color.name
            info["partner_rank_{}".format(i)] = partner_hand[i].rank.value
        except IndexError:
            info["partner_color_{}".format(i)] = " NONE "
            info["partner_rank_{}".format(i)] = " "
        for color in Color:
            c = color.name[0].upper()
            for r in range(1, 6):
                try:
                    if (color, r) in partner_hand[i].all_possibilities:
                        info["partner_{}{}_{}".format(c, r, i)] = "*"
                    else:
                        info["partner_{}{}_{}".format(c, r, i)] = " "
                except IndexError:
                    info["partner_{}{}_{}".format(c, r, i)] = " "
                try:
                    if (color, r) in current_hand[i].all_possibilities:
                        info["self_{}{}_{}".format(c, r, i)] = "*"
                    else:
                        info["self_{}{}_{}".format(c, r, i)] = " "
                except IndexError:
                    info["self_{}{}_{}".format(c, r, i)] = " "
        try:
            if current_hand[i].known_color:
                info["self_color_{}".format(i)] = current_hand[i].color.name
            else:
                info["self_color_{}".format(i)] = "------"
            if current_hand[i].known_rank:
                info["self_rank_{}".format(i)] = current_hand[i].rank.value
            else:
                info["self_rank_{}".format(i)] = " "
        except IndexError:
            info["self_color_{}".format(i)] = " NONE "
            info["self_rank_{}".format(i)] = "-"

    print(interface.format(**info))


class HumanPlayer(Player):
    name = "Team Human"
    raw_state = True
    action_mapping = {
        "p": "play",
        "p_": "play",
        "d": "discard",
        "d_": "discard",
        "hc": "hint_color",
        "hr": "hint_rank",
    }

    def get_action(self, game_obj) -> Action:
        show_game_obj(game_obj)
        while True:
            text = input(" Your Action: ")
            try:
                if len(text) == 1 or text[-2] in {"0", "1", "2"}:
                    return Action(int(text))
                else:
                    index = int(text[-1])
                    text = text[:-1].strip().replace(" ", "_").replace("-", "_").lower()
                    if text in self.action_mapping.keys():
                        text = self.action_mapping[text]
                    return Action.get("{}_{}".format(text, index))
            except (ValueError, AttributeError):
                pass

            print("unrecognized action, please try again")

    def inform(self, game_obj):
        show_game_obj(game_obj)
        sleep(1)
