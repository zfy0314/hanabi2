import json
import os
import pickle
from glob import glob
from multiprocessing import Pool
from typing import List, Optional, Tuple

from fire import Fire
from tqdm import tqdm

from const import Action, GameStateField
from game_components import Game

# from human_player import show_game_obj


class GameHanabReplay(Game):
    def __init__(self, game_log: dict):
        self.player_names = game_log["players"]
        deck = [(card["suitIndex"], card["rank"]) for card in game_log["deck"]]
        self.reset({"deck": {"deck": deck}})
        self.actions_history = game_log["actions"]
        self.data_logger = [{}, {}]
        self.hand_index = [list(range(5)), list(range(5, 10))]
        self.card_id = 10

        black_list = [
            "deckPlays",
            "oneExtraCard",
            "allOrNothing",
            "oneLessCard",
            "detrimentalCharacters",
        ]
        for item in black_list:
            if item in game_log.get("options", {}).keys():
                raise Exception("Only no variant games are used")

        if game_log.get("options", {}).get("startingPlayer") == 1:
            self.current_player_id = 1

    def turn(self) -> bool:

        try:
            action = self.actions_history[self.turns]
        except IndexError:
            return False
        if self.strikes <= 0:
            return False
        if action["type"] == 0:
            index = self.hand_index[self.current_player_id].index(action["target"])
            del self.hand_index[self.current_player_id][index]
            self.hand_index[self.current_player_id].append(self.card_id)
            self.card_id += 1
            action = getattr(Action, "play_{}".format(index))
        elif action["type"] == 1:
            index = self.hand_index[self.current_player_id].index(action["target"])
            del self.hand_index[self.current_player_id][index]
            self.hand_index[self.current_player_id].append(self.card_id)
            self.card_id += 1
            action = getattr(Action, "discard_{}".format(index))
        elif action["type"] == 2:
            action = Action(10 + action["value"])
        elif action["type"] == 3:
            action = Action(14 + action["value"])
        else:
            return False

        state = self.game_state
        # show_game_obj(self)
        # if self.turns > 10:
        #     exit()

        assert self.hints > 0 or (not action.name.startswith("hint")), str(action)
        assert self.hints < 8 or (not action.name.startswith("discard")), str(action)
        try:
            self.data_logger[self.current_player_id][action].append(state)
        except KeyError:
            self.data_logger[self.current_player_id][action] = [state]

        self.act(action)
        self.turns += 1

        self.current_hand.reset_hints()
        self.current_player_id = 1 - self.current_player_id

        return True

    def export_log(self, prefix: str):

        for i, name in enumerate(self.player_names):
            with open("{}__{}.pkl".format(prefix, name), "wb") as fout:
                pickle.dump(self.data_logger[i], fout)


def replay_hanab_live(json_file: str, output_dir: str):

    with open(json_file, "r") as fin:
        game_log = json.load(fin)

    game = GameHanabReplay(game_log)
    game.run()
    game.export_log(os.path.join(output_dir, os.path.basename(json_file)[:-5]))


def replay_hanab_live_all(
    source: str = "../data2P/*.json", destination: str = "../data2P_strip"
):

    black_list = ["212239.json"]
    from glob import glob

    for json_file in glob(source):
        if os.path.basename(json_file) in black_list:
            continue
        try:
            replay_hanab_live(json_file, destination)
        except Exception as E:
            print(json_file, "encountered exception", E.__class__.__name__, E)


def merge_by_user(
    source: str = "../data2P_strip",
    destination: str = "../data2P_user",
    thresh: int = 200,
):

    user_pkl = {}
    for pkl_file in glob(os.path.join(source, "*.pkl")):
        user = os.path.basename(pkl_file)[8:-4]
        try:
            user_pkl[user].append(pkl_file)
        except KeyError:
            user_pkl[user] = [pkl_file]
    for user, pkl_files in tqdm(user_pkl.items()):
        if len(pkl_files) > thresh:
            action = {k: [] for k in range(20)}
            for pkl_file in pkl_files:
                with open(pkl_file, "rb") as fout:
                    states = pickle.load(fout)
                for k in range(20):
                    action[k].extend(states.get(k, []))
            with open(os.path.join(destination, "{}.pkl".format(user)), "wb") as fout:
                pickle.dump(action, fout)


def build_tree(
    all_data: List[Tuple[Tuple[int], int]],
    redundant: bool = False,
    max_depth: int = len(GameStateField) + 1,
    min_states: int = 0,
    rebuild: int = 0,
) -> List[Tuple[Tuple[int], Tuple[int], int, int, int]]:

    todo = [(all_data, tuple(), tuple(), 0)]
    rules = {}
    unused_data = []

    while todo != []:
        subset, fpos, fneg, level = todo.pop()
        len_subset = len(subset)

        if len(set(x[1] for x in subset)) == 1:
            key = (fpos, fneg, subset[0][1])
            if key not in rules.keys():
                rules[key] = (1, len_subset)
            else:
                rules[key] = (1, max(len_subset, rules[key][1]))
        elif (
            len(set(x[0] for x in subset)) == 1
            or level > max_depth
            or len_subset < min_states
        ):
            if rebuild > 0:
                unused_data.extend(subset)
            else:
                all_actions = [x[1] for x in subset]
                num = len(all_actions)
                for action in set(all_actions):
                    count = all_actions.count(action)
                    key = (fpos, fneg, action)
                    if key not in rules.keys():
                        rules[key] = (count / num, len_subset)
                    elif rules[key][0] < count / num:
                        rules[key] = (count / num, len_subset)
        else:
            min_gini = 1
            max_feature = []
            for feature in range(len(GameStateField)):
                if (feature not in fpos) and (feature not in fneg):
                    pos = [x[-1] for x in subset if x[0][feature] == 1]
                    len_pos = len(pos)
                    neg = [x[-1] for x in subset if x[0][feature] == 0]
                    len_neg = len(neg)
                    if len_pos > 0 and len_neg > 0:
                        gini_pos = 1 - sum(pos.count(k) ** 2 for k in range(20)) / (
                            len_pos**2
                        )
                        gini_neg = 1 - sum(neg.count(k) ** 2 for k in range(20)) / (
                            len_neg**2
                        )
                        gini = (gini_pos * len_pos + gini_neg * len_neg) / len_subset
                        if gini < min_gini:
                            min_gini = gini
                            max_feature = [feature]
                        elif gini == min_gini and redundant:
                            max_feature.append(feature)
            for feature in max_feature:
                pos = [x for x in subset if x[0][feature] == 1]
                neg = [x for x in subset if x[0][feature] == 0]
                todo.append((pos, tuple(sorted(fpos + (feature,))), fneg, level + 1))
                todo.append((neg, fpos, tuple(sorted(fneg + (feature,))), level + 1))
            # import pdb

            # pdb.set_trace()

    rules = [
        (fpos, fneg, action, weight, count)
        for (fpos, fneg, action), (weight, count) in rules.items()
    ]

    if unused_data != []:
        rules.extend(
            build_tree(unused_data, redundant, max_depth, min_states, rebuild - 1)
        )

    return sorted(rules, key=lambda x: -x[-1])


class BDDNode:
    def __init__(
        self,
        feature: int,
        count: int,
        parent: Optional[Tuple["BDDNode", str]] = None,
        left: Optional["BDDNode"] = None,
        right: Optional["BDDNode"] = None,
    ):
        self.feature = feature
        self.count = count
        self.parent = parent
        self.left = left
        self.right = right

    def merge_isomorphic(self, other: "BDDNode"):
        if other.left == self.left and other.right == self.right:
            setattr(other.parent[0], other.parent[1], self)
            self.count += other.count

    def remove_redundant(self):
        if self.parent is not None and (self.parent.left == self.parent.right):
            self.parent = self.parent.parent
            if self.parent is not None:
                setattr(self.parent[0], self.parent[1], self)
            self.remove_redundant()


def build_bdd(all_data: List[Tuple[list, int]]) -> List[Tuple[set, set, int, int, int]]:

    pass


def build_one_tree(pkl_file: str, json_out: str) -> Tuple[int, int]:

    with open(pkl_file, "rb") as fin:
        states = pickle.load(fin)
    pairs: List[Tuple[list, int]] = sum(
        [[(tuple(state.tolist()), k) for state in v] for k, v in states.items()], []
    )

    rules = build_tree(pairs)

    agent = {"name": os.path.basename(pkl_file)[:-4], "action-rules": []}
    agent["action-rules"] = [
        {
            "name": "rule-#{}_action-{}_count-{}".format(i, action, count),
            "positive-conditions": sorted([GameStateField(f).name for f in fpos]),
            "negative-conditions": sorted([GameStateField(f).name for f in fneg]),
            "action": Action(action).name,
            "weight": weight,
        }
        for i, (fpos, fneg, action, weight, count) in enumerate(rules)
    ]
    with open(json_out, "w") as fout:
        json.dump(agent, fout, indent=2)

    return len(pairs), len(rules)


def build_individual_tree(
    pkl_file: str,
    json_out: str,
    **kwargs,
) -> Tuple[int, int]:
    with open(pkl_file, "rb") as fin:
        states = pickle.load(fin)
    pairs: List[Tuple[list, int]] = sum(
        [[(tuple(state.tolist()), k) for state in v] for k, v in states.items()], []
    )
    rules = []

    for action in tqdm(set(x[1] for x in pairs)):
        all_data = [(x[0], int(x[1] == action)) for x in pairs]
        raw_rules = build_tree(all_data, **kwargs)
        for rule in raw_rules:
            fpos, fneg, test, weight, count = rule
            if test:
                rules.append((fpos, fneg, action, weight, count))

    # rules.sort(key=lambda x: -x[-1])
    import pdb

    pdb.set_trace()

    agent = {"name": os.path.basename(pkl_file)[:-4], "action-rules": []}
    agent["action-rules"] = [
        {
            "name": "rule-#{}_action-{}_count-{}".format(i, action, count),
            "positive-conditions": sorted([GameStateField(f).name for f in fpos]),
            "negative-conditions": sorted([GameStateField(f).name for f in fneg]),
            "action": Action(action).name,
            "weight": weight,
        }
        for i, (fpos, fneg, action, weight, count) in enumerate(rules)
    ]
    with open(json_out, "w") as fout:
        json.dump(agent, fout, indent=2)

    return len(pairs), len(rules)


def build_rules_all(
    pkl_dir: str, json_out_dir: str, build_func: str = "build_one_tree"
):

    build_tree_func = globals().get(build_func)
    for pkl_file in tqdm(glob(os.path.join(pkl_dir, "*.pkl"))):
        n_state, n_rule = build_tree_func(
            pkl_file,
            os.path.join(
                json_out_dir, os.path.basename(pkl_file).replace(".pkl", ".json")
            ),
        )


def build_rules_all_para(
    pkl_dir: str, json_out_dir: str, build_func: str = "build_one_tree"
):

    p = Pool(os.cpu_count())
    build_tree_func = globals().get(build_func)
    for pkl_file in tqdm(glob(os.path.join(pkl_dir, "*.pkl"))):
        p.apply_async(
            build_tree_func,
            (
                pkl_file,
                os.path.join(
                    json_out_dir, os.path.basename(pkl_file).replace(".pkl", ".json")
                ),
            ),
        )
    p.close()
    p.join()


def plot_nums(user_dir: str, tree_dir: str, output_file: str):

    states = []
    rules = []
    for pkl_file in glob(os.path.join(user_dir, "*.pkl")):
        tree_file = os.path.join(
            tree_dir, os.path.basename(pkl_file).replace(".pkl", ".json")
        )
        with open(pkl_file, "rb") as fin:
            pairs = pickle.load(fin)
            states.append(sum(len(x) for x in pairs.values()))
        with open(tree_file, "r") as fin:
            tree = json.load(fin)
            rules.append(len(tree["action-rules"]))

    from matplotlib import pyplot as plt

    plt.scatter(states, rules)
    plt.xlabel("number of game states")
    plt.ylabel("number of rules")
    plt.savefig(output_file)


if __name__ == "__main__":
    Fire()
