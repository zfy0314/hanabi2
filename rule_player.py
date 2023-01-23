import json
from itertools import product
from typing import List

import numpy as np

from const import Action, Color, GameStateField, Rank
from player import Player, Spectator


class RuleBasedPlayer(Player, Spectator):
    def __init__(self, name: str = "agent"):
        self.name = name
        self.n_rules = 0
        self.pool_rule_names = []
        self.pool_matrix = np.empty((2, 1, len(GameStateField)), dtype=np.uint16)
        self.pool_test = np.zeros((2, 1), dtype=np.uint16)
        self.pool_actions = np.empty((1, len(Action)))
        self.pool_candidate_matrix = np.empty(
            (2, 1, len(GameStateField)), dtype=np.uint16
        )
        self.capacity = 1

    @property
    def rule_names(self):
        return self.pool_rule_names[: self.n_rules]

    @property
    def matrix(self):
        return self.pool_matrix[:, : self.n_rules, :]

    @property
    def test(self):
        return self.pool_test[:, : self.n_rules]

    @property
    def actions(self):
        return self.pool_actions[: self.n_rules, :]

    @property
    def candidate_matrix(self):
        return self.pool_candidate_matrix[:, : self.n_rules, :]

    def load_json(self, json_file: str):
        profile = json.load(open(json_file))
        self.name = profile["name"]

        self.n_rules = len(profile["action-rules"])
        self.pool_rule_names = ["" for _ in range(self.n_rules)]
        self.pool_matrix = np.empty((2, self.n_rules, len(GameStateField)))
        self.pool_test = np.zeros((2, self.n_rules))
        self.pool_actions = np.empty((self.n_rules, len(Action)))
        self.pool_candidate_matrix = np.empty(
            (2, self.n_rules, len(GameStateField)), dtype=np.uint16
        )
        for i, rule in enumerate(profile["action-rules"]):
            self.pool_rule_names[i] = rule.get("name", "")
            positive_conditions = rule.get("positive-conditions", [])
            negative_conditions = rule.get("negative-conditions", [])
            self.pool_matrix[0, i, :] = GameStateField.binary(positive_conditions)
            self.pool_matrix[1, i, :] = GameStateField.binary(negative_conditions)
            self.pool_test[0, i] = self.matrix[0, i, :].sum()
            self.pool_actions[i, :] = Action.onehot(rule["action"]) * rule["weight"]
            self.pool_candidate_matrix[0, i, :] = GameStateField.binary(
                rule.get(
                    "positive_conditions",
                    [x for x in GameStateField if x not in negative_conditions],
                )
            )
            self.pool_candidate_matrix[1, i, :] = GameStateField.binary(
                rule.get(
                    "negative_conditions",
                    [x for x in GameStateField if x not in positive_conditions],
                )
            )
        self.capacity = self.n_rules

    @staticmethod
    def from_json(json_file: str):
        player = RuleBasedPlayer()
        player.load_json(json_file)
        return player

    def to_json(self, json_file: str, include_candidates: bool = True):

        info = {"name": self.name, "action-rules": []}
        for i, name in enumerate(self.rule_names):
            info["action-rules"].append(
                {
                    "name": name,
                    "positive-conditions": GameStateField.to_fields(
                        self.pool_matrix[0, i, :]
                    ),
                    "negative-conditions": GameStateField.to_fields(
                        self.pool_matrix[1, i, :]
                    ),
                    "action": Action(self.pool_actions[i, :].argmax() + 1).name,
                    "weight": self.pool_actions[i, :].max(),
                }
            )
            if include_candidates:
                info["action-rules"][-1].update(
                    {
                        "candidate-positive-conditions": GameStateField.to_fields(
                            self.pool_candidate_matrix[0, i, :]
                        ),
                        "candidate-negative-conditions": GameStateField.to_fields(
                            self.pool_candidate_matrix[1, i, :]
                        ),
                    }
                )
        json.dump(info, open(json_file, "w"), indent=2)

    def double_space(self):

        new_matrix = np.empty(
            (2, self.capacity * 2, len(GameStateField)), dtype=np.uint16
        )
        new_matrix[:, : self.n_rules, :] = self.matrix
        self.pool_matrix = new_matrix
        new_test = np.zeros((2, self.capacity * 2), dtype=np.uint16)
        new_test[:, : self.n_rules] = self.test
        self.pool_test = new_test
        new_actions = np.empty((self.capacity * 2, len(Action)))
        new_actions[: self.n_rules, :] = self.actions
        self.pool_actions = new_actions
        new_matrix = np.empty(
            (2, self.capacity * 2, len(GameStateField)), dtype=np.uint16
        )
        new_matrix[:, : self.n_rules, :] = self.candidate_matrix
        self.pool_candidate_matrix = new_matrix
        self.capacity *= 2

    def add_rules(
        self,
        matrix: np.ndarray,
        actions: np.ndarray,
        candidate_matrix: np.ndarray,
        names: List[str],
    ):

        n_rules = len(names)

        if self.n_rules > 0:
            matching = np.empty((3, self.n_rules, n_rules))
            test = matrix.sum(axis=-1)[:, np.newaxis, :]
            test = np.maximum(
                np.broadcast_to(
                    self.test[:, :, np.newaxis],
                    (2, self.n_rules, n_rules),
                ),
                np.broadcast_to(test, (2, self.n_rules, n_rules)),
            )
            matching[:2] = (self.matrix @ matrix.transpose((0, 2, 1))) == test
            matching[2] = (self.actions @ actions.T) > 0
            matching = np.all(matching, axis=0)
            matched = np.any(matching, axis=0)
            keep_indices = np.logical_not(matched).nonzero()[0]
        else:
            keep_indices = np.arange(n_rules)

        n_rules = len(keep_indices)
        self.pool_rule_names.extend([names[i] for i in keep_indices])
        matrix = matrix[:, keep_indices]
        actions = actions[keep_indices]
        candidate_matrix = candidate_matrix[:, keep_indices]

        while n_rules + self.n_rules > self.capacity:
            self.double_space()
        end = n_rules + self.n_rules
        self.pool_matrix[:, self.n_rules : end] = matrix
        self.pool_test[:, self.n_rules : end] = matrix.sum(axis=-1)
        self.pool_actions[self.n_rules : end] = actions
        self.pool_candidate_matrix[:, self.n_rules : end] = candidate_matrix
        self.n_rules = end

    def get_action(self, game_state: np.ndarray) -> Action:

        test = (self.matrix @ game_state[np.newaxis, :, np.newaxis]).squeeze(axis=-1)
        matched = np.logical_and(test[0] == self.test[0], test[1] == self.test[1])
        weights = (matched[np.newaxis, :] @ self.actions).squeeze(axis=0)

        print("action weights raw:")
        print(weights)

        # adjust hints
        # hint_color = {color.name: [] for color in Color}
        # for color, i in product(Color, range(5)):
        #     if game_state[
        #         GameStateField.get("partner_hand_{}_is_{}".format(i, color.name)) - 1
        #     ]:
        #         hint_color[color.name].append(i)
        # hint_color_sum = {
        #     k: sum(weights[Action.get("hint_color_{}".format(i)) - 1] for i in v)
        #     for k, v in hint_color.items()
        # }
        # for color, indices in hint_color.items():
        #     for i in indices:
        #         weights[Action.get("hint_color_{}".format(i)) - 1] = 0
        #     weights[Action.get("hint_color_{}".format(i)) - 1] = hint_color_sum[color]
        # hint_rank = {rank.value: [] for rank in Rank}
        # for rank, i in product(Rank, range(5)):
        #     if game_state[
        #         GameStateField.get("partner_hand_{}_is_{}".format(i, rank.value)) - 1
        #     ]:
        #         hint_rank[rank.value].append(i)
        # hint_rank_sum = {
        #     k: sum(weights[Action.get("hint_rank_{}".format(i)) - 1] for i in v)
        #     for k, v in hint_rank.items()
        # }
        # for rank, indices in hint_rank.items():
        #     for i in indices:
        #         weights[Action.get("hint_rank_{}".format(i)) - 1] = 0
        #     weights[Action.get("hint_rank_{}".format(i)) - 1] = hint_rank_sum[rank]

        # print("action weights adjusted:")
        # print(weights)

        weights = np.exp(weights - weights.min()) * (weights > 0)
        weights /= weights.sum()

        print("action weights normalized:")
        print(weights)

        print("the following rules are matched:")
        for i, name in enumerate(self.rule_names):
            if matched[i]:
                print("  ", name)
                print("    weights: ", str(self.actions[i, :]))
        # print("normalized weights:")
        # for action in Action:
        #     print("  {} \t {}".format(action.name, weights[action - 1]))
        # print(test[self.rule_names.index("hint-color-playable-0")])
        action = Action(np.random.choice(len(Action), p=weights) + 1)
        print("chosen action: ", action.name)

        return action

    def observe_action(
        self, game_state: np.ndarray, action: Action, result_state: np.ndarray
    ):

        test = (self.matrix @ game_state[np.newaxis, :, np.newaxis]).squeeze(axis=-1)
        matched = np.logical_and(test[0] == self.test[0], test[1] == self.test[1])
        same_action = self.actions[:, action - 1] > 0
        matched_same = np.logical_and(matched, same_action).nonzero()[0]
        matched_diff = np.logical_and(matched, np.logical_not(same_action)).nonzero()[0]
        weights_diff = self.actions[matched_diff].sum()

        print("matched total: \t {}".format(matched.sum()))
        print("matched same: \t {}".format(matched_same.sum()))
        print("mathced diff: \t {}".format(matched_diff.sum()))

        if matched_same.sum() == 0:
            matrix = np.stack(
                (
                    np.eye(len(GameStateField), dtype=np.uint16),
                    np.eye(len(GameStateField), dtype=np.uint16),
                )
            )
            positives = game_state.nonzero()[0]
            negatives = (1 - game_state).nonzero()[0]
            matrix[1, positives] = 0
            matrix[0, negatives] = 0
            actions = np.zeros((len(GameStateField), len(Action)))
            actions[:, action - 1] = 1
            candidate_positive = np.broadcast_to(
                game_state, (len(GameStateField), len(GameStateField))
            )
            candidate_negative = 1 - candidate_positive
            candidate_matrix = np.stack((candidate_positive, candidate_negative))
            names = [
                "rule-{}".format(self.n_rules + i) for i in range(len(GameStateField))
            ]
            self.add_rules(matrix, actions, candidate_matrix, names)

            weights_same = len(GameStateField)
            weights_ratio = min(1, weights_diff / weights_same)
        else:
            weights_same = self.actions[matched_same].sum()
            weights_ratio = min(1, weights_diff / weights_same)
            self.actions[matched_same] *= weights_ratio

        self.actions[matched_diff] /= weights_ratio

        for index in matched_diff.nonzero()[0]:
            candidates = self.candidate_matrix[:, index, :]
            candidates -= np.vstack((game_state, 1 - game_state))
            candidates -= self.matrix[:, index, :]
            candidates = candidates == 1
            n_rules = candidates.sum()
            indices = np.empty((n_rules, 3), dtype=np.uint16)
            indices[:, 1] = np.arange(n_rules)
            indices[:, [0, 2]] = np.argwhere(candidates)
            matrix = np.broadcast_to(
                self.matrix[:, index : index + 1, :],
                (2, n_rules, len(GameStateField)),
            ).copy()
            matrix[tuple(indices.T)] = 1
            candidate_matrix = np.broadcast_to(
                candidates[:, np.newaxis, :].astype(np.uint16),
                (2, n_rules, len(GameStateField)),
            )
            actions = np.broadcast_to(
                self.actions[index : index + 1, :],
                (n_rules, len(Action)),
            )
            names = ["rule-{}".format(self.n_rules + i) for i in range(n_rules)]
            self.add_rules(matrix, actions, candidate_matrix, names)

        self.to_json("learning.json", include_candidates=False)
