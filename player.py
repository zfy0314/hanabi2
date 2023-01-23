from const import Action


class Player:
    name: str

    def __hash__(self):
        return hash(self.name)

    def reset(self):
        pass

    def get_action(self, game_state) -> Action:
        pass

    def inform(self, game_state):
        pass


class Spectator:
    name: str

    def __hash__(self):
        return hash(self.name)

    def reset(self):
        pass

    def observe_action(self, game_state, action: Action, result_state):
        pass
