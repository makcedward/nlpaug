class Token:
    def __init__(self, token, start_pos=-1, action='', change_seq=0):
        self._token = token
        self._start_pos = start_pos
        self._action = action
        self._change_seq = change_seq

    @property
    def start_pos(self):
        return self._start_pos

    @start_pos.setter
    def start_pos(self, v):
        self._start_pos = v

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, v):
        self._token = v

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, v):
        self._action = v

    @property
    def change_seq(self):
        return self._change_seq

    @change_seq.setter
    def change_seq(self, v):
        self._change_seq = v

    def to_dict(self):
        return {
            'token': self.token,
            'action': self.action,
            'start_pos': self.start_pos,
            'change_seq': self.change_seq
        }
