from nlpaug.util.doc.token import Token


class ChangeLog:
    def __init__(self, orig_token):
        self.orig_token = orig_token
        self.change_logs = []
        self.add(orig_token.token, 'original', orig_token.change_seq)
        self._is_changed = False

    def add(self, token, action, change_seq):
        if action != 'original' and not self._is_changed:
            self._is_changed = True
        self.change_logs.append(Token(token=token, action=action, change_seq=change_seq))

    def update(self, idx, token=None, action=None, change_seq=None):
        if not self._is_changed:
            self._is_changed = True

        if token:
            self.change_logs[idx].token = token
        if action:
            self.change_logs[idx].action = action
        if change_seq:
            self.change_logs[idx].change_seq = change_seq

    def size(self):
        return len(self.change_logs) - 1

    def is_changed(self):
        return self._is_changed

    def get_latest_token(self):
        return self.change_logs[-1]

    def update_last_token(self, start_pos):
        self.change_logs[-1].start_pos = start_pos

    def to_changed_dict(self):
        return {
            'orig_token': self.orig_token.token,
            'orig_start_pos': self.orig_token.start_pos,
            'new_token': self.get_latest_token().token,
            'new_start_pos': self.get_latest_token().start_pos,
            'change_seq': self.get_latest_token().change_seq,
            'action': self.get_latest_token().action
        }

    def to_dict(self):
        return {
            'orig_token': self.orig_token.to_dict(),
            'change_logs': [t.to_dict() for t in self.change_logs]
        }
