from nlpaug.util.doc.token import Token
from nlpaug.util.doc.change_log import ChangeLog


class Doc:
    def __init__(self, doc='', tokens=None):
        self.doc = doc
        if tokens is not None and len(tokens) > 0:
            self.tokens = self.token2obj(tokens)
        else:
            self.tokens = []
        self.changed_cnt = 0

    def token2obj(self, tokens):
        objs = []
        start_pos = 0
        for t in tokens:
            token_obj = Token(token=t, start_pos=start_pos+self.doc[start_pos:].find(t))
            change_log = ChangeLog(orig_token=token_obj)
            objs.append(change_log)

            start_pos += len(token_obj.token)
            start_pos += 1 # TODO: for textual only

        return objs

    def add_token(self, idx, token, action, change_seq):
        token_obj = Token(token=token, start_pos=-1, action=action, change_seq=change_seq)
        change_log = ChangeLog(orig_token=token_obj)
        self.tokens.insert(idx, change_log)

    def add_change_log(self, idx, new_token, action, change_seq):
        self.changed_cnt += 1
        self.tokens[idx].add(new_token, action=action, change_seq=change_seq)

    def update_change_log(self, token_idx, change_idx=None, token=None, action=None, change_seq=None):
        change_idx = self.tokens[token_idx].size() if change_idx is None else change_idx
        self.tokens[token_idx].update(change_idx, token=token, action=action, change_seq=change_seq)

    def get_token(self, idx):
        return self.tokens[idx]

    def get_original_tokens(self):
        return [t.orig_token.token for t in self.tokens]

    def get_augmented_tokens(self):
        return [t.get_latest_token().token for t in self.tokens if len(t.get_latest_token().token) > 0]

    def size(self):
        return len(self.tokens)

    def changed_count(self):
        return self.changed_cnt

    def get_change_logs(self, start_pos=0):
        for i, t in enumerate(self.tokens):
            self.tokens[i].update_last_token(start_pos)

            start_pos += len(t.get_latest_token().token)
            if len(t.get_latest_token().token) > 0:
                # TODO: for textual only
                start_pos += 1

        change_logs = [t for t in self.tokens if t.is_changed()]
        change_logs.sort(key=lambda x: x.get_latest_token().change_seq)
        return [c.to_changed_dict() for c in change_logs]
