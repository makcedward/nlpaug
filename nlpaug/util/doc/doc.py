import string

from nlpaug.util.doc.token import Token
from nlpaug.util.doc.change_log import ChangeLog


class Doc:
    SUBWORD_PREFIXES = ('##', 'Ġ', '▁')

    def __init__(self, doc='', tokens=None):
        self.doc = doc
        if tokens is not None and len(tokens) > 0:
            self.tokens = self.token2obj(tokens)
        else:
            self.tokens = []
        self.changed_cnt = 0

    def token2obj(self, tokens):
        objs = []
        search_pos = 0
        for t in tokens:
            surface_token = self._surface_token(t)
            start_pos = self.doc.find(surface_token, search_pos)
            if start_pos == -1 and surface_token != t:
                start_pos = self.doc.find(t, search_pos)

            token_obj = Token(token=t, start_pos=start_pos)
            change_log = ChangeLog(orig_token=token_obj)
            objs.append(change_log)

            if start_pos != -1:
                search_pos = start_pos + len(surface_token)

        return objs

    @classmethod
    def _surface_token(cls, token):
        for prefix in cls.SUBWORD_PREFIXES:
            if token.startswith(prefix):
                return token[len(prefix):]
        return token

    @classmethod
    def _starts_new_word(cls, token, previous_token, token_idx, orig_token, previous_orig_token):
        if token_idx == 0 or token.startswith(cls.SUBWORD_PREFIXES):
            return False
        if token in string.punctuation:
            return False
        if previous_orig_token is not None and orig_token.start_pos != -1:
            previous_end_pos = (
                previous_orig_token.start_pos
                + len(cls._surface_token(previous_orig_token.token))
            )
            if previous_end_pos == orig_token.start_pos:
                return False
        return previous_token not in '([{</'

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
        previous_token = ''
        previous_orig_token = None
        for i, t in enumerate(self.tokens):
            latest_token = t.get_latest_token().token
            if latest_token and self._starts_new_word(
                    latest_token, previous_token, i, t.orig_token, previous_orig_token):
                start_pos += 1

            self.tokens[i].update_last_token(start_pos)

            start_pos += len(self._surface_token(latest_token))
            if latest_token:
                previous_token = latest_token
            if t.orig_token.start_pos != -1:
                previous_orig_token = t.orig_token

        change_logs = [t for t in self.tokens if t.is_changed()]
        change_logs.sort(key=lambda x: x.get_latest_token().change_seq)
        return [c.to_changed_dict() for c in change_logs]
