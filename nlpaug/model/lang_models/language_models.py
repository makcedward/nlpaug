try:
    import torch
    import torch.nn.functional as F
except ImportError:
    # No installation required if not using this function
    pass

import nlpaug.util.selection.filtering as filtering


class LanguageModels:
    OPTIMIZE_ATTRIBUTES = ['external_memory', 'return_proba']

    def __init__(self, device=None, temperature=1.0, top_k=100, top_p=0.01, optimize=None):
        try:
            self.device = 'cuda' if device is None and torch.cuda.is_available() else device
        except NameError:
            raise ImportError('Missed torch, transformers libraries. Install it via '
                              '`pip install torch transformers`')
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimize = self.init_optimize(optimize)

    @classmethod
    def get_default_optimize_config(cls):
        return {
            'external_memory': 1024,  # GPT2 needs either zero or non-zero. XLNet needs number of extra memory tokens.
            'return_proba': False
        }

    def init_optimize(self, optimize):
        _optimize = self.get_default_optimize_config()
        if optimize is None:
            return _optimize

        for attr in self.OPTIMIZE_ATTRIBUTES:
            if attr in optimize:
                _optimize[attr] = optimize[attr]

        return _optimize

    def clean(self, text):
        return text.strip()

    def predict(self, text, target_word=None, n=1):
        raise NotImplementedError

    @classmethod
    def control_randomness(cls, logits, seed):
        temperature = seed['temperature']
        if temperature is not None:
            return logits / temperature
        return logits

    def filtering(self, logits, seed):
        top_k = seed['top_k']
        top_p = seed['top_p']

        if top_k is not None and 0 < top_k < len(logits):
            logits, idxes = filtering.filter_top_k(logits, top_k, replace=-float('Inf'))
        if top_p is not None and 0 < top_p < 1:
            logits, idxes = filtering.nucleus_sampling(logits, top_p)

        # keep = False
        # if not keep:
        #     if self.device == 'cuda':
        #         idxes = idxes.cpu()
        #     idxes = idxes.detach().numpy().tolist()
        #     logits = [logits[idx] for idx in idxes]
        #     logits = torch.tensor(logits)

        return logits, idxes

    def pick(self, logits, target_word, n=1):
        candidate_ids, candidate_probas = self.prob_multinomial(logits, n=n*10)
        results = self.get_candidiates(candidate_ids, candidate_probas, target_word, n)

        return results

    def id2token(self, _id):
        raise NotImplementedError()

    def prob_multinomial(self, logits, n):
        # Convert to probability
        probas = F.softmax(logits, dim=-1)

        # Draw candidates
        top_n_ids = torch.multinomial(probas, num_samples=n, replacement=False).tolist()

        if self.optimize['return_proba']:
            probas = probas.cpu() if self.device == 'cuda' else probas
            probas = probas.detach().data.numpy()
            top_n_probas = [probas[_id] for _id in top_n_ids]
            return top_n_ids, top_n_probas

        return top_n_ids, None

    def is_skip_candidate(self, candidate):
        return False

    def get_candidiates(self, candidate_ids, candidate_probas, target_word=None, n=1):
        # To have random behavior, NO sorting for candidate_probas.
        results = []
        if candidate_probas is None:
            candidate_probas = [0] * len(candidate_ids)

        for candidate_id, candidate_proba in zip(candidate_ids, candidate_probas):
            candidate_word = self.id2token(candidate_id)

            if candidate_word == '':
                continue

            if target_word is not None and candidate_word.lower() == target_word.lower():
                continue

            if self.is_skip_candidate(candidate_word):
                continue

            results.append((candidate_word, candidate_proba))

            if len(results) >= n:
                break

        return results
