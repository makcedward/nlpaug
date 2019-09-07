try:
    import torch
    import torch.nn.functional as F
except ImportError:
    # No installation required if not using this function
    pass


class LanguageModels:
    def __init__(self, device, top_k=100, top_p=0.01, cache=True):
        self.device = device
        self.cache = cache

        self.top_k = top_k # top_n + top_k for luck draw
        self.top_p = top_p

    def clean(self, text):
        return text.strip()

    def predict(self, input_tokens, target_word=None, top_n=5):
        raise NotImplementedError()

    def id2token(self, _id):
        raise NotImplementedError()

    def prob_multinomial(self, logits, top_n):
        # Convert to probability
        probas = F.softmax(logits, dim=-1)

        # Draw candidates
        top_n_ids = torch.multinomial(probas, num_samples=top_n, replacement=False).tolist()
        probas = probas.cpu() if self.device == 'cuda' else probas
        probas = probas.detach().data.numpy()
        top_n_probas = [probas[_id] for _id in top_n_ids]

        return top_n_ids, top_n_probas

    def is_skip_candidate(self, candidate):
        return False

    def get_candidiates(self, candidate_ids, candidate_probas, target_word=None, top_n=5):
        # To have random behavior, NO sorting for candidate_probas.
        results = []
        for candidate_id, candidate_proba in zip(candidate_ids, candidate_probas):
            candidate_word = self.id2token(candidate_id)

            if candidate_word == '':
                continue

            if target_word is not None and candidate_word.lower() == target_word.lower():
                continue

            if self.is_skip_candidate(candidate_word):
                continue

            results.append((candidate_word, candidate_proba))

            if len(results) >= top_n:
                break

        return results
