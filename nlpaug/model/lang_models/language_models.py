import torch
import torch.nn.functional as F


class LanguageModels:
    def __init__(self, cache=True):
        self.cache = cache

    def clean(self, text):
        return text.strip()

    def predict(self, input_tokens, target_word=None, top_n=5):
        raise NotImplementedError()

    def id2token(self, _id):
        raise NotImplementedError()

    @classmethod
    def prob_multinomial(cls, logits, top_n):
        # Convert to probability
        probas = F.softmax(logits, dim=-1)

        # Draw candidates
        top_n_ids = torch.multinomial(probas, num_samples=top_n, replacement=False).tolist()
        top_n_probas = [probas[_id] for _id in top_n_ids]

        return top_n_ids, top_n_probas

    def is_skip_candidate(self, candidate):
        return False

    def get_candidiates(self, candidate_ids, candidate_probas, target_word=None, top_n=5):
        results = []
        for _id, proba in zip(candidate_ids, candidate_probas):
            candidate_word = self.id2token(_id)
            candidate_proba = proba

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
