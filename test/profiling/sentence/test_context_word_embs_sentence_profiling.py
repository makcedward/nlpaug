import unittest
import os
import time
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas


class TestContextualWordEmbsAugProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.text = 'The quick brown fox jumps over the lazy dog.'

    def test_optimize(self):
        model_paths = ['gpt2', 'distilgpt2']
        device = 'cpu'
        enable_optimize = {'external_memory': 1024, 'return_proba': True}
        disable_optimize = {'external_memory': 0, 'return_proba': True}
        epoch = 10

        for model_path in model_paths:
            # Optimized
            durations = []
            aug = nas.ContextualWordEmbsForSentenceAug(
                model_path=model_path, device=device, optimize=enable_optimize, force_reload=True)
            for i in range(epoch):
                start_dt = time.monotonic()
                for j in range(epoch):
                    aug.augment(self.text)
                end_dt = time.monotonic()
                durations.append(round(end_dt-start_dt, 2))

            optimized_total_duration = sum(durations)
            optimized_average_duration = round(optimized_total_duration/len(durations), 2)

            # No optimized
            durations = []
            aug.model.optimize = disable_optimize
            for _ in range(epoch):
                start_dt = time.monotonic()
                for _ in range(epoch):
                    aug.augment(self.text)
                end_dt = time.monotonic()
                durations.append(round(end_dt - start_dt, 2))

            no_optimized_total_duration = sum(durations)
            no_optimized_average_duration = round(no_optimized_total_duration / len(durations), 2)

            print('Model:{}, Optimized: {}({}), No Optimized: {}({})'.format(
                model_path, optimized_total_duration, optimized_average_duration,
                no_optimized_total_duration, no_optimized_average_duration
            ))

            self.assertGreater(no_optimized_total_duration, optimized_total_duration)
            self.assertGreater(no_optimized_average_duration, optimized_average_duration)
