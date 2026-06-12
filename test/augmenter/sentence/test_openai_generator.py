import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nlpaug.augmenter.sentence as nas
from nlpaug.model.lang_models.openai_chat import OpenAIChat


class FakeOpenAIModel:
    def __init__(self):
        self.calls = []

    def get_device(self):
        return "remote"

    def generate(self, request, n=1):
        self.calls.append((request, n))
        return [{"text": "Go to McDonald", "intent": "navigate", "slots": {"destination": "McDonald"}}]


class TestOpenAIGenAug(unittest.TestCase):
    def test_augment_uses_template_and_override(self):
        fake_model = FakeOpenAIModel()
        with patch.object(nas.OpenAIGenAug, "get_model", return_value=fake_model):
            aug = nas.OpenAIGenAug(
                task_type="intent_and_slot",
                task_description="Generate virtual assistant navigation examples",
                examples=[{"text": "Go to Starbucks", "intent": "navigate", "slots": {"destination": "Starbucks"}}],
            )

            result = aug.augment(n=1)
            self.assertEqual(1, len(result))
            self.assertEqual("navigate", result[0]["intent"])

            override_result = aug.augment(
                {
                    "task_description": "Generate assistant booking examples",
                    "labels": ["book_restaurant"],
                },
                n=1,
            )
            self.assertEqual("navigate", override_result[0]["intent"])
            self.assertEqual(2, len(fake_model.calls))
            self.assertEqual("Generate assistant booking examples", fake_model.calls[1][0]["task_description"])

    def test_string_input_overrides_task_description(self):
        fake_model = FakeOpenAIModel()
        with patch.object(nas.OpenAIGenAug, "get_model", return_value=fake_model):
            aug = nas.OpenAIGenAug(task_description="initial", examples=[{"text": "a"}])
            aug.augment("Generate travel assistant samples", n=2)

            self.assertEqual("Generate travel assistant samples", fake_model.calls[0][0]["task_description"])
            self.assertEqual(2, fake_model.calls[0][1])


class TestOpenAIChat(unittest.TestCase):
    def _build_response(self, content, prompt_tokens=10, completion_tokens=20, total_tokens=30):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    def test_generate_with_preset_schema(self):
        fake_response = self._build_response(
            '{"records": [{"text": "Go to McDonald", "intent": "navigate", "slots": {"destination": "McDonald"}}]}'
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with patch.object(OpenAIChat, "_create_client", return_value=fake_client):
            model = OpenAIChat(model_name="gpt-4o-mini", api_key="test-key", max_retries=0)
            result = model.generate(
                {
                    "task_type": "intent_and_slot",
                    "task_description": "Generate navigation assistant data",
                    "examples": [{"text": "Go to Starbucks", "intent": "navigate", "slots": {"destination": "Starbucks"}}],
                },
                n=1,
            )

            self.assertEqual(1, len(result))
            self.assertEqual("navigate", result[0]["intent"])
            self.assertEqual(30, model.last_usage["total_tokens"])

    def test_generate_retries_after_invalid_json(self):
        responses = iter(
            [
                self._build_response("not-json"),
                self._build_response('{"records": [{"text": "Book a table", "intent": "book_restaurant", "slots": {"restaurant": "Chez Panisse"}}]}'),
            ]
        )

        def create(**kwargs):
            return next(responses)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        with patch.object(OpenAIChat, "_create_client", return_value=fake_client):
            model = OpenAIChat(model_name="gpt-4o-mini", api_key="test-key", max_retries=1)
            result = model.generate(
                {
                    "task_type": "intent_and_slot",
                    "task_description": "Generate booking assistant data",
                    "examples": [{"text": "Book a table at Chez Panisse", "intent": "book_restaurant", "slots": {"restaurant": "Chez Panisse"}}],
                },
                n=1,
            )

            self.assertEqual("book_restaurant", result[0]["intent"])

    def test_generate_deduplicates_records(self):
        fake_response = self._build_response(
            '{"records": ['
            '{"text": "Go to McDonald", "intent": "navigate", "slots": {"destination": "McDonald"}},'
            '{"text": "Go to McDonald", "intent": "navigate", "slots": {"destination": "McDonald"}},'
            '{"text": "Drive to Target", "intent": "navigate", "slots": {"destination": "Target"}}'
            ']}'
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with patch.object(OpenAIChat, "_create_client", return_value=fake_client):
            model = OpenAIChat(model_name="gpt-4o-mini", api_key="test-key", max_retries=0)
            result = model.generate(
                {
                    "task_type": "intent_and_slot",
                    "task_description": "Generate navigation assistant data",
                },
                n=2,
            )

            self.assertEqual(2, len(result))
            self.assertNotEqual(result[0]["text"], result[1]["text"])

    def test_missing_task_description_raises(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: None)
            )
        )

        with patch.object(OpenAIChat, "_create_client", return_value=fake_client):
            model = OpenAIChat(model_name="gpt-4o-mini", api_key="test-key", max_retries=0)
            with self.assertRaises(ValueError):
                model.generate({"task_type": "intent_and_slot"}, n=1)
