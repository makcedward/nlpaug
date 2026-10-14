from nlpaug.augmenter.sentence import SentenceAugmenter
import nlpaug.model.lang_models as nml
from nlpaug.util import Action


class OpenAIGenAug(SentenceAugmenter):
    """
    Remote OpenAI-backed synthetic NLP data generator.

    This augmenter-style class is designed for dataset generation tasks such as
    intent classification and slot filling. Users provide a task description plus
    examples, and the generator returns structured JSON records.
    """

    def __init__(
        self,
        task_type="intent_and_slot",
        task_description=None,
        examples=None,
        schema=None,
        schema_name="synthetic_nlp_records",
        labels=None,
        slots=None,
        constraints=None,
        model_name="gpt-4o-mini",
        api_key=None,
        base_url=None,
        timeout=60.0,
        max_retries=2,
        temperature=0.7,
        top_p=1.0,
        seed=None,
        name="OpenAIGen_Aug",
        verbose=0,
    ):
        super().__init__(
            action=Action.INSERT,
            name=name,
            tokenizer=None,
            stopwords=None,
            device="remote",
            include_detail=False,
            verbose=verbose,
        )
        self.request_template = {
            "task_type": task_type,
            "task_description": task_description,
            "examples": examples or [],
            "schema": schema,
            "schema_name": schema_name,
            "labels": labels or [],
            "slots": slots or {},
            "constraints": constraints or [],
        }
        self.model = self.get_model(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        self.device = self.model.get_device()

    def generate(self, data=None, n=1):
        request = dict(self.request_template)
        if isinstance(data, str):
            request["task_description"] = data
        elif isinstance(data, dict):
            request.update(data)
        elif data is not None:
            raise ValueError("OpenAIGenAug only supports dict, string, or None input")

        return self.model.generate(request, n=n)

    def augment(self, data=None, n=1, num_thread=1):
        return self.generate(data=data, n=n)

    def insert(self, data):
        return self.generate(data=data, n=1)

    @classmethod
    def get_model(
        cls,
        model_name="gpt-4o-mini",
        api_key=None,
        base_url=None,
        timeout=60.0,
        max_retries=2,
        temperature=0.7,
        top_p=1.0,
        seed=None,
    ):
        return nml.OpenAIChat(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
