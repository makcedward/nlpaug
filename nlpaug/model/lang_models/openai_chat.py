import json
import os
import time


class OpenAIChat:
    DEFAULT_SYSTEM_PROMPT = (
        "You generate synthetic NLP training data. "
        "Return valid JSON only. "
        "Follow the requested schema exactly. "
        "Do not include explanations, markdown, or commentary. "
        "Vary phrasing and avoid duplicates."
    )
    PRESET_TASK_TYPES = {
        "intent_classification",
        "slot_filling",
        "intent_and_slot",
    }

    def __init__(
        self,
        model_name="gpt-4o-mini",
        api_key=None,
        base_url=None,
        timeout=60.0,
        max_retries=2,
        temperature=0.7,
        top_p=1.0,
        seed=None,
        silence=True,
    ):
        self.device = "remote"
        self.model_type = "openai_chat"
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.silence = silence
        self.last_usage = None
        self.last_response = None
        self.client = self._create_client()
        self.model = self.client

    def _create_client(self):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missed openai library. Install it via `pip install openai` or `pip install nlpaug[openai]` "
                "to use OpenAI-backed generation."
            ) from exc

        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return OpenAI(**client_kwargs)

    def generate(self, request, n=1):
        normalized_request = self._normalize_request(request, n=n)
        messages = self._build_messages(normalized_request)
        response_format = self._build_response_format(
            task_type=normalized_request["task_type"],
            schema=normalized_request["schema"],
            schema_name=normalized_request["schema_name"],
        )

        attempt = 0
        last_error = None
        while attempt <= self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                )
                self.last_response = response
                self.last_usage = self._extract_usage(response)

                records = self._parse_records(response, normalized_request)
                if len(records) < normalized_request["n"]:
                    raise ValueError(
                        "Expected at least {} records but received {}".format(
                            normalized_request["n"], len(records)
                        )
                    )
                return records[:normalized_request["n"]]
            except Exception as exc:
                last_error = exc
                attempt += 1
                if attempt > self.max_retries:
                    break
                time.sleep(min(0.5 * attempt, 1.5))

        raise last_error

    def predict(self, request, target_word=None, n=1):
        return self.generate(request, n=n)

    def get_device(self):
        return "remote"

    def _normalize_request(self, request, n=1):
        if request is None:
            request = {}
        elif isinstance(request, str):
            request = {"task_description": request}
        elif not isinstance(request, dict):
            raise ValueError("Request must be a dict, string, or None")

        task_type = request.get("task_type", "intent_and_slot")
        if task_type not in self.PRESET_TASK_TYPES and request.get("schema") is None:
            raise ValueError(
                "Unknown task_type {}. Provide one of {} or a custom schema.".format(
                    task_type, sorted(self.PRESET_TASK_TYPES)
                )
            )

        task_description = request.get("task_description")
        if not task_description:
            raise ValueError("task_description is required")

        schema_name = request.get("schema_name", "synthetic_nlp_records")
        schema = request.get("schema")
        if schema is None:
            schema = self._get_preset_record_schema(task_type)

        return {
            "task_type": task_type,
            "task_description": task_description,
            "examples": request.get("examples", []),
            "labels": request.get("labels", []),
            "slots": request.get("slots", {}),
            "constraints": request.get("constraints", []),
            "schema": schema,
            "schema_name": schema_name,
            "system_prompt": request.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT),
            "n": n,
        }

    def _build_messages(self, request):
        payload = {
            "task_type": request["task_type"],
            "task_description": request["task_description"],
            "labels": request["labels"],
            "slots": request["slots"],
            "examples": request["examples"],
            "constraints": request["constraints"],
            "required_record_count": request["n"],
            "output_contract": "Return a JSON object with a top-level `records` array.",
        }

        return [
            {"role": "system", "content": request["system_prompt"]},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=True, indent=2),
            },
        ]

    def _build_response_format(self, task_type, schema, schema_name):
        response_schema = {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": schema,
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name or task_type or "synthetic_nlp_records",
                "strict": True,
                "schema": response_schema,
            },
        }

    def _get_preset_record_schema(self, task_type):
        if task_type == "intent_classification":
            return {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "intent": {"type": "string"},
                },
                "required": ["text", "intent"],
                "additionalProperties": False,
            }
        if task_type == "slot_filling":
            return {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "slots": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                    },
                },
                "required": ["text", "slots"],
                "additionalProperties": False,
            }

        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "intent": {"type": "string"},
                "slots": {
                    "type": "object",
                    "additionalProperties": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                            {"type": "array", "items": {"type": "string"}},
                        ]
                    },
                },
            },
            "required": ["text", "intent", "slots"],
            "additionalProperties": False,
        }

    def _parse_records(self, response, request):
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI response did not include content")

        parsed = json.loads(content)
        if isinstance(parsed, dict) and "records" in parsed:
            records = parsed["records"]
        elif isinstance(parsed, list):
            records = parsed
        else:
            raise ValueError("Expected a JSON object with `records` or a raw list")

        if not isinstance(records, list):
            raise ValueError("`records` must be a list")

        records = self._deduplicate_records(records)
        self._validate_record_list(records, request["task_type"])
        return records

    def _deduplicate_records(self, records):
        unique_records = []
        seen = set()

        for record in records:
            fingerprint = json.dumps(record, sort_keys=True, ensure_ascii=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique_records.append(record)

        return unique_records

    def _validate_record_list(self, records, task_type):
        required_fields_by_task = {
            "intent_classification": ["text", "intent"],
            "slot_filling": ["text", "slots"],
            "intent_and_slot": ["text", "intent", "slots"],
        }
        required_fields = required_fields_by_task.get(task_type, ["text"])

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each generated record must be a dict")
            for field in required_fields:
                if field not in record:
                    raise ValueError("Missing required field `{}` in generated record".format(field))

    def _extract_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
