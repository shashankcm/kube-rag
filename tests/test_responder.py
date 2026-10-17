"""The responder must not answer a technical question from model memory."""

from types import SimpleNamespace

from app.agents.nodes import responder


class FakePortkey:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature):
        self.prompts.append(messages[0]["content"])
        msg = SimpleNamespace(content="an answer")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _state(documents, query="how do pods restart"):
    return {
        "current_query": query,
        "messages": [{"role": "user", "content": "How do pods restart?"}],
        "documents": documents,
        "plan": ["Start"],
        "status": "",
    }


def test_no_documents_skips_the_llm_and_says_so(monkeypatch):
    fake = FakePortkey()
    monkeypatch.setattr(responder, "portkey_client", fake)

    out = responder.generate_node(_state([]))

    assert fake.prompts == []
    assert "don't have enough information" in out["final_answer"].lower()
    assert out["messages"][0]["role"] == "assistant"


def test_prompt_tells_model_to_stay_within_context(monkeypatch):
    fake = FakePortkey()
    monkeypatch.setattr(responder, "portkey_client", fake)
    monkeypatch.setattr(responder, "extract_cache_status", lambda r: "MISS")

    out = responder.generate_node(_state(["Pods restart per restartPolicy."]))

    assert out["final_answer"] == "an answer"
    prompt = fake.prompts[0]
    assert "Pods restart per restartPolicy." in prompt
    assert "ONLY" in prompt and "don't know" in prompt.lower()


def test_conversational_path_does_not_need_documents(monkeypatch):
    fake = FakePortkey()
    monkeypatch.setattr(responder, "portkey_client", fake)
    monkeypatch.setattr(responder, "extract_cache_status", lambda r: "MISS")

    out = responder.generate_node(_state([], query="CONVERSATIONAL"))

    assert len(fake.prompts) == 1
    assert out["final_answer"] == "an answer"
