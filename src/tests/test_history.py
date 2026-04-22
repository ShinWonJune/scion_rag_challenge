from src.llm_client.history import InMemoryHistoryStore
from src.llm_client.types import Message


def test_inmemory_append_get_prune_clear():
    s = InMemoryHistoryStore()
    s.append(Message(role="user", content="hi"))
    s.append(Message(role="assistant", content="ok"))
    assert len(s.get()) == 2
    s.prune(max_messages=1)
    assert len(s.get()) == 1
    s.clear()
    assert len(s.get()) == 0
