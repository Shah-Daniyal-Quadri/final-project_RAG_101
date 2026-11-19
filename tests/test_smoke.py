from src.evaluate import exact_match, f1
from src.retriever import search

def test_metrics():
    assert exact_match("The Nolan", "nolan") == 1
    assert 0.0 <= f1("christopher nolan", "Christopher Nolan") <= 1.0

def test_retriever():
    hits = search("Protestant Reformation", k=2)
    assert len(hits) == 2