"""Lookup order: Redis -> PostgreSQL -> existing Speculative RAG pipeline.

The pipeline itself (Speculative_RAG.answer_question) is called unchanged.
"""

import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import cache
import database
from text_utils import normalize_question, question_hash

logger = logging.getLogger("backend.qa")

SOURCE_REDIS = "redis"
SOURCE_POSTGRES = "postgresql"
SOURCE_PIPELINE = "speculative_rag"
SOURCE_DIRECT = "direct"


_SMALL_TALK_REPLIES = {
    "hello": "Hello! How can I help you?",
    "hi": "Hello! How can I help you?",
    "hey": "Hey! How can I help you?",
    "hiya": "Hey! How can I help you?",
    "good morning": "Good morning! How can I help you?",
    "good afternoon": "Good afternoon! How can I help you?",
    "good evening": "Good evening! How can I help you?",
    "how are you": "I'm doing well, thanks! How can I help you?",
    "how are you doing": "I'm doing well, thanks! How can I help you?",
    "how is it going": "Pretty well, thanks! How can I help you?",
    "how's it going": "Pretty well, thanks! How can I help you?",
    "whats up": "Not much! How can I help you?",
    "what's up": "Not much! How can I help you?",
    "thank you": "You're welcome!",
    "thanks": "You're welcome!",
    "goodbye": "Goodbye! Have a great day.",
    "bye": "Goodbye! Have a great day.",
    "hallo": "Hallo! Wie kann ich dir helfen?",
    "guten morgen": "Guten Morgen! Wie kann ich dir helfen?",
    "guten tag": "Guten Tag! Wie kann ich dir helfen?",
    "guten abend": "Guten Abend! Wie kann ich dir helfen?",
    "wie geht es dir": "Mir geht's gut, danke! Wie kann ich dir helfen?",
    "wie gehts dir": "Mir geht's gut, danke! Wie kann ich dir helfen?",
    "wie geht's": "Mir geht's gut, danke! Wie kann ich dir helfen?",
    "danke": "Gern geschehen!",
    "tschuss": "Tschüss! Hab einen schönen Tag.",
    "auf wiedersehen": "Auf Wiedersehen!",
}


def _small_talk_reply(question):
    """Return a quick, document-free reply for recognized small-talk phrases."""
    normalized = normalize_question(question)
    reply = _SMALL_TALK_REPLIES.get(normalized)
    if reply:
        return reply

    # Accept common stretched spellings such as "hii", "heyyy", and "hellooo".
    if re.fullmatch(r"h+i+", normalized):
        return _SMALL_TALK_REPLIES["hi"]
    if re.fullmatch(r"he+y+", normalized):
        return _SMALL_TALK_REPLIES["hey"]
    if re.fullmatch(r"hell+o+", normalized):
        return _SMALL_TALK_REPLIES["hello"]
    return None


@dataclass
class Answer:
    answer: str
    source: str
    documents: list = field(default_factory=list)


class SingleFlight:
    """One lock per key, so identical simultaneous questions run the pipeline only once."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks = {}  # key -> [lock, waiting_count]

    @contextmanager
    def lock(self, key):
        with self._guard:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    self._locks.pop(key, None)


class QAService:
    def __init__(self, rag):
        self.rag = rag  # the imported Speculative_RAG module
        self._flight = SingleFlight()

    # -- lookups ------------------------------------------------------------ #
    def _lookup(self, q_hash):
        hit = cache.get(q_hash)
        if hit:
            return Answer(hit["answer"], SOURCE_REDIS, hit.get("documents", []))
        row = database.find_answer(q_hash)
        if row:
            cache.put(q_hash, row["answer"], row["documents"])  # warm Redis
            return Answer(row["answer"], SOURCE_POSTGRES, row["documents"])
        return None

    # -- main entry --------------------------------------------------------- #
    def ask(self, question, history=None, document=None, top_k=20, n_subsets=5) -> Answer:
        question = question.strip()
        scope = document or ""

        direct_reply = _small_talk_reply(question)
        if direct_reply:
            return Answer(direct_reply, SOURCE_DIRECT)

        # Follow-ups ("what about it?") only make sense with the chat, so they are keyed on
        # the *standalone* rewrite. rewrite_question() returns the original text when it
        # fails, which we cannot tell apart from "already standalone", so mid-chat questions
        # that come back unchanged skip the cache rather than risk caching an ambiguous one.
        standalone = question
        cacheable = True
        if history:
            standalone = self.rag.rewrite_question(question, history)
            cacheable = standalone != question

        normalized = normalize_question(standalone)
        q_hash = question_hash(normalized, scope)

        if cacheable:
            hit = self._lookup(q_hash)
            if hit:
                logger.info("Answered from %s (%s)", hit.source, q_hash[:12])
                return hit

        def run_pipeline():
            # history=None: the question is already standalone, so don't rewrite it twice.
            answer, sources = self.rag.answer_question(
                standalone, k=top_k, n_subsets=n_subsets, history=None, source=document,
            )
            return Answer(answer, SOURCE_PIPELINE, list(sources))

        if not cacheable:
            return run_pipeline()

        with self._flight.lock(q_hash):
            hit = self._lookup(q_hash)  # another request may have finished while we waited
            if hit:
                return hit
            result = run_pipeline()
            # Fixed "no documents indexed" / "document not found" replies have no sources;
            # they describe the current state of the index, not an answer worth keeping.
            if result.documents:
                database.save_answer(question, normalized, scope, q_hash, result.answer, result.documents)
                cache.put(q_hash, result.answer, result.documents)
            return result

    # -- invalidation ------------------------------------------------------- #
    def invalidate_document(self, name):
        """Drop stored answers that may rest on an outdated version of document `name`.

        Affected: answers scoped to that PDF and answers over all documents (scope "").
        """
        hashes = database.delete_by_scope(["", name])
        if hashes is None:
            cache.flush_answers()  # PostgreSQL unavailable: clear the whole cache to be safe
        else:
            cache.delete(hashes)
        logger.info("Invalidated cached answers after a change to %r", name)
