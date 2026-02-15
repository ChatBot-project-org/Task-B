import os
import csv
import re
import time
import aiml
import wikipedia
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

from logic_engine import LogicEngine

AIML_FILE = "mybot.aiml"
KB_FILE = "qa_kb.csv"
FOL_FILE = "fol_kb.txt"
SIM_THRESHOLD = 0.35
LOG_DIR = "logs"
DEBUG = False
SPELL_FIX_ENABLED = True


def banner():
    print("=" * 60)
    print("Recycling Chatbot (AIML + TF-IDF fallback + Task-B FOL)")
    print("Type :help for commands, :quit to exit")
    print("=" * 60)


def load_aiml_kernel() -> aiml.Kernel:
    k = aiml.Kernel()
    if os.path.exists(AIML_FILE):
        k.learn(AIML_FILE)
    return k


def load_kb(path: str) -> Tuple[List[str], List[str], TfidfVectorizer, any]:
    questions, answers = [], []
    if not os.path.exists(path):
        return questions, answers, TfidfVectorizer(), None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                questions.append(q)
                answers.append(a)
    vectorizer = TfidfVectorizer(stop_words="english")
    q_mat = vectorizer.fit_transform(questions) if questions else None
    return questions, answers, vectorizer, q_mat


def best_answer_fallback(user: str, questions: List[str], answers: List[str],
                         vectorizer: TfidfVectorizer, q_mat) -> Optional[str]:
    if not questions or q_mat is None:
        return None
    u_vec = vectorizer.transform([user])
    sims = cosine_similarity(u_vec, q_mat)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score >= SIM_THRESHOLD:
        return answers[best_idx]
    return None


def wiki_summary(topic: str) -> str:
    try:
        return wikipedia.summary(topic, sentences=2)
    except Exception:
        return "Sorry, I couldn't fetch that from Wikipedia."


def build_vocab(questions: List[str]) -> Set[str]:
    vocab = set()
    for q in questions:
        for t in re.findall(r"[A-Za-z']+", q.lower()):
            vocab.add(t)
    
    vocab.update({
        "recycle", "recyclable", "plastic", "glass", "paper", "metal", "aluminum",
        "battery", "hazardous", "compost", "compostable", "bin", "waste"
    })
    return vocab


def normalize_typos(text: str, vocab: Set[str]) -> str:
    if not SPELL_FIX_ENABLED or not vocab:
        return text
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    out = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z']+", tok):
            w = tok.lower()
            if w in vocab:
                out.append(tok)
            else:
                match, score, _ = process.extractOne(w, vocab, scorer=fuzz.ratio)
                if score >= 85:
                    out.append(match)
                else:
                    out.append(tok)
        else:
            out.append(tok)
    return "".join(out)


def start_transcript() -> Path:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(LOG_DIR) / f"chat_{ts}.txt"
    path.write_text("Chat transcript started\n", encoding="utf-8")
    return path


def append_log(log_path: Path, speaker: str, text: str):
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {speaker}: {text}\n")
    except Exception:
        pass


def load_fol_seed() -> List[str]:
    if not os.path.exists(FOL_FILE):
        return []
    lines = []
    with open(FOL_FILE, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def main():
    global DEBUG, SPELL_FIX_ENABLED
    kernel = load_aiml_kernel()
    questions, answers, vectorizer, q_mat = load_kb(KB_FILE)
    vocab = build_vocab(questions)

    logic = LogicEngine()
    try:
        logic.seed(load_fol_seed())
    except ValueError as ex:
        print("ERROR: Knowledgebase integrity check failed.")
        print(str(ex))
        return

    banner()
    log_path = start_transcript()

    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue

        append_log(log_path, "User", user)
        low = user.lower()

        if low == ":quit":
            bot = "Goodbye!"
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            break

        if low == ":help":
            bot = (
                "AIML first; if no match, TF-IDF + cosine over Recycling Q/A.\n"
                "Task B logic:\n"
                "  I know that <stmt>\n"
                "  Check that <stmt>\n"
                "Examples: I know that Bottle is plastic.  Check that Bottle is recyclable.\n"
                "Commands: :help  :reload  :stats  :debug on/off  :spell on/off  :dict  :kb  :quit\n"
                "Wikipedia: wiki <topic>"
            )
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low == ":reload":
            kernel = load_aiml_kernel()
            questions, answers, vectorizer, q_mat = load_kb(KB_FILE)
            vocab = build_vocab(questions)

            logic = LogicEngine()
            try:
                logic.seed(load_fol_seed())
            except ValueError as ex:
                bot = "ERROR: Knowledgebase integrity check failed. Fix fol_kb.txt then reload."
                print("Bot:", bot)
                append_log(log_path, "Bot", bot)
                print(str(ex))
                continue

            bot = "Reloaded AIML, KB, and FOL knowledge."
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low == ":stats":
            bot = (
                f"AIML: active\n"
                f"KB size: {len(questions)}\n"
                f"Similarity threshold: {SIM_THRESHOLD}\n"
                f"Debug: {'on' if DEBUG else 'off'}\n"
                f"Spell-fix: {'on' if SPELL_FIX_ENABLED else 'off'}\n"
                f"FOL clauses: {len(logic.show())}"
            )
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low == ":kb":
            fol_lines = logic.show()
            bot = "KB:\n" + ("\n".join(fol_lines) if fol_lines else "(empty)")
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low.startswith(":debug"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                DEBUG = parts[1] == "on"
                bot = f"Debug mode set to {parts[1]}"
            else:
                bot = "Usage: :debug on | :debug off"
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low.startswith(":spell"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                SPELL_FIX_ENABLED = (parts[1] == "on")
                bot = f"Spell-fix set to {parts[1]}"
            else:
                bot = "Usage: :spell on | :spell off"
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low == ":dict":
            bot = f"Vocabulary contains {len(vocab)} tokens (KB + domain lexicon)."
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low.startswith("wiki "):
            topic = user[5:].strip()
            topic_norm = normalize_typos(topic, vocab)
            bot = wiki_summary(topic_norm) if topic_norm else "Usage: wiki <topic>"
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        m_know = re.match(r"^\s*i\s+know\s+that\s+(.+)$", user, flags=re.I)
        if m_know:
            stmt = m_know.group(1).strip()
            bot = logic.add_sentence(stmt)
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        m_check = re.match(r"^\s*check\s+that\s+(.+)$", user, flags=re.I)
        if m_check:
            stmt = m_check.group(1).strip()
            bot = logic.check_sentence(stmt)
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        user_norm = normalize_typos(user, vocab)
        if DEBUG and user_norm != user:
            print(f"[spell] '{user}' -> '{user_norm}'")

        resp = kernel.respond(user_norm)
        if resp:
            print("Bot:", resp)
            append_log(log_path, "Bot", resp)
            continue

        fall = best_answer_fallback(user_norm, questions, answers, vectorizer, q_mat)
        if fall:
            print("Bot:", fall)
            append_log(log_path, "Bot", fall)
            continue

        bot = "I'm not sure about that. Could you rephrase or ask something else?"
        print("Bot:", bot)
        append_log(log_path, "Bot", bot)


if __name__ == "__main__":
    main()
