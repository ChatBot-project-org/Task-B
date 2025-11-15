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
LOG_DIR = Path("chat_logs")

DEBUG = False
SPELL_FIX_ENABLED = True

def banner():
    print("======================================================")
    print(" ISYS30221 Chatbot — AIML + TF-IDF Similarity + Logic (Task B)")
    print(" Commands: :help  :reload  :stats  :debug on/off  :spell on/off  :dict  :kb  :quit")
    print(" Inputs:  I know that <stmt>   |   Check that <stmt>")
    print(" Wikipedia: wiki <topic>")
    print("======================================================")

def load_aiml_kernel() -> aiml.Kernel:
    kernel = aiml.Kernel()
    kernel.verbose(False)
    print("Loading mybot.aiml...", end="", flush=True)
    t0 = time.time()
    kernel.learn(AIML_FILE)
    print(f" done ({time.time()-t0:.2f}s)")
    return kernel

def _tolerant_parse_lines(lines: List[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            rec = next(csv.reader([s]))
            if len(rec) >= 2:
                q = rec[0].strip()
                a = ",".join(rec[1:]).strip() if len(rec) > 2 else rec[1].strip()
                if q and a:
                    parsed.append((q, a))
                continue
        except Exception:
            pass
        for delim in [",", "|", "\t"]:
            if delim in s:
                i = s.find(delim)
                q = s[:i].strip()
                a = s[i + 1 :].strip()
                if q and a:
                    parsed.append((q, a))
                break
    return parsed

def load_kb(kb_path: str):
    if not os.path.exists(kb_path):
        print(f"[warn] KB file not found: {kb_path}. Similarity fallback disabled.")
        return [], [], None, None
    try:
        df = pd.read_csv(kb_path, sep=None, engine="python")
    except Exception as e:
        print(f"[warn] pandas could not parse KB normally ({e}). Using tolerant parser.")
        with open(kb_path, "r", encoding="utf-8", newline="") as f:
            raw_lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
        if not raw_lines:
            print("[warn] KB is empty.")
            return [], [], None, None
        header = raw_lines[0].lower().replace(" ", "")
        data_lines = raw_lines[1:] if header in ("question,answer", "question|answer", "question\tanswer") else raw_lines
        rows = _tolerant_parse_lines(data_lines)
        if not rows:
            print("[warn] No rows parsed from KB.")
            return [], [], None, None
        sio = StringIO()
        sio.write("Question,Answer\n")
        for q, a in rows:
            q_esc = '"' + q.replace('"', '""') + '"'
            a_esc = '"' + a.replace('"', '""') + '"'
            sio.write(f"{q_esc},{a_esc}\n")
        sio.seek(0)
        df = pd.read_csv(sio)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    if "question" not in cols_lower or "answer" not in cols_lower:
        print("[warn] KB must have columns: Question, Answer")
        return [], [], None, None
    df = df.rename(columns={cols_lower["question"]: "Question", cols_lower["answer"]: "Answer"})
    df["Question"] = df["Question"].fillna("").astype(str).str.strip()
    df["Answer"] = df["Answer"].fillna("").astype(str).str.strip()
    df = df[(df["Question"] != "") & (df["Answer"] != "")]
    if df.empty:
        print("[warn] KB has no usable rows after cleaning.")
        return [], [], None, None
    questions = df["Question"].str.lower().tolist()
    answers = df["Answer"].tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    q_mat = vectorizer.fit_transform(questions)
    print(f"Loaded KB: {len(questions)} Q/A pairs")
    return questions, answers, vectorizer, q_mat

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-\']+")

DOMAIN_LEXICON = {
    "recycle","recycling","recyclables","recyclable","contamination","compost","composting","organics",
    "plastic","plastics","glass","metal","aluminum","aluminium","steel","tin","paper","cardboard","carton","cartons",
    "e-waste","ewaste","electronics","battery","batteries","hazardous","mercury",
    "landfill","circular","economy","reuse","reduce","recover",
    "pet","hdpe","ldpe","pp","ps","pet1","hdpe2","ldpe4","pp5","ps6","other7",
    "clamshell","biobased","compostable","liner","lining","foam","styrofoam",
    "caps","lids","jugs","bottles","trays","utensils","foil","pans","aerosol","paint",
    "tanglers","hoses","cords","wires",
    "magazines","newspapers","mail","shredded","tissue","wrap",
    "jars","mixed","color-separation","pyrex",
}

def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())

def build_vocab(questions: List[str]) -> Set[str]:
    vocab = set(DOMAIN_LEXICON)
    for q in questions:
        for w in tokenize_words(q):
            if len(w) > 1:
                vocab.add(w)
    vocab.update({"what","why","how","can","is","are","the","a","an","in","on","do","does","should","with","for","of","to","and"})
    return vocab

def normalize_typos(user_text: str, vocab: Set[str]) -> str:
    if not SPELL_FIX_ENABLED or not vocab:
        return user_text
    words = user_text.split()
    corrected = []
    for token in words:
        core = token.strip()
        low = core.lower()
        if len(low) < 4 or any(ch.isdigit() for ch in low) or low in vocab:
            corrected.append(core)
            continue
        cand, score, _ = process.extractOne(low, vocab, scorer=fuzz.WRatio)
        if score >= 88 or (score >= 80 and abs(len(cand) - len(low)) <= 2):
            if core.istitle():
                corrected.append(cand.capitalize())
            elif core.isupper():
                corrected.append(cand.upper())
            else:
                corrected.append(cand)
            if DEBUG and cand != low:
                print(f"[spell] {core} -> {cand} (score={score})")
        else:
            corrected.append(core)
    return " ".join(corrected)

def best_answer_fallback(user_text: str, questions: List[str], answers: List[str], vectorizer: Optional[TfidfVectorizer], q_mat, threshold: float = SIM_THRESHOLD) -> Optional[str]:
    if vectorizer is None or q_mat is None or not questions:
        return None
    vec = vectorizer.transform([user_text.lower()])
    sims = cosine_similarity(vec, q_mat)[0]
    idx = int(sims.argmax())
    if DEBUG:
        print(f"[dbg] best={sims[idx]:.3f} | Q≈ {questions[idx][:80]}")
    if sims[idx] >= threshold:
        return answers[idx]
    return None

def wiki_summary(topic: str) -> str:
    try:
        return wikipedia.summary(topic, sentences=2)
    except Exception:
        return "Sorry, I couldn't find a concise summary on Wikipedia."

def ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)

def start_transcript() -> Path:
    ensure_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"chat_{ts}.txt"
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
    logic.seed(load_fol_seed())

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
            logic.seed(load_fol_seed())
            bot = "Reloaded AIML, KB, and FOL knowledge."
            print("Bot:", bot)
            append_log(log_path, "Bot", bot)
            continue

        if low == ":stats":
            bot = f"AIML: active\nKB size: {len(questions)}\nSimilarity threshold: {SIM_THRESHOLD}\nDebug: {'on' if DEBUG else 'off'}\nSpell-fix: {'on' if SPELL_FIX_ENABLED else 'off'}\nFOL clauses: {len(logic.show())}"
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
                bot = "Usage: :debug on  |  :debug off"
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

        # ---- Task B: logic inputs ----
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

        # ---- Task A flow ----
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
