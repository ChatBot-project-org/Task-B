from typing import List, Tuple
import re
from nltk.sem.logic import Expression
from nltk.inference import ResolutionProver

Expr = Expression.fromstring


class LogicEngine:
    """
    Minimal FOL KB + inference engine for coursework Task-B.

    - Supports two conversational patterns (handled in mybot.py):
        1) "I know that <stmt>"
        2) "Check that <stmt>"

    - Uses NLTK FOL syntax internally.
    - Uses NLTK ResolutionProver for entailment.
    """

    def __init__(self):
        self.kb_exprs: List[Expression] = []
        self.raw_sentences: List[str] = []

    
    # Normalisation helpers
    def _sym(self, s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_ ]+", " ", s).strip().lower()
        parts = [p for p in s.split() if p]
        if not parts:
            return ""
        return "_".join(parts)

    def _const(self, s: str) -> str:
        c = self._sym(s)
        return c.capitalize() if c else c

    def _pred(self, s: str) -> str:
        p = self._sym(s)
        return p.capitalize() if p else p


    # Parsing (restricted patterns)
    def _parse_atomic(self, text: str) -> Tuple[str, bool]:
        """
        Parse simple atomic facts like:
           - 'Bottle is plastic'
           - 'Bottle is not recyclable'
        """
        t = text.strip()

        m = re.match(r"(.+?)\s+is\s+not\s+(.+)$", t, flags=re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"-{self._pred(b)}({self._const(a)})", True

        m = re.match(r"(.+?)\s+is\s+(?:a|an)\s+(.+)$", t, flags=re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{self._pred(b)}({self._const(a)})", True

        m = re.match(r"(.+?)\s+is\s+(.+)$", t, flags=re.I)
        if m:
            a, b = m.group(1), m.group(2)
            return f"{self._pred(b)}({self._const(a)})", True

        return "", False

    def _parse_rule(self, text: str) -> Tuple[str, bool]:
        """
        Optional convenience rules in English:
           'all plastic are recyclable' -> all x. (Plastic(x) -> Recyclable(x))
        """
        m = re.match(r"all\s+(.+?)\s+are\s+(.+)$", text.strip(), flags=re.I)
        if not m:
            return "", False
        a, b = m.group(1), m.group(2)
        x = "x"
        return f"all {x}. ({self._pred(a)}({x}) -> {self._pred(b)}({x}))", True

    def parse_to_fol(self, sent: str) -> Tuple[str, bool]:
        sent = sent.strip().rstrip(".")
        for f in (self._parse_rule, self._parse_atomic):
            expr, ok = f(sent)
            if ok:
                return expr, True
        return "", False

    # Proving helpers
    def _provable(self, goal: Expression) -> bool:
        return ResolutionProver().prove(goal, self.kb_exprs, verbose=False)

    def _neg(self, e: Expression) -> Expression:
        return Expr(f"-({e})")

    def _is_contradictory_with(self, new_e: Expression) -> bool:
        """True if KB already entails NOT(new_e)."""
        return self._provable(self._neg(new_e))

    def add_sentence(self, natural: str) -> str:
        fol, ok = self.parse_to_fol(natural)
        if not ok:
            return "I couldn't understand that as a logical statement. Use: 'X is Y' or 'X is not Y'."

        e = Expr(fol)

        
        if self._is_contradictory_with(e):
            return "Sorry this contradicts with what I know!"

        
        if any(str(existing) == str(e) for existing in self.kb_exprs):
            clean = natural.strip().rstrip(".")
            return f"I already know that {clean}."

        self.kb_exprs.append(e)
        self.raw_sentences.append(natural.strip())

        clean = natural.strip().rstrip(".")
        return f"OK, I will remember that {clean}."

    def check_sentence(self, natural: str) -> str:
        fol, ok = self.parse_to_fol(natural)
        if not ok:
            return "I couldn't understand that to check. Use: 'Check that X is Y'."

        goal = Expr(fol)

        
        if self._provable(goal):
            return "Correct."

        
        if self._provable(self._neg(goal)):
            return "Incorrect."

        
        return "I don't know."

    def seed(self, fol_list: List[str]) -> None:
        """
        Load initial KB and check integrity (no internal contradictions).
        Raises ValueError if a contradiction is detected.
        """
        self.kb_exprs = []
        self.raw_sentences = []

        for s in fol_list:
            s = s.strip()
            if not s or s.startswith("#"):
                continue

            e = Expr(s)

            
            if self._is_contradictory_with(e):
                raise ValueError(f"KB integrity failed: adding '{s}' contradicts existing knowledge.")

            self.kb_exprs.append(e)

    def show(self) -> List[str]:
        return [str(e) for e in self.kb_exprs]
