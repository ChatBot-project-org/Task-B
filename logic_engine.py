from typing import List, Tuple
import re
from nltk.sem.logic import Expression
from nltk.inference import ResolutionProver

Expr = Expression.fromstring

class LogicEngine:
    def __init__(self):
        self.kb_exprs: List[Expression] = []
        self.raw_sentences: List[str] = []

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

    def _parse_atomic(self, text: str) -> Tuple[str, bool]:
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

    def _provable(self, goal: Expression) -> bool:
        return ResolutionProver().prove(goal, self.kb_exprs, verbose=False)

    def _provable_not(self, goal: Expression) -> bool:
        return ResolutionProver().prove(Expr(f"-({goal})"), self.kb_exprs, verbose=False)

    def _is_contradictory_with(self, new_e: Expression) -> bool:
        if self._provable(Expr(f"-({new_e})")):
            return True
        return self._provable_not(new_e)

    def add_sentence(self, natural: str) -> str:
        fol, ok = self.parse_to_fol(natural)
        if not ok:
            return "I couldn't understand that as a logical statement."
        e = Expr(fol)
        if self._is_contradictory_with(e):
            return "This would contradict what I already know. I did not add it."
        self.kb_exprs.append(e)
        self.raw_sentences.append(natural.strip())
        return "Noted."

    def check_sentence(self, natural: str) -> str:
        fol, ok = self.parse_to_fol(natural)
        if not ok:
            return "I couldn't understand that to check."
        goal = Expr(fol)
        if self._provable(goal):
            return "Correct."
        if self._provable_not(goal):
            return "Incorrect."
        return "I don't know."

    def seed(self, fol_list: List[str]) -> None:
        for s in fol_list:
            s = s.strip()
            if not s:
                continue
            self.kb_exprs.append(Expr(s))

    def show(self) -> List[str]:
        return [str(e) for e in self.kb_exprs]
