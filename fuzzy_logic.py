import re

class FuzzyLogicEngine:
    """
    A simple Fuzzy Logic Engine for handling multi-valued logic degrees of truth (0.0 to 1.0).
    """
    def __init__(self):
        self.fuzzy_kb = {} 

    def _sym(self, s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_ ]+", " ", s).strip().lower()
        parts = [p for p in s.split() if p]
        return "_".join(parts) if parts else ""

    def add_fuzzy_fact(self, text: str) -> str:
        """
        Parses sentences like:
        'Plastic bottle is 90% recyclable'
        'Greasy cardboard is 0.1 recyclable'
        """

        m = re.match(r"(?i)^\s*(.+?)\s+is\s+([0-9.]+)\s*%\s*(.+)$", text)
        if m:
            subj = self._sym(m.group(1))
            pred = self._sym(m.group(3))
            try:
                score = float(m.group(2)) / 100.0
                if 0.0 <= score <= 1.0:
                    self.fuzzy_kb[(subj, pred)] = score
                    return f"Fuzzy KB updated: Certainty({m.group(1).strip()} is {m.group(3).strip()}) = {score:.2f}."
            except ValueError:
                pass
        
        m = re.match(r"(?i)^\s*(.+?)\s+is\s+(0\.\d+|1\.0|0)\s+(.+)$", text)
        if m:
            subj = self._sym(m.group(1))
            pred = self._sym(m.group(3))
            try:
                score = float(m.group(2))
                self.fuzzy_kb[(subj, pred)] = score
                return f"Fuzzy KB updated: Certainty({m.group(1).strip()} is {m.group(3).strip()}) = {score:.2f}."
            except ValueError:
                pass
        
        return "Couldn't parse fuzzy fact. Use format: 'Item is [Number]% Property' or 'Item is [0.0-1.0] Property'."

    def check_fuzzy_fact(self, text: str) -> str:
        """
        Parses queries like:
        'Check certainty that plastic bottle is recyclable'
        """
        m = re.match(r"(?i)^\s*check\s+certainty\s+that\s+(.+?)\s+is\s+(.+)$", text)
        if not m:
            return "Couldn't parse fuzzy query. Use format: 'Check certainty that [Item] is [Property]'."
        
        subj = self._sym(m.group(1))
        pred = self._sym(m.group(2))
        
        if (subj, pred) in self.fuzzy_kb:
            score = self.fuzzy_kb[(subj, pred)]
            if score >= 0.8:
                interpretation = "Highly Likely"
            elif score >= 0.6:
                interpretation = "Likely"
            elif score >= 0.4:
                interpretation = "Uncertain / Mixed"
            elif score >= 0.2:
                interpretation = "Unlikely"
            else:
                interpretation = "Highly Unlikely"
            return f"I am {score*100:.0f}% sure that {m.group(1).strip()} is {m.group(2).strip()}. This is {interpretation}."
        else:
            return f"I have no fuzzy knowledge whether {m.group(1).strip()} is {m.group(2).strip()}."
