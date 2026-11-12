"""Okapi BM25 keyword search: rewards exact words (names, numbers, rare terms) that
embeddings can blur. Small and dependency-free; the whole index lives in memory."""
import math
import re
from collections import Counter, defaultdict

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those i me my we our you your he she they them their what which who
whom how when where why do does did done have has had not no so than then there here into over
about can could would should will shall may might must also just only very more most such each""".split())
TOKEN = re.compile(r"[a-z0-9₹]+(?:[.,'\-][a-z0-9]+)*")


def tokenize(text: str):
    out = []
    for tok in TOKEN.findall(text.lower()):
        tok = tok.strip(".,'-")
        if not tok or tok in STOPWORDS:
            continue
        out.append(tok)
        if "-" in tok:                       # "rag-sequence" also counts as "rag" + "sequence"
            out.extend(p for p in tok.split("-") if p and p not in STOPWORDS)
    return out


class BM25:
    def __init__(self, documents, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = []
        self.postings = defaultdict(list)    # term -> [(doc index, term frequency)]
        for i, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def search(self, query: str, top_n: int = 30):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]
