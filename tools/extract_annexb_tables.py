"""Extract + verify the 16 low-entropy code/flush tables from CCSDS 123.0-B-2 Annex B."""
import re, json

LINES = open("/tmp/ccsds-full-standard.txt", encoding="utf-8").read().splitlines()
# Annex B code/flush tables live between "Table B-1" and Annex C.
START = next(i for i, l in enumerate(LINES) if "Table B-1: Code Table" in l)
END = next(i for i, l in enumerate(LINES) if "ANNEX C" in l and i > START)
region = LINES[START:END]

NOISE = ("CCSDS 123.0-B-2", "RECOMMENDED STANDARD", "LOSSLESS", "Page B-",
         "Input Codeword", "Output Codeword", "Active Prefix", "Flush Word", "Cor. 3",
         "February")   # NB: do NOT filter bare "2021"/"2019" — they are real codewords
OUT_RE = re.compile(r"<?\d+'h(?:\([0-9A-Fr+ ]*\)|[0-9A-F]+)>?")
SYM = {**{str(d): d for d in range(10)}, "A": 10, "B": 11, "C": 12, "X": "X"}


def out_bits(tok, r=0):
    rev = tok.startswith("<")
    tok = tok.strip("<>")
    n, _, val = tok.partition("'h")
    n = int(n)
    if val.startswith("("):
        first, _, second = val.strip("()").partition("+")
        v = int(first, 16)
        if second:
            coef = second.replace("r", "")
            v += (int(coef) if coef else 1) * r
    else:
        v = int(val, 16)
    bits = format(v, "0{}b".format(n))[-n:].zfill(n)
    return bits[::-1] if rev else bits


def syms(template, r=0):
    m = re.search(r"0\^\{(\d+|r)\}", template)
    if m:
        cnt = r if m.group(1) == "r" else int(m.group(1))
        template = template[:m.start()] + "0" * cnt + template[m.end():]
    return tuple(SYM[c] for c in template)


def parse_row(text):
    """Yield (input_symbol_tuple, output_bits) for one input/output spec text."""
    text = text.strip().rstrip(",").strip()
    rng = re.search(r",?\s*(\d+)\s*≤\s*r\s*≤\s*(\d+)", text)
    out = None  # filled by caller
    return text, rng


def extract():
    text = "\n".join(l for l in region if not any(k in l for k in NOISE))
    text = text.replace("’", "'")
    # split into per-table chunks
    chunks = re.split(r"Table B-\d+:\s*(Code|Flush) Table for Low-Entropy Code (\d+)", text)
    codes = {i: {"code": {}, "flush": {}} for i in range(16)}
    # chunks: [pre, kind, idx, body, kind, idx, body, ...]
    for k in range(1, len(chunks), 3):
        kind, idx, body = chunks[k], int(chunks[k + 1]), chunks[k + 2]
        table = {}
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            matches = list(OUT_RE.finditer(line))
            prev = 0
            for mobj in matches:
                inp = line[prev:mobj.start()].strip().rstrip(",").strip()
                prev = mobj.end()
                tok = mobj.group()
                inp = inp.replace("(null)", "").strip()
                rng = re.search(r"(\d+)\s*≤\s*r\s*≤\s*(\d+)", inp)
                base = re.split(r",\s*\d+\s*≤", inp)[0].strip()
                if rng:
                    for r in range(int(rng.group(1)), int(rng.group(2)) + 1):
                        table[syms(base, r)] = out_bits(tok, r)
                else:
                    table[syms(base)] = out_bits(tok)
        codes[idx]["code" if kind == "Code" else "flush"].update(table)
    return codes


codes = extract()

# ---- verification ----
assert out_bits("7'h1F") == "0011111", out_bits("7'h1F")
assert out_bits("<7'h1F>") == "1111100"
# spec example: code 13, input 001 (=0^{2}1) -> <7'h(3F+2r)> at r=2 -> 1100001
assert codes[13]["code"][(0, 0, 1)] == "1100001", codes[13]["code"].get((0, 0, 1))
# direct table spot-checks (no reversal/arithmetic): B-1 "00"->5'h19, B-27 "100"->7'h51
assert codes[0]["code"][(0, 0)] == format(0x19, "05b"), codes[0]["code"].get((0, 0))
assert codes[13]["code"][(1, 0, 0)] == format(0x51, "07b"), codes[13]["code"].get((1, 0, 0))


def _prefix_free(keys):
    ks = sorted(keys, key=len)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if len(a) < len(b) and b[:len(a)] == a:
                return False
    return True


for i in range(16):
    assert _prefix_free(codes[i]["code"].keys()), f"code {i} inputs not prefix-free"


def _complete(cw, L):
    pref = set()
    for k in cw:
        for j in range(len(k)):
            pref.add(k[:j])
    alpha = list(range(L + 1)) + ["X"]
    for P in pref:
        for s in alpha:
            ext = P + (s,)
            if ext not in cw and ext not in pref:
                return False, (P, s)
    return True, None


_Lcheck = [12, 10, 8, 6, 6, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 0]
for i in range(16):
    ok, miss = _complete(set(codes[i]["code"].keys()), _Lcheck[i])
    assert ok, f"code {i} INCOMPLETE: prefix {miss[0]} has no continuation {miss[1]}"
print("all 16 codes verified COMPLETE (every prefix continues on every symbol)")
L = [12, 10, 8, 6, 6, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 0]
print("code | #code-entries #flush-entries | sample")
for i in range(16):
    c, f = codes[i]["code"], codes[i]["flush"]
    # prefix-free check on inputs, suffix-free on outputs
    outs = list(c.values())
    suffix_free = all(not (a != b and a.endswith(b)) for a in outs for b in outs)
    null_in_flush = () in f
    print(f"{i:4d} | {len(c):4d} {len(f):4d} | L={L[i]:2d} suffixfree={suffix_free} flush_null={null_in_flush}")
print("\nVERIFIED against spec worked examples (7'h1F, <7'h1F>, code13[001]).")
json.dump({str(i): {"code": {",".join(map(str, k)): v for k, v in codes[i]["code"].items()},
                    "flush": {",".join(map(str, k)): v for k, v in codes[i]["flush"].items()}}
           for i in range(16)}, open("/tmp/annexb.json", "w"))
print("wrote /tmp/annexb.json")
