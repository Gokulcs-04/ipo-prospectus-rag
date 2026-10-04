"""
Semantic chunker for IPO prospectus pages  ->  DocumentChunk.

Priority (from the instruction docs):
  semantic meaning > context > table integrity > section/subsection > page provenance > size

Pipeline
  1. clean   : drop repeated headers/footers + standalone page numbers (noise only)
  2. lines   : tag every line with its original 1-indexed PDF page + section/subsection
  3. tables  : detect financial tables across page boundaries, keep title/columns/rows together
  4. units   : headings / paragraphs / tables
  5. pack    : group units into chunks inside one (section, subsection); size is a *secondary* limit
  6. finalize: add heading context, assign chunk_index (per document) and chunk_id

Nothing is calculated, corrected or paraphrased. OCR text is treated like any other source text.
Metadata (heading context, page provenance, chunk_id / chunk_index) lives in metadata.py.
Adjust the imports below to your package layout.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from chunk import DocumentChunk
from document import Document, DocumentPage
from metadata import RawChunk, build_document_chunks


# --------------------------------------------------------------------------- config
@dataclass
class ChunkerConfig:
    max_chars: int = 2500            # secondary size limit (~600 tokens)
    min_chars: int = 400             # don't flush on a heading / keep merging below this
    table_slack: float = 1.5         # a whole table may exceed max_chars by this factor before splitting
    max_context_chars: int = 600     # max size of intro/outro paragraph attached to a table
    edge_lines: int = 3              # first/last N lines per page checked for repeated header/footer
    repeat_ratio: float = 0.25       # edge line on >= this share of a section's pages => running header/footer
    repeat_global_ratio: float = 0.10  # ... or on >= this share of all pages
    repeat_min_pages: int = 3
    min_table_data_lines: int = 3
    column_major_label_run: int = 6  # >= N consecutive label-only lines => flattened column-major table
    prefix_headings: bool = True


# --------------------------------------------------------------------------- line classification
_NUM = re.compile(r"^\(?[-–—]?[₹$]?\d[\d,]*(?:\.\d+)?\)?%?$")
_NIL = {"-", "–", "—"}
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(rf"\b{_MONTH}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b", re.I)
_HDR_WORDS = re.compile(
    r"^(particulars|notes?|as at|as of|for the (?:year|period|six|nine|three)|year ended|"
    r"period ended|(?:six|nine|three) months?|fiscal(?:\s+\d{4})?|fy\s?\d{2,4})\b",
    re.I,
)
_UNIT = re.compile(
    r"^\(.*(?:₹|rs\.?|inr|million|crore|lakh|thousand).*\)$|^(?:₹|rs\.?|inr)\s*(?:in\s+)?"
    r"(?:million|crore|lakh|thousand)",
    re.I,
)
_YEAR_TOKEN = re.compile(r"^(?:FY|Fiscal)?\s?(?:19|20)\d{2}(?:[-–/](?:\d{2}|\d{4}))?$", re.I)
_BOILERPLATE = re.compile(r"intentionally left blank|^\s*page\s+\d+\s+of\s+\d+\s*$", re.I)
_PAGENUM = re.compile(r"^(?:page\s*)?[-–—]?\s*(\d{1,4}|[ivxl]{1,6})\s*[-–—]?$", re.I)


def _is_unit_line(t: str) -> bool:
    return len(t) <= 100 and bool(_UNIT.search(t.strip()))


def _is_header_line(t: str) -> bool:
    t = t.strip()
    if not t or len(t) > 120:
        return False
    if _is_unit_line(t):
        return True
    words = t.split()
    if all(_YEAR_TOKEN.match(w) for w in words):
        return True
    if len(words) <= 12 and not t.endswith("."):
        if _DATE.search(t) or _HDR_WORDS.match(t):
            return True
    return False


def _line_kind(t: str) -> str:
    """blank | hdr (column heading/unit) | num (values only) | row (label + >=2 values) | text"""
    s = t.strip()
    if not s:
        return "blank"
    if _is_header_line(s):
        return "hdr"
    toks = s.split()
    flags = [bool(_NUM.match(x)) or x in _NIL for x in toks]
    if all(flags):
        return "num"
    k = 0
    for f in reversed(flags):
        if not f:
            break
        k += 1
    if k >= 2 and any(c.isalpha() for c in " ".join(toks[: len(toks) - k])):
        return "row"
    return "text"


def _norm_key(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _shape(t: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))


# --------------------------------------------------------------------------- data classes
@dataclass
class Line:
    text: str
    page: int
    section: str | None
    subsection: str | None


@dataclass
class Piece:
    text: str
    pages: set[int]


@dataclass
class Row:
    label: str
    values: list[str]
    lines: list[Line]


@dataclass
class Table:
    title_lines: list[Line]
    unit_lines: list[Line]
    header_lines: list[Line]
    rows: list[Row]
    raw: bool = False
    intro: "Unit | None" = None
    outro: "Unit | None" = None

    def render_row(self, r: Row) -> str:
        if self.raw:
            return r.label
        if r.values:
            return " | ".join(([r.label] if r.label else []) + r.values)
        return r.label


@dataclass
class Unit:
    kind: str                        # heading | para | table
    lines: list[Line] = field(default_factory=list)
    table: Table | None = None
    sub: str | None = None
    claimed: bool = False

    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def pages(self) -> set[int]:
        return {l.page for l in self.lines}


# --------------------------------------------------------------------------- 1. noise removal
def _clean_pages(pages: list[DocumentPage], cfg: ChunkerConfig) -> dict[int, list[str]]:
    """Return lines per page with repeated headers/footers and standalone page numbers removed."""
    texts: dict[int, list[str]] = {}
    for p in pages:
        raw = (p.text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
        texts[p.page_number] = [l.strip() for l in raw.split("\n")]
    order = [p.page_number for p in pages]
    by_num = {p.page_number: p for p in pages}
    nonblank = {pn: [i for i, l in enumerate(ls) if l] for pn, ls in texts.items()}

    # repeated edge lines (running titles, company names, "Page # of #", ...): global or per-section frequency
    n_text = sum(1 for v in nonblank.values() if v)
    sec_size = Counter(by_num[pn].section for pn, v in nonblank.items() if v)
    g_counts: Counter[str] = Counter()
    s_counts: Counter[tuple[str | None, str]] = Counter()
    for pn, idxs in nonblank.items():
        if not idxs:
            continue
        edge = set(idxs[: cfg.edge_lines] + idxs[-cfg.edge_lines:])
        shapes = {s for s in (_shape(texts[pn][i]) for i in edge if len(texts[pn][i]) <= 100) if s and s != "#"}
        g_counts.update(shapes)
        s_counts.update((by_num[pn].section, s) for s in shapes)
    g_thr = max(cfg.repeat_min_pages, math.ceil(cfg.repeat_global_ratio * n_text))
    repeated_g = {s for s, c in g_counts.items() if c >= g_thr}
    repeated_s = {k for k, c in s_counts.items()
                  if c >= max(cfg.repeat_min_pages, math.ceil(cfg.repeat_ratio * sec_size[k[0]]))}

    # running section labels (section/subsection name repeated at page edge after its first page)
    label_pages: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for pn in order:
        for name in {by_num[pn].section, by_num[pn].subsection}:
            if name:
                k = _norm_key(name)
                label_pages[k] += 1
                first_seen.setdefault(k, pn)

    # standalone page numbers (first/last line only)
    cand: dict[tuple[int, str], tuple[int, str]] = {}
    for pn in order:
        idxs = nonblank[pn]
        if not idxs:
            continue
        for pos, i in (("first", idxs[0]), ("last", idxs[-1])):
            m = _PAGENUM.match(texts[pn][i])
            if m:
                cand[(pn, pos)] = (i, m.group(1))

    drop: dict[int, set[int]] = {pn: set() for pn in order}
    for pn in order:
        idxs = nonblank[pn]
        if not idxs:
            continue
        edge = set(idxs[: cfg.edge_lines] + idxs[-cfg.edge_lines:])
        keys = {_norm_key(by_num[pn].section), _norm_key(by_num[pn].subsection)} - {""}
        for i in edge:
            t = texts[pn][i]
            if _line_kind(t) in ("num", "row", "hdr"):
                continue  # never treat table-ish content as noise
            if len(t) <= 100 and (_shape(t) in repeated_g or (by_num[pn].section, _shape(t)) in repeated_s
                                  or _BOILERPLATE.search(t)):
                drop[pn].add(i)
                continue
            k = _norm_key(t)
            if k in keys and label_pages[k] >= 3 and first_seen[k] != pn:
                drop[pn].add(i)

    for (pn, pos), (i, val) in cand.items():
        explicit = texts[pn][i].lower().startswith("page") or not val.isdigit()
        confirmed = False
        if val.isdigit():
            v = int(val)
            for nb, want in ((pn + 1, v + 1), (pn - 1, v - 1)):
                c = cand.get((nb, pos))
                if c and c[1].isdigit() and int(c[1]) == want:
                    confirmed = True
        # a lone number without evidence may be a table value -> keep it
        if explicit or confirmed:
            drop[pn].add(i)

    return {pn: [l for i, l in enumerate(ls) if i not in drop[pn]] for pn, ls in texts.items()}


# --------------------------------------------------------------------------- 3. tables
def _short(t: str) -> bool:
    t = t.strip()
    return len(t) <= 90 and len(t.split()) <= 12 and not t.endswith(".")


def _title_candidates(run: list[Line], kinds: list[str], start: int, floor: int) -> list[int]:
    out: list[int] = []
    j, blanks = start - 1, 0
    while j >= floor and len(out) < 2:
        t = run[j].text.strip()
        if not t:
            blanks += 1
            if blanks > 1:
                break
            j -= 1
            continue
        if (kinds[j] != "text" or len(t) > 100 or not (t[0].isupper() or t[0].isdigit())
                or t.endswith((".", ":", ";", ","))):
            break
        out.append(j)
        j -= 1
    return sorted(out)


def _build_table(title: list[Line], region: list[Line], kinds: list[str], cfg: ChunkerConfig) -> Table:
    i = 0
    hdr: list[Line] = []
    while i < len(region) and kinds[i] in ("hdr", "blank"):
        if kinds[i] == "hdr":
            hdr.append(region[i])
        i += 1
    unit_lines = [l for l in hdr if _is_unit_line(l.text)]
    header_lines = [l for l in hdr if not _is_unit_line(l.text)]
    header_norm = {_norm_key(l.text) for l in hdr}
    body, bkinds = region[i:], kinds[i:]

    # flattened column-major tables (all labels first, then all numbers): don't guess structure
    run = best = 0
    for k in bkinds:
        if k == "text":
            run += 1
            best = max(best, run)
        elif k != "blank":
            run = 0
    raw = best >= cfg.column_major_label_run

    rows: list[Row] = []
    cur: Row | None = None
    skipped: list[Line] = []
    for ln, k in zip(body, bkinds):
        t = ln.text.strip()
        if k == "blank":
            continue
        if raw:
            rows.append(Row(t, [], [ln]))
            continue
        if k == "hdr" and _norm_key(t) in header_norm:
            skipped.append(ln)          # column heading repeated on a continuation page
            continue
        if k == "row":
            toks = t.split()
            n = 0
            for x in reversed(toks):
                if _NUM.match(x) or x in _NIL:
                    n += 1
                else:
                    break
            rows.append(Row(" ".join(toks[:-n]), toks[-n:], skipped + [ln]))
            skipped, cur = [], None
        elif k == "num" or (k == "hdr" and all(_NUM.match(x) or x in _NIL for x in t.split())):
            if cur is None:
                cur = Row("", [], skipped)
                skipped = []
                rows.append(cur)
            cur.values.extend(t.split())
            cur.lines.append(ln)
        else:
            cur = Row(t, [], skipped + [ln])
            skipped = []
            rows.append(cur)
    if skipped and rows:
        rows[-1].lines.extend(skipped)
    return Table(title, unit_lines, header_lines, rows, raw)


def _scan_tables(run: list[Line], cfg: ChunkerConfig) -> list[tuple[int, int, Table]]:
    kinds = [_line_kind(l.text) for l in run]
    n = len(run)

    def next_kinds(j: int, k: int) -> list[str]:
        out, x = [], j + 1
        while x < n and len(out) < k:
            if kinds[x] != "blank":
                out.append(kinds[x])
            x += 1
        return out

    def starts(j: int) -> bool:
        if kinds[j] in ("num", "row", "hdr"):
            return True
        if kinds[j] == "text" and _short(run[j].text):
            nk = next_kinds(j, 1)
            return bool(nk) and nk[0] in ("num", "row")
        return False

    def accepts(j: int, prev: str) -> bool:
        if kinds[j] in ("num", "row", "hdr", "blank"):
            return True
        if kinds[j] == "text" and _short(run[j].text):
            if any(x in ("num", "row") for x in next_kinds(j, 2)):
                return True
            return prev in ("num", "row", "hdr")
        return False

    found: list[tuple[int, int, Table]] = []
    floor, i = 0, 0
    while i < n:
        if not starts(i):
            i += 1
            continue
        j, last, prev = i, -1, "blank"
        while j < n and accepts(j, prev):
            if kinds[j] in ("num", "row"):
                last = j
            if kinds[j] != "blank":
                prev = kinds[j]
            j += 1
        end = last + 1
        data = sum(1 for x in range(i, end) if kinds[x] in ("num", "row"))
        nonblank = sum(1 for x in range(i, end) if kinds[x] != "blank")
        if last >= i and data >= cfg.min_table_data_lines and data / max(nonblank, 1) >= 0.35:
            tidx = _title_candidates(run, kinds, i, floor)
            tb = _build_table([run[x] for x in tidx], run[i:end], kinds[i:end], cfg)
            found.append((tidx[0] if tidx else i, end, tb))
            floor = i = end
        else:
            i = max(i + 1, j)
    return found


# --------------------------------------------------------------------------- 4. units
def _is_heading(ln: Line) -> bool:
    t = ln.text.strip()
    k = _norm_key(t)
    if k and k in (_norm_key(ln.section), _norm_key(ln.subsection)):
        return True
    letters = [c for c in t if c.isalpha()]
    return (len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= 0.9
            and len(t) <= 100 and not t.endswith((".", ",", ";")) and _line_kind(t) == "text")


def _text_units(seg: list[Line]) -> list[Unit]:
    units: list[Unit] = []
    cur: list[Line] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            units.append(Unit("para", cur, sub=cur[0].subsection))
            cur = []

    for ln in seg:
        if not ln.text.strip():
            flush()
        elif _is_heading(ln):
            flush()
            units.append(Unit("heading", [ln], sub=ln.subsection))
        else:
            if cur and ln.subsection != cur[0].subsection:
                flush()
            cur.append(ln)
    flush()
    return units


_INTRO = re.compile(r"\b(following|below|table|set out|presents?|summary|summari[sz]es?|as under|as follows)\b", re.I)
_OUTRO = re.compile(r"read in conjunction|should be read|to be read together|^notes?\b|^source\s*:|^\*", re.I)


def _is_intro(text: str, cfg: ChunkerConfig) -> bool:
    t = text.strip()
    return len(t) <= cfg.max_context_chars and (t.endswith(":") or bool(_INTRO.search(t)))


def _is_outro(text: str, cfg: ChunkerConfig) -> bool:
    t = text.strip()
    return len(t) <= cfg.max_context_chars and bool(_OUTRO.search(t))


# --------------------------------------------------------------------------- paragraph splitting
_ABBR = r"(?<!\bNo)(?<!\bRs)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bLtd)(?<!\bPvt)(?<!\bInc)(?<!\bCo)(?<!\bvs)"
_SENT = re.compile(_ABBR + r"(?<=[.!?])\s+(?=[A-Z(\"“0-9])")


def _split_para(u: Unit, budget: int) -> list[Piece]:
    lines = u.lines
    text = "\n".join(l.text for l in lines)
    offs, pos = [], 0
    for l in lines:
        offs.append((pos, pos + len(l.text), l.page))
        pos += len(l.text) + 1

    def pages(a: int, b: int) -> set[int]:
        return {pg for s, e, pg in offs if s < b and e > a} or {lines[0].page}

    if len(text) <= budget:
        return [Piece(text, pages(0, len(text)))]

    bounds = [0] + [m.end() for m in _SENT.finditer(text)] + [len(text)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    flat: list[tuple[int, int]] = []
    for a, b in spans:                       # last resort: a single sentence longer than budget
        while b - a > budget:
            cut = text.rfind(" ", a, a + budget)
            cut = cut if cut > a else a + budget
            flat.append((a, cut))
            a = cut
        flat.append((a, b))

    out: list[Piece] = []
    a0, b0 = flat[0]
    for a, b in flat[1:]:
        if b - a0 > budget:
            out.append(Piece(text[a0:b0].strip(), pages(a0, b0)))
            a0 = a
        b0 = b
    out.append(Piece(text[a0:b0].strip(), pages(a0, b0)))
    return [p for p in out if p.text]


# --------------------------------------------------------------------------- tables -> chunks
def _emit_table(tb: Table, lead: list[Line], section: str | None, sub: str | None,
                cfg: ChunkerConfig) -> list[RawChunk]:
    title = " ".join(l.text for l in tb.title_lines + tb.unit_lines).strip()
    cols = " | ".join(l.text for l in tb.header_lines)
    hdr = "\n".join(x for x in ((f"Table title: {title}" if title else ""),
                                (f"Columns: {cols}" if cols else "")) if x)
    rts = [tb.render_row(r) for r in tb.rows]
    rpages = [{l.page for l in r.lines} for r in tb.rows]
    lead_t = "\n".join(l.text for l in lead)
    intro_t = tb.intro.text() if tb.intro else ""
    outro_t = tb.outro.text() if tb.outro else ""
    head_pages = ({l.page for l in tb.title_lines + tb.unit_lines + tb.header_lines}
                  | {l.page for l in lead} | (tb.intro.pages() if tb.intro else set()))
    outro_pages = tb.outro.pages() if tb.outro else set()

    def assemble(first: bool, last: bool, idx: list[int]) -> RawChunk:
        parts = []
        if first:
            parts += [x for x in (lead_t, intro_t) if x]
        parts.append("\n".join(([hdr] if hdr else []) + [rts[i] for i in idx]))
        if last and outro_t:
            parts.append(outro_t)
        pg = set().union(*(rpages[i] for i in idx)) if idx else set()
        if first:
            pg |= head_pages
        if last:
            pg |= outro_pages
        return RawChunk("table", section, sub, "\n\n".join(parts), pg)

    everything = list(range(len(rts)))
    whole = assemble(True, True, everything)
    if len(whole.body) <= cfg.max_chars * cfg.table_slack:
        return [whole]                       # table integrity beats the size limit

    # oversized: split by row groups, repeat title + columns in every chunk
    room = max(cfg.max_chars - len(hdr) - 1, cfg.max_chars // 3)
    room_first = max(room - len(lead_t) - len(intro_t) - 4, room // 2)

    def best_break(cur: list[int]) -> int:
        if tb.raw:
            return len(cur)
        for k in range(len(cur) - 1, 0, -1):
            r, prev = tb.rows[cur[k]], tb.rows[cur[k - 1]]
            if (not r.values and r.label) or prev.label.lower().startswith("total"):
                return k
        return len(cur)

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_len, room_now = 0, room_first
    for i in everything:
        rl = len(rts[i]) + 1
        while cur and cur_len + rl > room_now:
            cut = best_break(cur)
            groups.append(cur[:cut])
            cur = cur[cut:]
            cur_len = sum(len(rts[k]) + 1 for k in cur)
            room_now = room
        cur.append(i)
        cur_len += rl
    if cur:
        groups.append(cur)
    return [assemble(n == 0, n == len(groups) - 1, g) for n, g in enumerate(groups)]


# --------------------------------------------------------------------------- 5. packing
def _chunk_section(run: list[Line], cfg: ChunkerConfig) -> list[RawChunk]:
    while run and not run[0].text.strip():
        run = run[1:]
    while run and not run[-1].text.strip():
        run = run[:-1]
    if not run:
        return []
    section = run[0].section

    units: list[Unit] = []
    pos = 0
    for s, e, tb in _scan_tables(run, cfg):
        units += _text_units(run[pos:s])
        units.append(Unit("table", table=tb, sub=run[s].subsection))
        pos = e
    units += _text_units(run[pos:])

    # attach minimal explanatory context to tables
    for k, u in enumerate(units):
        if u.kind != "table":
            continue
        if k > 0 and units[k - 1].kind == "para" and not units[k - 1].claimed \
                and _is_intro(units[k - 1].text(), cfg):
            u.table.intro = units[k - 1]
            units[k - 1].claimed = True
        if k + 1 < len(units) and units[k + 1].kind == "para" and not units[k + 1].claimed \
                and _is_outro(units[k + 1].text(), cfg):
            u.table.outro = units[k + 1]
            units[k + 1].claimed = True

    out: list[RawChunk] = []
    pending: list[Unit] = []                 # headings waiting for their content (never orphaned)
    parts: list[Piece] = []
    cur_len = 0
    cur_sub: str | None = None

    def budget_for(sub: str | None) -> int:
        pre = len("\n".join(h for h in (section, sub) if h))
        return max(cfg.max_chars - pre - 1, cfg.max_chars // 2)

    def flush() -> None:
        nonlocal parts, cur_len
        if parts:
            out.append(RawChunk("text", section, cur_sub, "\n\n".join(p.text for p in parts),
                                set().union(*(p.pages for p in parts))))
        parts, cur_len = [], 0

    for u in units:
        if u.claimed:
            continue
        if u.kind == "heading":
            if parts and cur_len >= cfg.min_chars:
                flush()
            pending.append(u)
            continue
        if u.kind == "table":
            flush()
            lead = [l for h in pending for l in h.lines]
            pending = []
            out.extend(_emit_table(u.table, lead, section, u.sub, cfg))
            continue
        if parts and u.sub != cur_sub:
            flush()
        cur_sub = u.sub
        budget = budget_for(u.sub)
        pieces = _split_para(u, budget)
        if pending:
            h_lines = [l for h in pending for l in h.lines]
            pieces[0] = Piece("\n".join(l.text for l in h_lines) + "\n" + pieces[0].text,
                              pieces[0].pages | {l.page for l in h_lines})
            pending = []
        for pc in pieces:
            if parts and cur_len + len(pc.text) + 2 > budget:
                flush()
            parts.append(pc)
            cur_len += len(pc.text) + 2
    flush()
    # a heading left in `pending` with nothing after it in the section is dropped (no content to attach to)

    merged: list[RawChunk] = []
    for rc in out:                           # absorb tiny trailing text chunks
        prev = merged[-1] if merged else None
        if (prev and rc.kind == "text" and prev.kind == "text" and rc.subsection == prev.subsection
                and len(rc.body) < cfg.min_chars
                and len(prev.body) + len(rc.body) + 2 <= cfg.max_chars * 1.25):
            prev.body += "\n\n" + rc.body
            prev.pages |= rc.pages
        else:
            merged.append(rc)
    return merged


# --------------------------------------------------------------------------- 6. public API
def chunk_document(doc: Document, cfg: ChunkerConfig | None = None) -> list[DocumentChunk]:
    cfg = cfg or ChunkerConfig()
    pages = sorted(doc.pages, key=lambda p: p.page_number)
    cleaned = _clean_pages(pages, cfg)

    lines: list[Line] = []
    for p in pages:
        for t in cleaned[p.page_number]:
            lines.append(Line(t, p.page_number, p.section, p.subsection))

    raw: list[RawChunk] = []
    for _, run in groupby(lines, key=lambda l: l.section):
        raw.extend(_chunk_section(list(run), cfg))

    return build_document_chunks(doc.document_id, raw, prefix_headings=cfg.prefix_headings)


def chunk_documents(docs: list[Document], cfg: ChunkerConfig | None = None) -> list[DocumentChunk]:
    return [c for d in docs for c in chunk_document(d, cfg)]
