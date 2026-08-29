"""
Numera — Math-To-Manim inspired pipeline.
Architecture adapted from github.com/HarleyCoops/Math-To-Manim (MIT).

Pipeline:
  1. SymPy Math Engine   → exact facts (roots, derivative, integral, domain)
  2. Curriculum Agent    → pedagogical narrative + shot list
  3. Scene-Composer Agent→ complete Manim CE scene spec
  4. Code-Gen Agent      → executable Manim Python code
  5. Execute + Heal      → run Manim CLI, fix errors (up to 3 passes)
"""

import os
import re
import json
import hashlib
import logging
import sqlite3
import shutil
import subprocess
import sys
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Load .env FIRST
from dotenv import load_dotenv
load_dotenv()

import sympy as sp
from openai import OpenAI
from anthropic import Anthropic

# ── 1. OBSERVABILITY ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

# OpenRouter client (free models)
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_retries=0,
)
# Direct OpenAI fallback (bypasses OpenRouter rate limits)
openai_direct = OpenAI(api_key=OPENAI_API_KEY, max_retries=0) if OPENAI_API_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

DB_PATH = os.environ.get("PRODUCTION_DB_PATH", "production_state.db")
MAX_RETRIES = int(os.environ.get("RENDER_MAX_RETRIES", "3"))

# ── 2. STATE MANAGEMENT ───────────────────────────────────────────────────────
@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS jobs
                     (job_id TEXT PRIMARY KEY, status TEXT, equation TEXT,
                      code TEXT, video_path TEXT, error TEXT,
                      updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS cache
                     (hash_key TEXT PRIMARY KEY, video_path TEXT)''')


def update_job_state(job_id: str, status: str, code: str = "",
                     video_path: str = "", error: str = ""):
    with db_conn() as conn:
        conn.execute('''INSERT INTO jobs (job_id, status, code, video_path, error)
                     VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status,
                        code=CASE WHEN excluded.code != '' THEN excluded.code ELSE jobs.code END,
                        video_path=CASE WHEN excluded.video_path != '' THEN excluded.video_path ELSE jobs.video_path END,
                        error=excluded.error,
                        updated_at=CURRENT_TIMESTAMP''',
                  (job_id, status, code, video_path, error))
    log.info(f"Job {job_id} → {status}")


def get_cache(hash_key: str) -> Optional[str]:
    with db_conn() as conn:
        row = conn.execute("SELECT video_path FROM cache WHERE hash_key=?", (hash_key,)).fetchone()
    return row[0] if row else None


def set_cache(hash_key: str, video_path: str):
    with db_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO cache (hash_key, video_path) VALUES (?, ?)",
                     (hash_key, video_path))


# ── 3. MODEL ROUTER ───────────────────────────────────────────────────────────
FREE_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # 550B — best quality
    "nvidia/nemotron-3-super-120b-a12b:free",   # 120B — reliable fallback
    "cohere/north-mini-code:free",               # code-specialised
    "google/gemma-4-31b-it:free",               # 262k ctx
    "openai/gpt-oss-20b:free",                  # GPT-based
    "nvidia/nemotron-3-nano-30b-a3b:free",      # small but fast
]


def call_llm(prompt: str, system_role: str,
             temperature: float = 0.1, max_tokens: int = 2000) -> str:
    last_err = None
    for model in FREE_MODELS:
        try:
            log.info(f"Trying model: {model}")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user",   "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Empty response from model.")
            log.info(f"Success: {model}")
            return content
        except Exception as e:
            log.warning(f"Model {model} failed: {e}")
            last_err = e

    # Fallback: direct OpenAI API (bypasses OpenRouter rate limits)
    if openai_direct:
        for model in ("gpt-4.1-mini", "gpt-4o-mini"):
            try:
                log.info(f"Falling back to OpenAI direct: {model}")
                response = openai_direct.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Empty response")
                log.info(f"OpenAI direct success: {model}")
                return content
            except Exception as e:
                log.warning(f"OpenAI {model} failed: {e}")
                last_err = e

    if anthropic_client:
        try:
            log.info("Falling back to Anthropic…")
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_role,
                messages=[{"role": "user", "content": prompt}]
            )
            return "".join(b.text for b in response.content if b.type == "text")
        except Exception as e:
            raise RuntimeError(f"All LLM providers failed. Last: {e}") from e

    raise RuntimeError(f"All LLM providers failed. Last: {last_err}") from last_err


def extract_code_block(raw: str) -> str:
    """Extract Python code from LLM output. Multiple fallback strategies."""
    # Strategy 1: fenced ```python ... ``` block
    m = re.search(r"```python\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Strategy 2: fenced ``` ... ``` block (no language tag)
    m = re.search(r"```\s*\n(.*?)\n```", raw, re.DOTALL)
    if m and ("from manim" in m.group(1) or "class MathExplainer" in m.group(1)):
        return m.group(1).strip()
    # Strategy 3: raw code starts after some planning text
    idx = raw.find("from manim import")
    if idx != -1:
        return raw[idx:].strip()
    # Strategy 4: class definition present somewhere
    idx = raw.find("class MathExplainerScene")
    if idx != -1:
        # walk back to find "from manim"
        pre = raw[:idx].rfind("from manim")
        if pre != -1:
            return raw[pre:].strip()
        return raw[idx:].strip()
    raise ValueError(f"No Manim code found in LLM output:\n{raw[:500]}")


# ── 4. MATH ENGINE ────────────────────────────────────────────────────────────
@dataclass
class MathFacts:
    raw_input:    str
    parsed_clean: bool
    expression:   Optional[str] = None
    variable:     Optional[str] = None
    derivative:   Optional[str] = None
    integral:     Optional[str] = None
    error:        Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def extract_clean_expression(user_input: str) -> str:
    raw = call_llm(
        f"User request: {user_input}\n\n"
        "Extract the core mathematical expression and rewrite it in plain "
        "SymPy-parseable Python syntax (** for powers, * for multiplication, "
        "standard function names: sin, cos, log, exp, sqrt). "
        "Respond with ONLY the expression, nothing else.",
        "You are a precise math-notation translator. Output only the expression.",
        temperature=0.0
    )
    return raw.strip().strip("`")


def analyze_mathematical_properties(user_input: str) -> MathFacts:
    log.info("Parsing expression via SymPy…")
    try:
        expr_str = extract_clean_expression(user_input)
        if "=" in expr_str:
            lhs, rhs = expr_str.split("=", 1)
            expr = sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())
        else:
            expr = sp.sympify(expr_str)

        free_syms = sorted(expr.free_symbols, key=str)
        primary_var = free_syms[0] if free_syms else None

        facts = MathFacts(
            raw_input=user_input,
            parsed_clean=True,
            expression=str(expr),
            variable=str(primary_var) if primary_var else None,
        )
        if primary_var:
            facts.derivative = str(sp.simplify(sp.diff(expr, primary_var)))
            facts.integral   = str(sp.simplify(sp.integrate(expr, primary_var)))
        return facts
    except Exception as e:
        log.error(f"Math parsing failed: {e}")
        return MathFacts(raw_input=user_input, parsed_clean=False, error=str(e))


# ── 5. MATH DOSSIER BUILDER ───────────────────────────────────────────────────
def build_math_dossier(facts: MathFacts) -> dict:
    """Compute exact mathematical facts via SymPy for the LLM to use."""
    if not facts.parsed_clean or not facts.expression or not facts.variable:
        return {}

    var  = sp.Symbol(facts.variable)
    expr = sp.sympify(facts.expression)
    deriv_expr = sp.sympify(facts.derivative) if facts.derivative else None

    # Domain detection
    is_log  = any(a.has(sp.log)  for a in sp.preorder_traversal(expr))
    is_trig = any(isinstance(a, (sp.sin, sp.cos, sp.tan))
                  for a in sp.preorder_traversal(expr))
    if is_log:
        x_min, x_max = 0.05, 10.0
    elif is_trig:
        x_min, x_max = -4 * float(sp.pi), 4 * float(sp.pi)
    else:
        x_min, x_max = -6.0, 6.0

    # Roots
    roots = []
    try:
        r_sols = sp.solve(expr, var)
        for r in r_sols:
            if sp.im(r) == 0:
                rv = float(sp.re(r))
                if x_min <= rv <= x_max:
                    roots.append(round(rv, 4))
        roots = roots[:4]
    except Exception:
        pass

    # Critical points
    crits = []
    if deriv_expr:
        try:
            c_sols = sp.solve(deriv_expr, var)
            for c in c_sols:
                if sp.im(c) == 0:
                    cv = float(sp.re(c))
                    if x_min <= cv <= x_max:
                        crits.append(round(cv, 4))
            crits = crits[:3]
        except Exception:
            pass

    # Critical point values
    f_num = sp.lambdify(var, expr, "numpy")
    crit_vals = []
    for cx in crits:
        try:
            crit_vals.append((cx, round(float(f_num(cx)), 4)))
        except Exception:
            pass

    # LaTeX representations
    fn_latex    = sp.latex(expr)
    deriv_latex = sp.latex(deriv_expr) if deriv_expr else "\\text{n/a}"
    try:
        integ_expr  = sp.integrate(expr, var)
        integ_latex = sp.latex(integ_expr)
    except Exception:
        integ_latex = "\\text{n/a}"

    # Y range — clip to a pedagogically useful window (show 2 x-periods of height)
    x_arr = np.linspace(x_min, x_max, 300)
    try:
        y_arr = np.array(f_num(x_arr), dtype=float)
    except Exception:
        y_arr = np.array([float(f_num(xi)) for xi in x_arr], dtype=float)
    y_arr  = np.where(np.isfinite(y_arr), y_arr, np.nan)
    y_clean = y_arr[np.isfinite(y_arr)]
    if len(y_clean):
        raw_lo = float(np.nanmin(y_clean))
        raw_hi = float(np.nanmax(y_clean))
        span   = max(raw_hi - raw_lo, 1.0)
        # For polynomial/exponential functions that explode, cap the window
        # so the interesting region (near zeros/vertex) is visible
        if span > 40:
            # Focus on a ±10-unit window around the mean y value of key points
            key_ys = []
            for rx in roots:
                try: key_ys.append(float(f_num(rx)))
                except: pass
            for cx in crits:
                try: key_ys.append(float(f_num(cx)))
                except: pass
            if key_ys:
                center = float(np.mean(key_ys))
                raw_lo = center - span * 0.15
                raw_hi = center + span * 0.15
            else:
                # just show ±15 around 0
                raw_lo = max(raw_lo, -20)
                raw_hi = min(raw_hi, 20)
        y_min = round(raw_lo - abs(raw_lo) * 0.15 - 1.0, 1)
        y_max = round(raw_hi + abs(raw_hi) * 0.15 + 1.0, 1)
    else:
        y_min, y_max = -5.0, 10.0

    return {
        "variable":     facts.variable,
        "expression":   facts.expression,
        "fn_latex":     fn_latex,
        "deriv_expr":   facts.derivative or "",
        "deriv_latex":  deriv_latex,
        "integ_latex":  integ_latex,
        "roots":        roots,
        "crits":        crits,
        "crit_vals":    crit_vals,
        "x_min":        round(x_min, 2),
        "x_max":        round(x_max, 2),
        "y_min":        y_min,
        "y_max":        y_max,
        "is_trig":      is_trig,
        "is_log":       is_log,
    }


# ── 6. FILM CONTRACT (The Core System Prompt) ─────────────────────────────────
MANIM_FILM_CONTRACT = """
You are an expert Manim Community Edition developer creating 3Blue1Brown-style
math explainer animations for high school students. Your output must be a single,
complete, runnable Python file.

═══════════════════════════════════════════════════════
PEDAGOGICAL RULES (from Math-To-Manim architecture)
═══════════════════════════════════════════════════════
1. HEADLINE BEFORE SYMBOLS — Start every major section with a plain-English
   statement in large text. Earn the formula; never open with it.
2. EARN THE NOTATION — Introduce each piece of notation only after the student
   understands the concept it names.
3. CAPTIONS — Every formula on screen has a plain-language caption below it.
   Max 14 words. Italic style. Font size 26-28.
4. ONE IDEA PER SECTION — Each section introduces exactly one concept.
5. BIG ZOOM MOMENT — Somewhere around 60-70% through the scene, have a
   memorable moment: a formula morphing, a key point appearing dramatically.
6. PACING — Leave breathing room. Use self.wait(0.8) to self.wait(2.0)
   between concepts. Never rush.

═══════════════════════════════════════════════════════
MANIM CE TECHNICAL RULES (strict — do not violate)
═══════════════════════════════════════════════════════
Required imports:
  from manim import *
  import numpy as np

Scene class:
  class MathExplainerScene(Scene):
  First line of construct(): self.camera.background_color = "#0f0f0f"

Color palette (3B1B style):
  CURVE_COLOR   = "#58C4DD"   # bright blue for the main function
  ZERO_COLOR    = "#83C167"   # green for zeros/roots
  DERIV_COLOR   = "#FC6255"   # red/coral for derivative/tangent
  INTEG_COLOR   = "#9B59B6"   # lavender for integral shading
  CRIT_COLOR    = "#FFFF00"   # yellow for critical points
  HEADLINE_CLR  = "#FFFFFF"   # white for headlines
  CAPTION_CLR   = "#888888"   # grey for captions
  BG_COLOR      = "#0f0f0f"   # background

Text and formulas:
  - Headlines: Text("...", font_size=52, color=WHITE, weight=BOLD)
  - Captions:  Text("...", font_size=26, color="#888888", slant=ITALIC)
               place with .to_edge(DOWN, buff=0.5)
  - Formulas:  MathTex(r"...", font_size=48, color=WHITE)
  - NEVER index into MathTex parts (e.g. formula[0][3:5]) — this crashes.
    Instead, create SEPARATE MathTex objects for terms you want to highlight,
    or use Indicate(formula) / SurroundingRectangle(formula) for emphasis.

Axes and graphs:
  axes = Axes(
      x_range=[x_min, x_max, 1],
      y_range=[y_min, y_max, 1],
      x_length=9, y_length=5.5,
      axis_config={"color": "#555555", "include_numbers": True,
                   "numbers_to_exclude": []},
      tips=False,
  )
  axes.center().shift(DOWN * 0.3)
  graph = axes.plot(lambda x: ..., color=CURVE_COLOR, stroke_width=3.5)
  
  Always Create(axes) first, then Create(graph).
  Use axes.c2p(x, y) to get screen coordinates (NOT coords_to_point).
  Use Dot(axes.c2p(x, y), color=..., radius=0.1) for points.
  Use axes.get_area(graph, x_range=[a, b], color=INTEG_COLOR, opacity=0.35)
      for integral shading.

Animations:
  self.play(Write(text), run_time=1.2)
  self.play(FadeIn(obj, shift=UP*0.2), run_time=0.8)
  self.play(Create(axes), run_time=1.5)
  self.play(Create(graph), run_time=2.5)
  self.play(Indicate(formula, color=YELLOW), run_time=1.0)
  self.play(obj.animate.set_color(RED), run_time=0.6)
  self.play(FadeOut(obj), run_time=0.5)
  self.play(Transform(old, new), run_time=1.0)

Labels on graph:
  label = MathTex(r"x=1", font_size=30, color=ZERO_COLOR)
  label.next_to(dot, UP, buff=0.15)

VGroup for positioning:
  group = VGroup(formula, caption)
  group.arrange(DOWN, buff=0.4).move_to(ORIGIN)

Tangent line on axes:
  tangent = axes.plot(lambda x: slope*(x - x0) + y0,
                      color=DERIV_COLOR, stroke_width=2.5,
                      x_range=[x0-2, x0+2])

Scene structure REQUIRED (in this order):
  1. HEADLINE section — plain-language title, what this function is
  2. FORMULA REVEAL — show f(x) = ..., explain each major term
  3. GRAPH section — draw axes + curve, with caption
  4. KEY POINTS — mark zeros (green dots + labels), critical points (yellow)
  5. DERIVATIVE — new headline, show f'(x) formula, add tangent line to graph
  6. INTEGRAL — new headline, show ∫f dx formula, shade area
  7. SUMMARY — brief recap with all elements visible together

Target duration: 75-100 seconds total. Use run_time + self.wait() accordingly.

SAFETY RULES:
  - No try/except blocks inside construct() — if something might fail, just omit it
  - No file I/O, no network calls, no subprocess
  - No f-strings inside MathTex — always use raw strings r"..."
  - All lambda functions must use numpy-safe math (np.sin, not math.sin)
  - If roots list is empty, skip the zeros section gracefully
  - Every self.play() must have at least one animation argument
  - After FadeOut all objects, the screen must be clear before new section
""".strip()


CURRICULUM_SYSTEM = """You are a mathematical pedagogy expert who plans explainer
video narratives for high school students. You write concisely and precisely.
Always output valid JSON."""


# ── 6b. NUMPY HELPER (module-level) ──────────────────────────────────────────
def _to_numpy(s: str) -> str:
    """Make a SymPy expression string safe for numpy lambdas in Manim scenes."""
    s = s.replace('exp(', 'np.exp(')
    s = s.replace('sin(', 'np.sin(')
    s = s.replace('cos(', 'np.cos(')
    s = s.replace('tan(', 'np.tan(')
    s = s.replace('log(', 'np.log(')
    s = s.replace('sqrt(', 'np.sqrt(')
    s = re.sub(r'\bpi\b', 'np.pi', s)
    s = re.sub(r'\bE\b',  'np.e',  s)
    return s


# ── 6c. DETERMINISTIC TEMPLATE GENERATOR (no LLM) ────────────────────────────
def generate_manim_template(dossier: dict) -> str:
    """
    3B1B-style Manim scene: the graph is drawn immediately and stays on screen
    the ENTIRE time. Mathematical features (zeros, vertex, derivative curve,
    integral shading) are built up progressively ON TOP of the graph.
    No slideshow slides — one continuous visual story.
    """
    var         = dossier['variable']
    fn_latex    = dossier['fn_latex'].replace('"', "'")
    drv_latex   = dossier['deriv_latex'].replace('"', "'")
    int_latex   = dossier['integ_latex'].replace('"', "'")
    lambda_body = _to_numpy(dossier['expression'])
    deriv_body  = _to_numpy(dossier.get('deriv_expr') or f'0*{var}')

    x_min = float(dossier['x_min'])
    x_max = float(dossier['x_max'])
    y_min = float(dossier['y_min'])
    y_max = float(dossier['y_max'])
    is_log = bool(dossier.get('is_log', False))

    x_step = max(0.5, round((x_max - x_min) / 6, 1))
    y_step = max(0.5, round((y_max - y_min) / 4, 1))

    roots = [round(float(r), 2) for r in (dossier.get('roots') or [])][:5]
    crits = [(round(float(cx), 2), round(float(cy), 2))
              for cx, cy in (dossier.get('crit_vals') or [])][:3]

    # Integral shading bounds
    left_pad  = 0.05 if is_log else 0.0
    area_l = round(x_min + (x_max - x_min) * (0.1 + left_pad), 2)
    area_r = round(x_min + (x_max - x_min) * 0.9, 2)

    # ─── Build the scene line-by-line ─────────────────────────────────────────
    L = []   # list of code lines

    def add(*args):
        for s in args:
            L.append(s)

    add(
        "from manim import *",
        "import numpy as np",
        "",
        "class MathExplainerScene(Scene):",
        "    def construct(self):",
        "        self.camera.background_color = \"#0f0f0f\"",
        "",
        "        # ═══ PHASE 1: The graph — drawn first, stays on screen the whole time",
        "        axes = Axes(",
        f"            x_range=[{x_min}, {x_max}, {x_step}],",
        f"            y_range=[{y_min:.2f}, {y_max:.2f}, {y_step}],",
        "            x_length=9, y_length=5.5,",
        "            axis_config={\"color\": \"#2a2a2a\", \"stroke_width\": 2},",
        "            tips=True,",
        "        )",
        "        axes.center().shift(DOWN * 0.3)",
        "        ax_lbl = axes.get_axis_labels(",
        "            x_label=MathTex(r\"" + var + "\", font_size=24, color=\"#555\"),",
        "            y_label=MathTex(r\"f(" + var + ")\", font_size=24, color=\"#555\"),",
        "        )",
        "        graph = axes.plot(",
        f"            lambda {var}: {lambda_body},",
        f"            x_range=[{x_min}, {x_max}],",
        "            color=\"#58C4DD\", stroke_width=3.5, use_smoothing=True,",
        "        )",
        "        # Formula label lives in the top-left corner throughout",
        "        f_lbl = MathTex(",
        "            r\"f(" + var + ") = " + fn_latex + "\",",
        "            font_size=34, color=WHITE,",
        "        )",
        "        f_lbl.to_corner(UL, buff=0.5)",
        "",
        "        # Axes appear, then label, then curve traces out",
        "        self.play(Create(axes), Write(ax_lbl), run_time=1.5)",
        "        self.play(Write(f_lbl), run_time=0.8)",
        "        self.play(Create(graph), run_time=2.5)",
        "        self.wait(0.8)",
        "",
    )

    # ─── PHASE 2: Zeros (green dots pop onto the graph) ───────────────────────
    if roots:
        add(
            "        # ═══ PHASE 2: Zeros — where the curve crosses the x-axis",
            "        z_cap = Text(",
            "            \"Zeros: where f(" + var + ") = 0\",",
            "            font_size=26, color=\"#83C167\", slant=ITALIC,",
            "        )",
            "        z_cap.to_edge(DOWN, buff=0.4)",
            "        self.play(FadeIn(z_cap), run_time=0.5)",
        )
        for i, rx in enumerate(roots):
            add(
                f"        zdot{i} = Dot(axes.c2p({rx}, 0), color=\"#83C167\", radius=0.12)",
                "        zlbl" + str(i) + " = MathTex(",
                "            r\"" + var + "=" + str(rx) + "\",",
                "            font_size=30, color=\"#83C167\",",
                "        )",
                f"        zlbl{i}.next_to(zdot{i}, UP, buff=0.2)",
                f"        self.play(FadeIn(zdot{i}, scale=2.5), Write(zlbl{i}), run_time=0.8)",
                f"        self.play(Indicate(zdot{i}, color=WHITE, scale_factor=1.4), run_time=0.5)",
            )
        add(
            "        self.wait(1.0)",
            "        self.play(FadeOut(z_cap))",
            "",
        )

    # ─── PHASE 3: Critical points (vertex/extremum marked on graph) ───────────
    if crits:
        cx0, cy0 = crits[0]
        cy0s = round(float(max(y_min + 0.01, min(y_max - 0.01, cy0))), 2)
        crit_word = "Minimum" if cy0 < (y_min + y_max) / 2 else "Maximum"
        add(
            "        # ═══ PHASE 3: Critical point — vertex / extremum",
            "        c_cap = Text(",
            "            \"" + crit_word + " at x = " + str(cx0) + "\",",
            "            font_size=26, color=\"#FFBE5C\", slant=ITALIC,",
            "        )",
            "        c_cap.to_edge(DOWN, buff=0.4)",
            f"        cdot = Dot(axes.c2p({cx0}, {cy0s}), color=\"#FFBE5C\", radius=0.14)",
            "        clbl = MathTex(",
            "            r\"(" + str(cx0) + ",\\\\ " + str(cy0s) + ")\",",
            "            font_size=28, color=\"#FFBE5C\",",
            "        )",
            "        clbl.next_to(cdot, UR, buff=0.18)",
            "        # Horizontal dashed line showing zero slope at the extremum",
            f"        h_dash = DashedLine(",
            f"            axes.c2p({cx0} - 1.5, {cy0s}),",
            f"            axes.c2p({cx0} + 1.5, {cy0s}),",
            "            color=\"#FFBE5C\", dash_length=0.1, stroke_opacity=0.5,",
            "        )",
            "        self.play(FadeIn(cdot, scale=2.5), Write(clbl), FadeIn(c_cap), run_time=1.0)",
            "        self.play(Create(h_dash), run_time=0.7)",
            "        self.wait(1.2)",
            "        self.play(FadeOut(c_cap), FadeOut(h_dash))",
            "",
        )

    # ─── PHASE 4: Derivative curve drawn on the SAME axes ─────────────────────
    add(
        "        # ═══ PHASE 4: Derivative — curve drawn ON the same axes",
        "        drv_cap = Text(",
        "            \"Derivative: slope at every point\",",
        "            font_size=26, color=\"#d97757\", slant=ITALIC,",
        "        )",
        "        drv_cap.to_edge(DOWN, buff=0.4)",
        "        drv_lbl = MathTex(",
        "            r\"f'(" + var + ") = " + drv_latex + "\",",
        "            font_size=30, color=\"#d97757\",",
        "        )",
        "        drv_lbl.to_corner(UR, buff=0.5)",
        "        self.play(Write(drv_lbl), FadeIn(drv_cap), run_time=1.0)",
        "        deriv_graph = axes.plot(",
        f"            lambda {var}: {deriv_body},",
        f"            x_range=[{x_min}, {x_max}],",
        "            color=\"#d97757\", stroke_width=2.5, stroke_opacity=0.9,",
        "        )",
        "        # Derivative curve traces out alongside the original",
        "        self.play(Create(deriv_graph), run_time=2.2)",
        "        self.wait(1.5)",
        "        self.play(FadeOut(drv_cap))",
        "",
    )

    # ─── PHASE 5: Integral shading on the SAME graph ──────────────────────────
    add(
        "        # ═══ PHASE 5: Integral — area shaded under the original curve",
        "        int_cap = Text(",
        "            \"Integral: the area under the curve\",",
        "            font_size=26, color=\"#9ECC5A\", slant=ITALIC,",
        "        )",
        "        int_cap.to_edge(DOWN, buff=0.4)",
        "        int_lbl = MathTex(",
        "            r\"\\int f\\,d" + var + " = " + int_latex + " + C\",",
        "            font_size=26, color=\"#9ECC5A\",",
        "        )",
        "        int_lbl.to_corner(DR, buff=0.5)",
        "        self.play(FadeOut(drv_lbl), run_time=0.4)",
        "        self.play(Write(int_lbl), FadeIn(int_cap), run_time=1.0)",
        "        area = axes.get_area(",
        f"            graph, x_range=[{area_l}, {area_r}],",
        "            color=\"#9ECC5A\", opacity=0.3,",
        "        )",
        "        self.play(FadeIn(area), run_time=1.8)",
        "        self.wait(1.5)",
        "        self.play(FadeOut(int_cap), FadeOut(int_lbl), FadeOut(area))",
        "",
    )

    # ─── PHASE 6: Summary card ────────────────────────────────────────────────
    summary_rows = [
        "f(" + var + ") = " + fn_latex,
        "f'(" + var + ") = " + drv_latex,
    ]
    if roots:
        summary_rows.append("zeros: x = " + ",  x = ".join(str(r) for r in roots[:3]))
    if crits:
        cx0, cy0 = crits[0]
        summary_rows.append("vertex: (" + str(cx0) + ", " + str(round(cy0, 2)) + ")")

    add(
        "        # ═══ PHASE 6: Summary",
        "        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)",
        "        s_title = Text(\"Summary\", font_size=46, color=WHITE, weight=BOLD)",
        "        s_title.to_edge(UP, buff=0.7)",
        "        self.play(Write(s_title), run_time=0.7)",
    )
    for i, row in enumerate(summary_rows[:4]):
        row_safe = row.replace('"', "'")
        anchor = f"s_title" if i == 0 else f"row{i-1}"
        buff   = "0.5" if i == 0 else "0.28"
        add(
            f"        row{i} = Text(\"{row_safe}\", font_size=27, color=\"#a0a0a0\")",
            f"        row{i}.next_to({anchor}, DOWN, buff={buff})",
            f"        self.play(FadeIn(row{i}, shift=UP * 0.12), run_time=0.45)",
        )
    add(
        "        self.wait(2.5)",
        "        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.7)",
    )

    return "\n".join(L) + "\n"


CODEGEN_SYSTEM = """You are a Manim CE code generator. You output ONLY Python code.

CRITICAL RULES — violating any of these will break the pipeline:
1. Your ENTIRE response must be one ```python ... ``` fenced code block.
2. Do NOT write any explanation, planning, outline, or prose — not before, not after, not inside comments that explain what you are about to do.
3. The very first characters of your response must be: ```python
4. The very last characters of your response must be: ```
5. The class MUST be named MathExplainerScene and subclass Scene.
6. The first line of construct() MUST be: self.camera.background_color = "#0f0f0f"

If you feel the urge to write an outline or planning notes, suppress it completely and go straight to the code."""


# ── 7. MULTI-STAGE PIPELINE ───────────────────────────────────────────────────
def build_curriculum(dossier: dict) -> str:
    """Stage 1: Generate pedagogical narrative and shot outline."""
    prompt = f"""
You are planning a 75-100 second math explainer video for high school students.
The function is f({dossier['variable']}) = {dossier['fn_latex']}

Mathematical facts (computed exactly by SymPy):
- Expression: {dossier['expression']}
- Zeros (where f=0): {dossier['roots']} (x values)
- Critical points (where f'=0): {dossier['crit_vals']} (x, y pairs)
- Derivative: f'({dossier['variable']}) = {dossier['deriv_latex']}
- Integral: ∫f d{dossier['variable']} = {dossier['integ_latex']} + C
- Graph x range: [{dossier['x_min']}, {dossier['x_max']}]
- Graph y range: [{dossier['y_min']}, {dossier['y_max']}]
- Is trigonometric: {dossier['is_trig']}
- Is logarithmic: {dossier['is_log']}

Output JSON with this structure:
{{
  "title": "Short cinematic title for this explainer (e.g. 'The Parabola Within')",
  "headline": "Plain-English opening statement (1 sentence, no symbols)",
  "sections": [
    {{"name": "Introduction", "concept": "...", "duration_s": 12}},
    {{"name": "The Formula", "concept": "...", "duration_s": 10}},
    {{"name": "The Graph", "concept": "...", "duration_s": 15}},
    {{"name": "Zeros & Key Points", "concept": "...", "duration_s": 15}},
    {{"name": "The Derivative", "concept": "...", "duration_s": 15}},
    {{"name": "The Integral", "concept": "...", "duration_s": 15}},
    {{"name": "Summary", "concept": "...", "duration_s": 8}}
  ],
  "big_zoom_moment": "Describe the most memorable visual moment in the video",
  "key_captions": {{
    "formula": "Caption for the main formula (max 10 words)",
    "zeros": "Caption for the roots section (max 10 words)",
    "derivative": "Caption for the derivative (max 10 words)",
    "integral": "Caption for the integral (max 10 words)"
  }}
}}
""".strip()
    return call_llm(prompt, CURRICULUM_SYSTEM, temperature=0.2, max_tokens=1000)


def generate_manim_code(dossier: dict, curriculum_json: str) -> str:
    """Stage 2: Generate complete Manim scene from dossier + curriculum."""
    try:
        curriculum = json.loads(re.search(r'\{.*\}', curriculum_json, re.DOTALL).group(0))
    except Exception:
        curriculum = {"title": "Math Explainer", "headline": "Let's explore this function."}

    expr_str = dossier['expression']
    var      = dossier['variable']

    lambda_str = _to_numpy(expr_str)
    _raw_deriv = dossier.get('deriv_expr') or ''
    deriv_str  = _to_numpy(_raw_deriv) if _raw_deriv else '0 * ' + var

    roots_info = ""
    if dossier['roots']:
        roots_info = f"Zeros at x = {dossier['roots']} — use green dots with MathTex labels"
    else:
        roots_info = "No real zeros in the displayed range — skip the zeros section"

    crits_info = ""
    if dossier['crit_vals']:
        crits_info = f"Critical points at {dossier['crit_vals']} — use yellow dots with labels showing (x, f(x))"
    else:
        crits_info = "No critical points in the displayed range — skip critical point markers"

    prompt = f"""
{MANIM_FILM_CONTRACT}

═══════════════════════════════════════════════════════
FUNCTION DETAILS
═══════════════════════════════════════════════════════
Title: {curriculum.get('title', 'Math Explainer')}
Opening headline: "{curriculum.get('headline', 'Let us explore this function.')}"

Function: f({var}) = {dossier['fn_latex']}  (SymPy: {expr_str})
Python lambda:  lambda {var}: {lambda_str}

Derivative: f'({var}) = {dossier['deriv_latex']}  (SymPy: {dossier.get('deriv_expr', 'n/a')})
Derivative lambda: lambda {var}: {deriv_str if deriv_str else '0'}

Integral: ∫f d{var} = {dossier['integ_latex']} + C

Axes ranges:
  x_range = [{dossier['x_min']}, {dossier['x_max']}, 1]
  y_range = [{dossier['y_min']}, {dossier['y_max']}, 1]

Key points:
  {roots_info}
  {crits_info}

Captions to use:
  formula caption:    "{curriculum.get('key_captions', {}).get('formula', 'This is our function.')}"
  zeros caption:      "{curriculum.get('key_captions', {}).get('zeros', 'Where the curve crosses zero.')}"
  derivative caption: "{curriculum.get('key_captions', {}).get('derivative', 'The instantaneous rate of change.')}"
  integral caption:   "{curriculum.get('key_captions', {}).get('integral', 'The area under the curve.')}"

Big zoom moment: {curriculum.get('big_zoom_moment', 'When the zeros appear on the graph.')}

═══════════════════════════════════════════════════════
OUTPUT: One complete ```python code block.
The class MUST be named MathExplainerScene.
Use ONLY the techniques described in the Film Contract above.
Write enough self.wait() calls that the total runtime is 75-100 seconds.
═══════════════════════════════════════════════════════
""".strip()

    raw = call_llm(prompt, CODEGEN_SYSTEM, temperature=0.05, max_tokens=4000)
    try:
        return extract_code_block(raw)
    except ValueError:
        log.warning("First code-gen attempt had no code block — retrying with explicit prompt.")
        retry_prompt = (
            f"Write a complete Manim CE scene for f({var}) = {expr_str}.\n"
            f"x_range = [{dossier['x_min']}, {dossier['x_max']}, 1]\n"
            f"y_range = [{dossier['y_min']}, {dossier['y_max']}, 1]\n"
            f"lambda {var}: {lambda_str}\n\n"
            "Requirements:\n"
            "- Class named MathExplainerScene(Scene)\n"
            "- self.camera.background_color = '#0f0f0f' first\n"
            "- Show formula, then graph with axes, then key points\n"
            "- Use np. prefix for all math functions in lambdas\n"
            "- Use axes.c2p(x, y) for coordinates\n"
            "- ONLY output code, nothing else."
        )
        raw2 = call_llm(
            retry_prompt,
            "Output ONLY a ```python code block. No prose. No explanation. Just code.",
            temperature=0.0,
            max_tokens=4000,
        )
        return extract_code_block(raw2)


def heal_manim_code(broken_code: str, error_log: str, attempt: int) -> str:
    """Self-healing: fix the Manim code based on the error output."""
    prompt = f"""
This Manim CE scene failed to render. Fix it.

REPAIR PASS {attempt}

Common causes of Manim errors:
- Indexing into MathTex (formula[0][3:5]) → replace with Indicate(formula) or SurroundingRectangle
- Wrong lambda (using math.sin instead of np.sin, or plain expressions without np.)
- axes.coords_to_point() → use axes.c2p() instead
- get_area() with wrong syntax → use axes.get_area(graph, x_range=[a,b], color=..., opacity=...)
- self.play() with no animation → remove or add FadeIn/FadeOut
- Objects referenced before being created → reorder
- Missing self.wait() → add after every self.play() block
- Duplicate variable names → rename

BROKEN CODE:
```python
{broken_code}
```

ERROR:
```
{error_log[-3000:]}
```

Output the COMPLETE fixed code in a single ```python block.
Keep MathExplainerScene as the class name.
""".strip()
    raw = call_llm(prompt, CODEGEN_SYSTEM, temperature=0.1, max_tokens=4000)
    return extract_code_block(raw)


# ── 8. MANIM EXECUTOR ─────────────────────────────────────────────────────────
# Fix: Replit's Nix has pycairo for Python 3.12 in sys.path; pip-installed
# pycairo for Python 3.11 lives in .pythonlibs. Force .pythonlibs first.
_PYLIBS = "/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages"


def _manim_env() -> dict:
    """Build subprocess env with correct PYTHONPATH so pycairo is found."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_PYLIBS}:{existing}" if existing else _PYLIBS
    # Suppress TeX live warnings that can clutter logs
    env.setdefault("TEXMFHOME", "/tmp/texmf")
    return env


def _find_manim_video(work_dir: str) -> Optional[str]:
    """
    Walk the manim media output tree and return the final concatenated MP4.
    Manim writes: videos/scene/<quality>/MathExplainerScene.mp4  (the full video)
    and also:     videos/scene/<quality>/partial_movie_files/MathExplainerScene/uncached_*.mp4
    We must return the full video, NOT a partial clip.
    """
    # Pass 1: file named exactly MathExplainerScene.mp4, not inside partial_movie_files
    for root, dirs, files in os.walk(work_dir):
        # Skip partial clip directories entirely
        dirs[:] = [d for d in dirs if d != "partial_movie_files"]
        for f in files:
            if f == "MathExplainerScene.mp4":
                return os.path.join(root, f)
    # Pass 2: any .mp4 outside partial_movie_files
    for root, dirs, files in os.walk(work_dir):
        dirs[:] = [d for d in dirs if d != "partial_movie_files"]
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


def execute_manim_scene(code: str, job_id: str, quality: str = "m") -> Optional[str]:
    """
    Write Manim scene to disk and render it.
    quality: 'l' (480p15), 'm' (720p30), 'h' (1080p60)
    Returns the path to the rendered MP4 or None on failure.
    """
    work_dir = os.path.abspath(f"./runs/{job_id}")
    os.makedirs(work_dir, exist_ok=True)

    scene_file = os.path.join(work_dir, "scene.py")
    with open(scene_file, "w") as f:
        f.write(code)

    # -q{quality} flag, --media_dir keeps output in our work_dir
    cmd = [
        sys.executable, "-m", "manim",
        f"-q{quality}",
        "--media_dir", work_dir,
        "--disable_caching",
        scene_file,
        "MathExplainerScene",
    ]
    log.info(f"Running Manim: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,
            env=_manim_env(),
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        log.error("Manim render timed out after 240s")
        return None

    # Always save error log so healing can read it without re-running Manim
    error_log_path = os.path.join(work_dir, "error.log")
    with open(error_log_path, "w") as f:
        f.write(result.stdout + "\n" + result.stderr)

    if result.returncode == 0:
        video = _find_manim_video(work_dir)
        if video:
            log.info(f"Manim render OK → {video}")
            out_dir  = os.path.abspath(f"./output_{job_id}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "render.mp4")
            shutil.copy2(video, out_path)
            return out_path
        log.warning("Manim exited 0 but no .mp4 found")
    else:
        log.warning(f"Manim failed (exit {result.returncode})")
        log.warning(f"STDERR tail: {result.stderr[-1500:]}")

    return None


def _manim_error_log(job_id: str) -> str:
    """Read cached error log written by execute_manim_scene — never re-runs Manim."""
    path = os.path.join(os.path.abspath(f"./runs/{job_id}"), "error.log")
    try:
        with open(path) as f:
            return f.read()[-3000:]
    except FileNotFoundError:
        return "No error log found for this job."


# ── 9. TOP-LEVEL RENDER ENTRY POINT ───────────────────────────────────────────
def render_3b1b_style(facts: MathFacts, job_id: str,
                       push_event=None) -> Optional[str]:
    """
    Math-To-Manim inspired pipeline (streamlined — no curriculum step):
      1. Build math dossier (SymPy, ~0s)
      2. Code-gen agent  (1 LLM call, ~30-50s)
      3. Execute Manim at 480p  (~15-25s)
      4. Self-heal up to MAX_RETRIES times if it fails
    """
    def _push(stage: str, message: str, progress: int = None):
        if push_event:
            d = {"stage": stage, "message": message}
            if progress is not None:
                d["progress"] = progress
            try:
                push_event(d)
            except Exception:
                pass

    if not facts.parsed_clean or not facts.expression or not facts.variable:
        log.warning("Cannot render: incomplete math facts.")
        return None

    # ── Stage 1: Math dossier ─────────────────────────────────────────
    _push("analyzing", "① Computing exact math facts (roots, derivative, integral)…", 10)
    log.info(f"[{job_id}] Building math dossier…")
    dossier = build_math_dossier(facts)
    if not dossier:
        log.error("Dossier build failed.")
        return None

    # ── Stage 2: Code generation (LLM, with deterministic fallback) ──
    _push("codegen", "② Writing Manim animation code…", 30)
    log.info(f"[{job_id}] Code-gen agent…")
    used_template = False
    try:
        code = generate_manim_code(dossier, "{}")
    except Exception as e:
        log.warning(f"LLM code-gen failed ({e}). Using built-in template.")
        _push("codegen", "② LLM unavailable — using built-in template…", 35)
        try:
            code = generate_manim_template(dossier)
            used_template = True
        except Exception as te:
            log.error(f"Template generation also failed: {te}")
            _push("error", f"Code generation failed: {e}")
            return None

    log.info(f"[{job_id}] Generated {len(code.splitlines())} lines of Manim code.")
    work_dir = os.path.abspath(f"./runs/{job_id}")
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "scene_v0.py"), "w") as f:
        f.write(code)

    # ── Stage 3+: Execute + self-heal ─────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        pct = 50 + attempt * 12          # 62 → 74 → 86
        _push(f"rendering_attempt_{attempt}",
              f"③ Manim rendering (pass {attempt}/{MAX_RETRIES}) — please wait ~20s…",
              pct)
        log.info(f"[{job_id}] Manim attempt {attempt}/{MAX_RETRIES}…")
        update_job_state(job_id, f"rendering_attempt_{attempt}")

        video = execute_manim_scene(code, job_id, quality="l")   # always 480p15 for speed
        if video:
            update_job_state(job_id, "completed", code=code, video_path=video)
            return video

        if attempt < MAX_RETRIES:
            _push("healing",
                  f"⟳ Auto-fixing render errors (attempt {attempt})…",
                  pct + 6)
            log.info(f"[{job_id}] Healing…")
            error_log = _manim_error_log(job_id)
            try:
                code = heal_manim_code(code, error_log, attempt)
                with open(os.path.join(work_dir, f"scene_v{attempt}.py"), "w") as f:
                    f.write(code)
            except Exception as e:
                log.error(f"Healing failed: {e}")
                break

    log.error(f"[{job_id}] All render attempts exhausted.")
    update_job_state(job_id, "failed", error="All Manim render attempts failed.")
    return None


# ── EXECUTION ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    requests = [
        {"job_id": "job_001", "eq": "x**2 - 4*x + 3"},
    ]
    ui_params = {"x_range": [-5, 5, 1]}
    log.info("Pushing tasks to worker queue…")
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(
            lambda r: None,  # placeholder — real jobs go through app.py
            req
        ) for req in requests]
        for f in futures:
            f.result()
    log.info("Done.")
