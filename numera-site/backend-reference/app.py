import os, uuid, json, logging, threading, re, hashlib, sqlite3, secrets
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
import requests as http
from flask import (Flask, render_template, request, jsonify, send_file,
                   Response, redirect, url_for, flash, session)
from flask_cors import CORS
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
load_dotenv()

from main import (
    init_db as init_math_db, render_3b1b_style, set_cache, get_cache,
    update_job_state, call_llm, MathFacts
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', secrets.token_hex(32))
CORS(app)

# ── Flask-Login ────────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message = ''

# ── User DB ────────────────────────────────────────────────────────
AUTH_DB = os.path.abspath('./users.db')

def init_user_db():
    conn = sqlite3.connect(AUTH_DB, timeout=30)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            name          TEXT,
            password_hash TEXT,
            google_id     TEXT,
            created_at    TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()
    conn.close()

class User(UserMixin):
    def __init__(self, uid, email, name, google_id=None):
        self.id      = str(uid)
        self.email   = email
        self.name    = name or email.split('@')[0]
        self.google_id = google_id

def _get_user_row(uid):
    conn = sqlite3.connect(AUTH_DB, timeout=30)
    row  = conn.execute(
        'SELECT id,email,name,google_id FROM users WHERE id=?', (uid,)
    ).fetchone()
    conn.close()
    return row

def _get_user_by_email(email):
    conn = sqlite3.connect(AUTH_DB, timeout=30)
    row  = conn.execute(
        'SELECT id,email,name,password_hash,google_id FROM users WHERE email=?',
        (email,)
    ).fetchone()
    conn.close()
    return row

def _create_user(email, name, password=None, google_id=None):
    pw_hash = generate_password_hash(password) if password else None
    conn = sqlite3.connect(AUTH_DB, timeout=30)
    try:
        conn.execute(
            'INSERT INTO users (email,name,password_hash,google_id) VALUES (?,?,?,?)',
            (email, name, pw_hash, google_id)
        )
        conn.commit()
        uid = conn.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()[0]
        conn.close()
        return uid
    except sqlite3.IntegrityError:
        conn.close()
        return None

@login_manager.user_loader
def load_user(uid):
    row = _get_user_row(uid)
    return User(*row) if row else None

# ── CSRF simple token ──────────────────────────────────────────────
@app.context_processor
def inject_csrf():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    return dict(csrf_token=lambda: session['csrf_token'])

def _check_csrf():
    token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    return token == session.get('csrf_token')

# ── Google OAuth helpers ───────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')

def _google_redirect_uri():
    base = os.environ.get('APP_BASE_URL', '').rstrip('/')
    if not base:
        base = request.url_root.rstrip('/')
    return f"{base}/auth/google/callback"

# ── Job events store ───────────────────────────────────────────────
_job_events: dict[str, list[str]] = {}
_job_lock = threading.Lock()

# ── Equation library ───────────────────────────────────────────────
EQUATION_LIBRARY = [
    {"id": "quadratic",   "name": "Quadratic",    "sympy": "x**2 - 4*x + 3",   "latex": "x^2 - 4x + 3",              "category": "Algebra",    "color": "#89b4fa"},
    {"id": "sine",        "name": "Sine Wave",     "sympy": "sin(x)",            "latex": "\\sin(x)",                   "category": "Trig",       "color": "#a6e3a1"},
    {"id": "gaussian",    "name": "Bell Curve",    "sympy": "exp(-x**2)",        "latex": "e^{-x^2}",                   "category": "Statistics", "color": "#cba6f7"},
    {"id": "cubic",       "name": "Cubic",         "sympy": "x**3 - 3*x",       "latex": "x^3 - 3x",                   "category": "Algebra",    "color": "#f38ba8"},
    {"id": "sigmoid",     "name": "Sigmoid",       "sympy": "1/(1 + exp(-x))",  "latex": "\\frac{1}{1+e^{-x}}",        "category": "ML / Stats", "color": "#fab387"},
    {"id": "exponential", "name": "Exponential",   "sympy": "exp(x)",            "latex": "e^x",                        "category": "Calculus",   "color": "#f9e2af"},
    {"id": "logarithm",   "name": "Natural Log",   "sympy": "log(x)",            "latex": "\\ln(x)",                    "category": "Calculus",   "color": "#94e2d5"},
    {"id": "sinc",        "name": "Sinc",          "sympy": "sin(x)/x",          "latex": "\\frac{\\sin(x)}{x}",        "category": "Signal",     "color": "#89dceb"},
    {"id": "cos_sin",     "name": "cos − sin",     "sympy": "cos(x) - sin(x)",  "latex": "\\cos(x) - \\sin(x)",        "category": "Trig",       "color": "#b4befe"},
    {"id": "quartic",     "name": "Quartic",       "sympy": "x**4 - 4*x**2",    "latex": "x^4 - 4x^2",                 "category": "Algebra",    "color": "#74c7ec"},
]

_SYMPY_LOCALS = {
    'e': sp.E, 'pi': sp.pi, 'exp': sp.exp,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt,
    'Abs': sp.Abs, 'abs': sp.Abs, 'asin': sp.asin,
    'acos': sp.acos, 'atan': sp.atan, 'sinh': sp.sinh,
    'cosh': sp.cosh, 'tanh': sp.tanh,
}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def fast_parse(equation_str: str) -> dict | None:
    clean = equation_str.strip().replace('^', '**')
    clean = re.sub(r'\bln\b', 'log', clean)
    try:
        expr = parse_expr(clean, transformations=_TRANSFORMS, local_dict=_SYMPY_LOCALS)
        free = sorted(expr.free_symbols, key=str)
        var  = free[0] if free else None
        out  = {
            "expression": str(expr),
            "latex":      sp.latex(expr),
            "variable":   str(var) if var else "x",
        }
        if var:
            deriv = sp.simplify(sp.diff(expr, var))
            integ = sp.simplify(sp.integrate(expr, var))
            out["derivative"]       = str(deriv)
            out["derivative_latex"] = sp.latex(deriv)
            out["integral"]         = str(integ)
            out["integral_latex"]   = sp.latex(integ)
        return out
    except Exception:
        return None


def build_plot_data(expr_str: str, var_str: str) -> dict | None:
    try:
        var  = sp.Symbol(var_str)
        expr = sp.sympify(expr_str)
        f    = sp.lambdify(var, expr, "numpy")
        x    = np.linspace(-6, 6, 500)
        try:
            y = np.array(f(x), dtype=float)
        except Exception:
            y = np.array([float(f(xi)) for xi in x], dtype=float)
        y = np.where(np.isfinite(y), np.clip(y, -30, 30), np.nan)
        return {"x": x.tolist(), "y": y.tolist()}
    except Exception:
        return None


def _push_event(job_id: str, data: dict):
    with _job_lock:
        _job_events.setdefault(job_id, []).append(json.dumps(data))


def _run_job(job_id: str, equation: str, parsed: dict):
    try:
        facts = MathFacts(
            raw_input   =equation,
            parsed_clean=True,
            expression  =parsed["expression"],
            variable    =parsed["variable"],
            derivative  =parsed.get("derivative"),
            integral    =parsed.get("integral"),
        )
        video_path = render_3b1b_style(
            facts, job_id,
            push_event=lambda d: _push_event(job_id, d)
        )
        if video_path:
            update_job_state(job_id, "completed", video_path=video_path)
            set_cache(hashlib.sha256(equation.encode()).hexdigest(), video_path)
            _push_event(job_id, {
                "stage": "done", "progress": 100,
                "message": "✓ Explainer ready!",
                "video_url": f"/api/video/{job_id}"
            })
        else:
            _push_event(job_id, {
                "stage": "error",
                "message": "Render failed — try a simpler equation or check the Errors tab."
            })
    except Exception as e:
        log.error(f"Job {job_id} failed: {e}", exc_info=True)
        _push_event(job_id, {"stage": "error", "message": str(e)})
    finally:
        _push_event(job_id, {"stage": "end"})


# ── LaTeX parser ───────────────────────────────────────────────────
def latex_to_parsed(latex_str: str) -> dict | None:
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex_str)
        free = sorted(expr.free_symbols, key=str)
        var  = free[0] if free else sp.Symbol('x')
        deriv = sp.simplify(sp.diff(expr, var))
        integ = sp.simplify(sp.integrate(expr, var))
        return {
            "expression":       str(expr),
            "latex":            sp.latex(expr),
            "variable":         str(var),
            "derivative":       str(deriv),
            "derivative_latex": sp.latex(deriv),
            "integral":         str(integ),
            "integral_latex":   sp.latex(integ),
        }
    except Exception:
        pass
    s = latex_str
    s = re.sub(r'\\left\s*\(', '(', s);  s = re.sub(r'\\right\s*\)', ')', s)
    s = re.sub(r'\\left\s*\[', '[', s);  s = re.sub(r'\\right\s*\]', ']', s)
    s = re.sub(r'\\cdot', '*', s)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    for fn in ['sin','cos','tan','ln','log','exp','sqrt','asin','acos','atan','sinh','cosh','tanh']:
        s = re.sub(rf'\\{fn}', fn, s)
    s = s.replace('\\pi', 'pi').replace('\\infty', 'oo').replace('\\cdot', '*')
    s = re.sub(r'\^{([^}]+)}', r'**(\1)', s)
    s = re.sub(r'\^([^{])', r'**\1', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace('{', '(').replace('}', ')')
    return fast_parse(s)


# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

# ── Landing ────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")


# ── Auth pages ─────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('app_page'))
    if request.method == "GET":
        return render_template("auth.html", mode="signup")

    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not email or not password or len(password) < 8:
        flash("Please fill in all fields (password ≥ 8 chars).", "error")
        return render_template("auth.html", mode="signup")

    if _get_user_by_email(email):
        flash("An account with that email already exists. Sign in instead.", "error")
        return render_template("auth.html", mode="signin")

    uid  = _create_user(email, name or email.split('@')[0], password=password)
    user = User(uid, email, name)
    login_user(user, remember=True)
    return redirect(url_for('app_page'))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('app_page'))
    if request.method == "GET":
        return render_template("auth.html", mode="signin")

    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    row      = _get_user_by_email(email)

    if not row or not row[3] or not check_password_hash(row[3], password):
        flash("Incorrect email or password.", "error")
        return render_template("auth.html", mode="signin")

    user = User(row[0], row[1], row[2], row[4])
    login_user(user, remember=True)
    return redirect(url_for('app_page'))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))


# ── Google OAuth ───────────────────────────────────────────────────
@app.route("/auth/google")
def auth_google():
    if not GOOGLE_CLIENT_ID:
        flash("Google sign-in is not configured yet. "
              "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable it.", "error")
        return redirect(url_for('login_page'))

    state = secrets.token_urlsafe(16)
    session['oauth_state'] = state

    params = {
        'client_id':     GOOGLE_CLIENT_ID,
        'redirect_uri':  _google_redirect_uri(),
        'response_type': 'code',
        'scope':         'openid email profile',
        'state':         state,
        'access_type':   'online',
        'prompt':        'select_account',
    }
    url = 'https://accounts.google.com/o/oauth2/v2/auth?' + \
          '&'.join(f'{k}={http.utils.quote(str(v))}' for k, v in params.items())
    return redirect(url)


@app.route("/auth/google/callback")
def auth_google_callback():
    if request.args.get('error'):
        flash("Google sign-in was cancelled.", "error")
        return redirect(url_for('login_page'))

    code = request.args.get('code')
    if not code:
        flash("No authorisation code from Google.", "error")
        return redirect(url_for('login_page'))

    # Exchange code for tokens
    try:
        token_r = http.post('https://oauth2.googleapis.com/token', data={
            'client_id':     GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code':          code,
            'grant_type':    'authorization_code',
            'redirect_uri':  _google_redirect_uri(),
        }, timeout=10)
        access_token = token_r.json().get('access_token')
        if not access_token:
            raise ValueError("No access token")

        userinfo = http.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}, timeout=10
        ).json()
    except Exception as e:
        log.error(f"Google OAuth error: {e}")
        flash("Google sign-in failed. Please try again.", "error")
        return redirect(url_for('login_page'))

    email     = userinfo.get('email', '').lower()
    name      = userinfo.get('name', email.split('@')[0])
    google_id = userinfo.get('sub')

    if not email:
        flash("Could not get email from Google.", "error")
        return redirect(url_for('login_page'))

    row = _get_user_by_email(email)
    if row:
        uid = row[0]
        if not row[4]:  # update google_id if missing
            conn = sqlite3.connect(AUTH_DB, timeout=30)
            conn.execute('UPDATE users SET google_id=?,name=? WHERE id=?',
                         (google_id, name, uid))
            conn.commit()
            conn.close()
    else:
        uid = _create_user(email, name, google_id=google_id)

    user = User(uid, email, name, google_id)
    login_user(user, remember=True)
    return redirect(url_for('app_page'))


# ── Main App (protected) ───────────────────────────────────────────
@app.route("/app")
@login_required
def app_page():
    return render_template("index.html", user=current_user)


# ── API — public or lightly guarded ───────────────────────────────
@app.route("/api/equations")
def equations():
    return jsonify(EQUATION_LIBRARY)


@app.route("/api/analyze-latex", methods=["POST"])
def analyze_latex():
    body  = request.get_json(force=True)
    latex = body.get("latex", "").strip()
    if not latex:
        return jsonify({"error": "No LaTeX"}), 400
    parsed = latex_to_parsed(latex)
    if not parsed:
        return jsonify({"error": "Could not parse expression"}), 400
    plot = build_plot_data(parsed["expression"], parsed["variable"])
    return jsonify({**parsed, "plot": plot})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    eq = request.get_json(force=True).get("equation", "").strip()
    if not eq:
        return jsonify({"error": "No equation"}), 400
    parsed = fast_parse(eq)
    if not parsed:
        return jsonify({"error": "Could not parse equation."}), 400
    plot = build_plot_data(parsed["expression"], parsed["variable"])
    return jsonify({**parsed, "plot": plot})


@app.route("/api/solution", methods=["POST"])
def solution():
    body  = request.get_json(force=True)
    eq    = body.get("equation", "").strip()
    latex = body.get("latex", "").strip()

    if latex:
        parsed = latex_to_parsed(latex)
    elif eq:
        parsed = fast_parse(eq)
    else:
        return jsonify({"error": "No equation"}), 400

    if not parsed:
        return jsonify({"error": "Cannot parse"}), 400

    var = parsed["variable"]
    prompt = (
        f"You are MathGPT, an expert mathematics AI.\n"
        f"Analyse: f({var}) = {parsed['expression']}\n"
        f"Known: derivative f'({var}) = {parsed.get('derivative','')}, "
        f"integral = {parsed.get('integral','')}\n\n"
        "Return ONLY valid JSON (no markdown fences) with EXACTLY this structure:\n"
        '{\n'
        '  "function_type": "short type name",\n'
        '  "latex_display": "full LaTeX for f(x)=...",\n'
        '  "steps": [\n'
        '    {"heading": "Step title", "latex": "LaTeX expression", "explanation": "one-line plain text"}\n'
        '  ],\n'
        '  "domain": "LaTeX domain",\n'
        '  "range": "LaTeX range or description",\n'
        '  "key_insight": "one interesting mathematical fact"\n'
        '}'
    )
    try:
        raw  = call_llm(prompt, "You are MathGPT. Return ONLY valid JSON.", temperature=0.15)
        m    = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m: raise ValueError("No JSON found")
        data = json.loads(m.group(0))
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/render", methods=["POST"])
@login_required
def render():
    body  = request.get_json(force=True)
    eq    = body.get("equation", "").strip()
    latex = body.get("latex", "").strip()

    if latex:
        parsed = latex_to_parsed(latex)
        eq = eq or latex
    elif eq:
        parsed = fast_parse(eq)
    else:
        return jsonify({"error": "No equation"}), 400

    if not parsed:
        return jsonify({"error": "Cannot parse equation"}), 400

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    init_math_db()
    update_job_state(job_id, "queued")
    threading.Thread(target=_run_job, args=(job_id, eq, parsed), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    def gen():
        seen = 0
        import time
        for _ in range(600):
            with _job_lock:
                events = _job_events.get(job_id, [])
            while seen < len(events):
                yield f"data: {events[seen]}\n\n"
                evt = json.loads(events[seen])
                seen += 1
                if evt.get("stage") in ("done", "error", "end"):
                    return
            time.sleep(0.4)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/video/<job_id>")
def video(job_id: str):
    path = os.path.abspath(f"./output_{job_id}/render.mp4")
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="video/mp4")


# ── Contact form (landing page) ────────────────────────────────────
@app.route("/api/contact", methods=["POST"])
def contact():
    data = request.get_json(force=True)
    log.info(f"Contact form: {data.get('name')} <{data.get('email')}> — {data.get('reason')}")
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_user_db()
    init_math_db()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
