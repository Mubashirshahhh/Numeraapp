# Numera — Landing Page (GitHub Pages)

This is the **static marketing/landing page** for Numera, set up to deploy on GitHub Pages.

## ⚠️ Important: this is the landing page only

GitHub Pages can only serve static files (HTML/CSS/JS). It **cannot** run:
- The Flask backend (`app.py` / `main.py`)
- SQLite user accounts / login
- Manim + LaTeX + ffmpeg rendering

So the actual "type a prompt, get an animation" app has to be deployed
somewhere that runs Python — e.g. **Render**, **Railway**, **Fly.io**, or a VPS.
This landing page just needs to know the URL of that deployment.

## One-time setup after deploying the backend

1. Deploy `workspace/app.py` + `workspace/main.py` (and requirements.txt) to
   Render/Railway/Fly.io. Make sure CORS stays enabled (it already is, via
   `flask_cors`).
2. Open `index.html` in this repo, find this line near the top:
   ```html
   <script>const NUMERA_APP_URL = "";</script>
   ```
   and put your backend's URL inside the quotes, e.g.:
   ```html
   <script>const NUMERA_APP_URL = "https://numera-backend.onrender.com";</script>
   ```
3. Commit and push — GitHub Pages will pick up the change automatically.

Until you set that URL, the "Try the App" / "Get Started" buttons will just
append `/app` or `/signup` to the empty string (a broken link) — so do this
before sharing the link widely.

## Deploying to GitHub Pages

```bash
git init
git add .
git commit -m "Numera landing page"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Then in GitHub: **Settings → Pages → Source → Deploy from a branch → main → / (root)**.

Your site will be live at `https://<your-username>.github.io/<your-repo>/`.

## `backend-reference/`

Copies of `app.py`, `main.py`, `app.html`, and `requirements.txt` from your
Numera workspace, kept here just for reference. These are what you deploy to
your Python host (Render/Railway/Fly.io) — they don't get served by GitHub
Pages. `app.html` still has Flask/Jinja template tags in it (`{% if user %}`),
so it must stay served by Flask, not GitHub Pages.
