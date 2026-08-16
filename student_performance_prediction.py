"""
================================================
  Student Performance Predictor
  Python Desktop App (Tkinter + scikit-learn)
================================================
  Run: python student_performance_app.py
  No extra install needed — uses built-in Tkinter
  Requires: pip install scikit-learn pandas numpy
================================================
"""
 
import tkinter as tk
from tkinter import ttk, font
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import threading
 
# ─────────────────────────────────────────────
#  1. TRAIN MODEL ON SYNTHETIC DATASET
# ─────────────────────────────────────────────
 
def generate_dataset(n=1500):
    np.random.seed(42)
    study_hours      = np.random.normal(15, 7, n).clip(1, 40)
    attendance       = np.random.normal(78, 15, n).clip(20, 100)
    previous_gpa     = np.random.normal(6.5, 1.8, n).clip(0, 10)
    assignments_done = np.random.normal(80, 15, n).clip(10, 100)
    sleep_hours      = np.random.normal(7, 1.2, n).clip(4, 10)
    extracurriculars = np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.40, 0.30, 0.15])
    parental_support = np.random.choice([0, 1, 2],    n, p=[0.20, 0.50, 0.30])
    learning_style   = np.random.choice([0, 1, 2, 3], n, p=[0.30, 0.20, 0.30, 0.20])
 
    extra_w  = np.array([0, 0.10, -0.05, -0.20])[extracurriculars]
    support_w= np.array([-0.40, 0.0, 0.50])[parental_support]
    style_w  = np.array([0.10, 0.05, 0.15, 0.0])[learning_style]
 
    gpa = (
        0.25 * (study_hours / 40 * 10) +
        0.20 * (attendance / 100 * 10) +
        0.30 * previous_gpa +
        0.15 * (assignments_done / 100 * 10) +
        0.05 * (sleep_hours / 10 * 10) +
        extra_w + support_w + style_w +
        np.random.normal(0, 0.5, n)
    ).clip(0, 10)
 
    return pd.DataFrame({
        "study_hours":      study_hours,
        "attendance":       attendance,
        "previous_gpa":     previous_gpa,
        "assignments":      assignments_done,
        "sleep":            sleep_hours,
        "extracurriculars": extracurriculars,
        "parental_support": parental_support,
        "learning_style":   learning_style,
        "gpa":              gpa
    })
 
FEATURES = ["study_hours","attendance","previous_gpa","assignments","sleep",
            "extracurriculars","parental_support","learning_style"]
 
df     = generate_dataset(1500)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(df[FEATURES])
model  = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                    max_depth=5, random_state=42)
model.fit(X_sc, df["gpa"])
 
 
def predict_gpa(inputs: dict) -> float:
    row = pd.DataFrame([inputs])[FEATURES]
    sc  = scaler.transform(row)
    return float(np.clip(model.predict(sc)[0], 0, 10))
 
 
def gpa_to_grade(g):
    if g >= 9.0:  return "A", "#1a7a4a"
    if g >= 7.5:  return "B", "#185FA5"
    if g >= 6.0:  return "C", "#b8860b"
    if g >= 4.5:  return "D", "#8b4513"
    return            "F", "#8b1a1a"
 
 
# ─────────────────────────────────────────────
#  2. GUI APPLICATION
# ─────────────────────────────────────────────
 
COLORS = {
    "bg":       "#f5f3ee",
    "surface":  "#ffffff",
    "surface2": "#f0ede6",
    "border":   "#e2ddd4",
    "accent":   "#2d5a3d",
    "accent2":  "#4a8c60",
    "text":     "#1a1815",
    "text2":    "#6b6560",
    "text3":    "#9a948d",
    "green_bg": "#e8f2eb",
    "blue_bg":  "#e8eef8",
    "gold_bg":  "#fdf6e3",
    "warn_bg":  "#fdf0e6",
    "red_bg":   "#fde8e8",
}
 
 
class StudentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Predictor")
        self.geometry("860x700")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
 
        self._build_fonts()
        self._build_header()
        self._build_main()
 
    # ── fonts
    def _build_fonts(self):
        self.f_title  = font.Font(family="Georgia", size=20, weight="bold")
        self.f_sub    = font.Font(family="Georgia", size=12)
        self.f_label  = font.Font(family="Helvetica", size=9,  weight="bold")
        self.f_body   = font.Font(family="Helvetica", size=11)
        self.f_body_b = font.Font(family="Helvetica", size=11, weight="bold")
        self.f_small  = font.Font(family="Helvetica", size=9)
        self.f_grade  = font.Font(family="Georgia",   size=42, weight="bold")
        self.f_gpa    = font.Font(family="Georgia",   size=28, weight="bold")
        self.f_metric = font.Font(family="Helvetica", size=13, weight="bold")
        self.f_btn    = font.Font(family="Helvetica", size=12, weight="bold")
        self.f_section= font.Font(family="Helvetica", size=10, weight="bold")
 
    # ── header bar
    def _build_header(self):
        bar = tk.Frame(self, bg=COLORS["surface"], height=54)
        bar.pack(fill="x")
        bar.pack_propagate(False)
 
        tk.Label(bar, text="PredictIQ", font=self.f_body_b,
                 bg=COLORS["surface"], fg=COLORS["accent"]).pack(side="left", padx=18, pady=14)
        tk.Label(bar, text="ML-Powered · Gradient Boosting Regressor",
                 font=self.f_small, bg=COLORS["surface"], fg=COLORS["text3"]).pack(side="right", padx=18)
 
        sep = tk.Frame(self, bg=COLORS["border"], height=1)
        sep.pack(fill="x")
 
    # ── scrollable main area
    def _build_main(self):
        container = tk.Frame(self, bg=COLORS["bg"])
        container.pack(fill="both", expand=True)
 
        canvas = tk.Canvas(container, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=COLORS["bg"])
 
        self.scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
 
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
 
        self._build_hero(self.scroll_frame)
        self._build_form(self.scroll_frame)
        self._build_button(self.scroll_frame)
 
        self.report_frame = tk.Frame(self.scroll_frame, bg=COLORS["bg"])
        self.report_frame.pack(fill="x", padx=28, pady=(0, 30))
 
    # ── hero
    def _build_hero(self, parent):
        f = tk.Frame(parent, bg=COLORS["bg"])
        f.pack(fill="x", padx=28, pady=(24, 12))
        tk.Label(f, text="Student Performance Predictor",
                 font=self.f_title, bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        tk.Label(f, text="Enter your academic details below and click Predict Performance to get your personalised report.",
                 font=self.f_sub, bg=COLORS["bg"], fg=COLORS["text2"],
                 wraplength=780, justify="left").pack(anchor="w", pady=(4, 0))
 
    # ── form card
    def _build_form(self, parent):
        card = tk.Frame(parent, bg=COLORS["surface"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack(fill="x", padx=28, pady=(0, 12))
 
        # card header
        ch = tk.Frame(card, bg=COLORS["surface2"], height=40)
        ch.pack(fill="x")
        ch.pack_propagate(False)
        tk.Label(ch, text="  Student Details", font=self.f_section,
                 bg=COLORS["surface2"], fg=COLORS["text2"]).pack(side="left", padx=10, pady=8)
 
        sep = tk.Frame(card, bg=COLORS["border"], height=1)
        sep.pack(fill="x")
 
        body = tk.Frame(card, bg=COLORS["surface"], padx=20, pady=20)
        body.pack(fill="x")
 
        # two-column grid
        left  = tk.Frame(body, bg=COLORS["surface"])
        right = tk.Frame(body, bg=COLORS["surface"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 20))
        right.pack(side="left", fill="both", expand=True)
 
        self.vars = {}
 
        # helper to add a slider field
        def add_slider(parent, key, label, from_, to, default, fmt_fn, resolution=1):
            f = tk.Frame(parent, bg=COLORS["surface"])
            f.pack(fill="x", pady=10)
            header = tk.Frame(f, bg=COLORS["surface"])
            header.pack(fill="x")
            tk.Label(header, text=label.upper(), font=self.f_label,
                     bg=COLORS["surface"], fg=COLORS["text3"]).pack(side="left")
            val_lbl = tk.Label(header, text=fmt_fn(default), font=self.f_body_b,
                               bg=COLORS["surface"], fg=COLORS["accent"])
            val_lbl.pack(side="right")
 
            var = tk.DoubleVar(value=default)
            self.vars[key] = var
 
            def on_change(v):
                val_lbl.config(text=fmt_fn(float(v)))
 
            s = tk.Scale(f, variable=var, from_=from_, to=to,
                         orient="horizontal", resolution=resolution,
                         showvalue=False, command=on_change,
                         bg=COLORS["surface"], fg=COLORS["accent"],
                         highlightthickness=0, troughcolor=COLORS["surface2"],
                         activebackground=COLORS["accent2"], sliderrelief="flat",
                         sliderlength=20, bd=0)
            s.pack(fill="x", pady=(4, 0))
 
        # helper to add a dropdown field
        def add_dropdown(parent, key, label, options):
            f = tk.Frame(parent, bg=COLORS["surface"])
            f.pack(fill="x", pady=10)
            tk.Label(f, text=label.upper(), font=self.f_label,
                     bg=COLORS["surface"], fg=COLORS["text3"]).pack(anchor="w")
            var = tk.StringVar(value=options[1])
            self.vars[key] = var
            combo = ttk.Combobox(f, textvariable=var, values=options,
                                 state="readonly", font=self.f_body)
            combo.pack(fill="x", pady=(6, 0))
 
        # LEFT column
        add_slider(left, "study_hours", "Study Hours / Week",
                   1, 40, 15, lambda v: f"{int(v)} hrs")
        add_slider(left, "previous_gpa", "Previous GPA (0–10)",
                   0, 10, 7.0, lambda v: f"{float(v):.1f}", resolution=0.1)
        add_slider(left, "sleep", "Sleep (Hours / Night)",
                   4, 10, 7.0, lambda v: f"{float(v):.1f} hrs", resolution=0.5)
        add_dropdown(left, "parental_support", "Parental Support",
                     ["Low", "Medium", "High"])
 
        # RIGHT column
        add_slider(right, "attendance", "Attendance (%)",
                   20, 100, 80, lambda v: f"{int(v)}%")
        add_slider(right, "assignments", "Assignments Completed (%)",
                   10, 100, 85, lambda v: f"{int(v)}%")
        add_dropdown(right, "extracurriculars", "Extracurriculars",
                     ["None", "Light (1-2 activities)",
                      "Moderate (3-4 activities)", "Heavy (5+ activities)"])
        add_dropdown(right, "learning_style", "Learning Style",
                     ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"])
 
    # ── predict button
    def _build_button(self, parent):
        f = tk.Frame(parent, bg=COLORS["bg"])
        f.pack(fill="x", padx=28, pady=(0, 16))
 
        self.btn = tk.Button(
            f, text="Predict Performance  ->",
            font=self.f_btn, bg=COLORS["accent"], fg="white",
            activebackground="#1e4029", activeforeground="white",
            relief="flat", cursor="hand2", padx=20, pady=14,
            command=self._on_predict
        )
        self.btn.pack(fill="x")
 
        self.status_lbl = tk.Label(f, text="", font=self.f_small,
                                   bg=COLORS["bg"], fg=COLORS["text3"])
        self.status_lbl.pack(pady=(6, 0))
 
        # progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor=COLORS["surface2"],
                        background=COLORS["accent2"], thickness=5)
        self.progress = ttk.Progressbar(f, style="Green.Horizontal.TProgressbar",
                                        mode="determinate", maximum=100)
 
    # ── predict handler
    def _on_predict(self):
        self.btn.config(state="disabled", text="Analysing...")
        self.progress.pack(fill="x", pady=(6, 0))
        self.progress["value"] = 0
        self._animate_progress(0)
 
    def _animate_progress(self, val):
        messages = ["Collecting inputs...", "Scaling features...",
                    "Running model...", "Building report..."]
        if val <= 100:
            self.progress["value"] = val
            idx = min(int(val / 25), 3)
            self.status_lbl.config(text=messages[idx])
            self.after(60, self._animate_progress, val + 5)
        else:
            self._run_prediction()
 
    def _run_prediction(self):
        # read inputs
        extra_map   = {"None": 0, "Light (1-2 activities)": 1,
                       "Moderate (3-4 activities)": 2, "Heavy (5+ activities)": 3}
        support_map = {"Low": 0, "Medium": 1, "High": 2}
        style_map   = {"Visual": 0, "Auditory": 1, "Reading/Writing": 2, "Kinesthetic": 3}
 
        inputs = {
            "study_hours":      self.vars["study_hours"].get(),
            "attendance":       self.vars["attendance"].get(),
            "previous_gpa":     self.vars["previous_gpa"].get(),
            "assignments":      self.vars["assignments"].get(),
            "sleep":            self.vars["sleep"].get(),
            "extracurriculars": extra_map[self.vars["extracurriculars"].get()],
            "parental_support": support_map[self.vars["parental_support"].get()],
            "learning_style":   style_map[self.vars["learning_style"].get()],
        }
 
        gpa = predict_gpa(inputs)
        grade, grade_color = gpa_to_grade(gpa)
 
        self.progress.pack_forget()
        self.status_lbl.config(text="")
        self.btn.config(state="normal", text="Predict Again  ->")
 
        self._show_report(inputs, gpa, grade, grade_color)
 
    # ── build report
    def _show_report(self, inputs, gpa, grade, grade_color):
        for w in self.report_frame.winfo_children():
            w.destroy()
 
        # ── score banner
        banner = tk.Frame(self.report_frame, bg=COLORS["surface"],
                          highlightbackground=COLORS["border"], highlightthickness=1)
        banner.pack(fill="x", pady=(0, 12))
 
        top = tk.Frame(banner, bg=COLORS["surface"], pady=16, padx=20)
        top.pack(fill="x")
 
        # grade circle (label styled as circle)
        grade_bg = {
            "A": COLORS["green_bg"], "B": COLORS["blue_bg"],
            "C": COLORS["gold_bg"],  "D": COLORS["warn_bg"], "F": COLORS["red_bg"]
        }[grade]
 
        circ = tk.Label(top, text=grade, font=self.f_grade,
                        bg=grade_bg, fg=grade_color,
                        width=2, relief="flat", padx=10, pady=6)
        circ.pack(side="left", padx=(0, 20))
 
        info = tk.Frame(top, bg=COLORS["surface"])
        info.pack(side="left", fill="both", expand=True)
 
        grade_text = {
            "A": "Outstanding — you are in the top tier!",
            "B": "Good performance — above average standing.",
            "C": "Average — room for improvement.",
            "D": "Below average — improvement needed.",
            "F": "At risk — urgent attention required."
        }[grade]
 
        tk.Label(info, text=f"Predicted GPA: {gpa:.2f} / 10.0",
                 font=self.f_gpa, bg=COLORS["surface"], fg=COLORS["text"]).pack(anchor="w")
        tk.Label(info, text=grade_text,
                 font=self.f_body, bg=COLORS["surface"], fg=COLORS["text2"]).pack(anchor="w", pady=(4,0))
 
        # metrics strip
        sep = tk.Frame(banner, bg=COLORS["border"], height=1)
        sep.pack(fill="x")
 
        metrics = tk.Frame(banner, bg=COLORS["surface"])
        metrics.pack(fill="x")
 
        pct = min(99, max(1, int((gpa / 10) * 85 + 5)))
        for i, (label, value) in enumerate([
            ("Predicted GPA", f"{gpa:.2f}"),
            ("Letter Grade",  grade),
            ("Percentile",    f"{pct}th"),
        ]):
            m = tk.Frame(metrics, bg=COLORS["surface"], padx=20, pady=12)
            m.pack(side="left", fill="both", expand=True)
            if i < 2:
                tk.Frame(metrics, bg=COLORS["border"], width=1).pack(side="left", fill="y")
            tk.Label(m, text=label.upper(), font=self.f_label,
                     bg=COLORS["surface"], fg=COLORS["text3"]).pack(anchor="center")
            tk.Label(m, text=value, font=self.f_metric,
                     bg=COLORS["surface"], fg=grade_color).pack(anchor="center", pady=(4,0))
 
        # ── factor analysis
        self._section_card(
            title="Factor Analysis",
            build_fn=lambda f: self._build_factors(f, inputs)
        )
 
        # ── recommendations
        self._section_card(
            title="Personalized Recommendations",
            build_fn=lambda f: self._build_recos(f, inputs, grade)
        )
 
        # ── model info
        info_bar = tk.Frame(self.report_frame, bg=COLORS["surface2"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
        info_bar.pack(fill="x")
        tk.Label(info_bar,
                 text="Model: Gradient Boosting Regressor trained on 1,500 synthetic student records. "
                      "Features: study hours, attendance, GPA, assignments, sleep, extracurriculars, "
                      "parental support, learning style.",
                 font=self.f_small, bg=COLORS["surface2"], fg=COLORS["text3"],
                 wraplength=780, justify="left", padx=12, pady=8).pack(anchor="w")
 
    def _section_card(self, title, build_fn):
        card = tk.Frame(self.report_frame, bg=COLORS["surface"],
                        highlightbackground=COLORS["border"], highlightthickness=1)
        card.pack(fill="x", pady=(0, 12))
 
        ch = tk.Frame(card, bg=COLORS["surface2"], height=38)
        ch.pack(fill="x")
        ch.pack_propagate(False)
        tk.Label(ch, text=f"  {title}", font=self.f_section,
                 bg=COLORS["surface2"], fg=COLORS["text2"]).pack(side="left", padx=8, pady=8)
 
        tk.Frame(card, bg=COLORS["border"], height=1).pack(fill="x")
 
        body = tk.Frame(card, bg=COLORS["surface"], padx=16, pady=14)
        body.pack(fill="x")
        build_fn(body)
 
    def _build_factors(self, parent, inputs):
        factors = [
            ("Study Hours",      inputs["study_hours"] / 40 * 100,                   "#2d5a3d"),
            ("Attendance",       (inputs["attendance"] - 20) / 80 * 100,             "#185FA5"),
            ("Previous GPA",     inputs["previous_gpa"] / 10 * 100,                  "#534AB7"),
            ("Assignments",      (inputs["assignments"] - 10) / 90 * 100,            "#BA7517"),
            ("Sleep Quality",    (inputs["sleep"] - 4) / 6 * 100,                   "#8b1a6b"),
            ("Parental Support", [25, 55, 90][inputs["parental_support"]],            "#1a3a6b"),
            ("Extracurriculars", [55, 75, 58, 30][inputs["extracurriculars"]],        "#6b3a1a"),
        ]
 
        for name, score, color in factors:
            score = int(min(100, max(0, score)))
            row = tk.Frame(parent, bg=COLORS["surface"])
            row.pack(fill="x", pady=5)
 
            tk.Label(row, text=name, font=self.f_body,
                     bg=COLORS["surface"], fg=COLORS["text2"],
                     width=18, anchor="w").pack(side="left")
 
            # canvas bar
            bar_canvas = tk.Canvas(row, height=10, bg=COLORS["surface2"],
                                   highlightthickness=0, bd=0)
            bar_canvas.pack(side="left", fill="x", expand=True, padx=8)
            bar_canvas.update_idletasks()
            w = bar_canvas.winfo_width() or 300
            fill_w = int(w * score / 100)
            bar_canvas.create_rectangle(0, 2, fill_w, 8, fill=color, outline="")
 
            tag = "Good" if score >= 70 else "Fair" if score >= 45 else "Low"
            tag_colors = {
                "Good": (COLORS["green_bg"], "#1a7a4a"),
                "Fair": (COLORS["gold_bg"],  "#b8860b"),
                "Low":  (COLORS["red_bg"],   "#8b1a1a"),
            }
            tbg, tfg = tag_colors[tag]
            tk.Label(row, text=f"{score}%", font=self.f_small,
                     bg=COLORS["surface"], fg=COLORS["text2"], width=4).pack(side="left")
            tk.Label(row, text=tag, font=self.f_small,
                     bg=tbg, fg=tfg, padx=6, pady=2).pack(side="left", padx=4)
 
    def _build_recos(self, parent, inputs, grade):
        sh   = inputs["study_hours"]
        att  = inputs["attendance"]
        asgn = inputs["assignments"]
        slp  = inputs["sleep"]
        ex   = inputs["extracurriculars"]
        sup  = inputs["parental_support"]
 
        recos = []
        if sh < 10:
            recos.append(("danger",  "Low study hours detected. Aim for at least 15-20 hrs/week to improve GPA significantly."))
        elif sh < 18:
            recos.append(("warning", "Increasing study hours to 18-22 hrs/week could raise your GPA by 0.5-1 point."))
        else:
            recos.append(("good",    "Great study habit! Use active recall and spaced repetition to maximize retention."))
 
        if att < 70:
            recos.append(("danger",  "Attendance below 70% is critical. Missing classes creates knowledge gaps and hurts exam scores."))
        elif att < 85:
            recos.append(("warning", "Bring attendance above 85%. A 10% improvement correlates strongly with better grades."))
        else:
            recos.append(("good",    "Excellent attendance! You are maximizing classroom learning opportunities."))
 
        if asgn < 70:
            recos.append(("danger",  "Less than 70% assignments completed is dragging your GPA. Prioritize finishing all work."))
        elif asgn < 90:
            recos.append(("warning", "Push assignment completion above 90%. Completing all work reinforces learning and boosts grades."))
        else:
            recos.append(("good",    "Great job completing assignments! This habit strongly supports your academic success."))
 
        if slp < 6:
            recos.append(("danger",  "Under 6 hrs of sleep hurts memory and focus. Aim for 7-8 hrs for optimal performance."))
        elif slp < 7:
            recos.append(("warning", "Try to get 7+ hours of sleep. Better rest improves retention and reduces exam fatigue."))
 
        if ex == 3:
            recos.append(("warning", "Heavy extracurricular load may be limiting study time. Consider reducing to 2-3 activities."))
 
        if sup == 0:
            recos.append(("warning", "Low parental support? Seek peer study groups, tutoring, or mentorship at your institution."))
 
        if grade in ("A", "B"):
            recos.append(("good",    "You're on track! Challenge yourself with advanced coursework or research opportunities."))
 
        colors = {
            "good":    (COLORS["green_bg"], "#1a7a4a", "OK"),
            "warning": (COLORS["gold_bg"],  "#b8860b", "!"),
            "danger":  (COLORS["red_bg"],   "#8b1a1a", "!!"),
        }
 
        for rtype, text in recos:
            bg, fg, icon = colors[rtype]
            row = tk.Frame(parent, bg=bg,
                           highlightbackground=fg, highlightthickness=1)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f" {icon} ", font=self.f_body_b,
                     bg=bg, fg=fg, padx=4).pack(side="left")
            tk.Label(row, text=text, font=self.f_body,
                     bg=bg, fg=COLORS["text"],
                     wraplength=660, justify="left", pady=8, padx=4).pack(side="left", fill="x")
 
 
# ─────────────────────────────────────────────
#  3. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = StudentApp()
    app.mainloop()
 