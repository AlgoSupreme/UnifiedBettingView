import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
import numpy as np
from datetime import datetime, timedelta
import time
import pandas as pd
import seaborn as sns
import nhl_team_list as tcl
from nhlpy import NHLClient
import sys
import pywinstyles

# Matplotlib integration
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import scipy.stats as stats

# AC_COLOR -> Accent Color
AC_COLOR="#000000"
AC_MINOR="#E5E5E5"
BG_COLOR="#14213D"
FG_COLOR="#FCA311"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": FG_COLOR,
    "axes.labelcolor": FG_COLOR,
    "xtick.color": FG_COLOR,
    "ytick.color": FG_COLOR,
    "text.color": FG_COLOR,
    "legend.facecolor": BG_COLOR,
    "legend.edgecolor": FG_COLOR,
})

# =============================================================================
# VIEW 1: GOALIE PERFORMANCE VIEWER
# =============================================================================
class GoalieView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.SaveDir = "dump"
        self.controller = controller
        
        # --- Data Loading Logic ---
        # We handle data loading here to prevent crashing the whole app if file is missing
        self.goalie_data = {}
        self.sorted_names = []
        self.load_data()

        # --- GUI Setup ---
        self.setup_ui()

    def load_data(self):
        # Dynamic filename based on date
        fname = f"goalie_analysis_Uni{datetime.now().date()}.json"
        file_path = os.path.join(self.SaveDir, fname)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    self.goalie_data = json.load(f)
            except Exception as e:
                print(f"Error reading goalie data: {e}")
        else:
            messagebox.showerror("ERROR", "FILE MISSING PLEASE DUMP DATA BEFORE USING")

        self.name_to_id = {}
        for gid, data in self.goalie_data.items():
            name = data.get("Name", "Unknown")
            self.name_to_id[name] = gid
            self.sorted_names.append(name)
        self.sorted_names.sort()

    def setup_ui(self):
        # Top Frame for Controls
        top_frame = tk.Frame(self, pady=10, bg=AC_COLOR)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH)

        tk.Label(top_frame, text="GOALIE VIEW", bg=AC_MINOR, fg="#555", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)

        # 1. Goalie Selection
        tk.Label(top_frame, text="Select Goalie:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.combo_goalie = ttk.Combobox(top_frame, values=self.sorted_names, state="readonly", width=20)
        self.combo_goalie.pack(side=tk.LEFT, padx=5)
        self.combo_goalie.bind("<<ComboboxSelected>>", self.update_plot)

        # 2. Chart Style Selection
        tk.Label(top_frame, text="Style:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.chart_styles = ["Trend (Combined)", "Trend (Separate)", "Gaussian Distribution"]
        self.combo_style = ttk.Combobox(top_frame, values=self.chart_styles, state="readonly", width=18)
        self.combo_style.current(0)
        self.combo_style.pack(side=tk.LEFT, padx=5)
        self.combo_style.bind("<<ComboboxSelected>>", self.update_plot)

        # 3. Reference Value Input
        tk.Label(top_frame, text="Ref Value:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.entry_ref = tk.Entry(top_frame, width=8)
        self.entry_ref.pack(side=tk.LEFT, padx=5)
        self.entry_ref.bind("<Return>", self.update_plot)

        btn_apply = tk.Button(top_frame, text="Apply", command=lambda: self.update_plot(None), bg="#e1e1e1")
        btn_apply.pack(side=tk.LEFT, padx=5)

        # Graph Area
        self.figure = Figure(figsize=(8, 8), dpi=100, facecolor=BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=BG_COLOR, highlightthickness=0)
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        if self.sorted_names:
            self.combo_goalie.set(self.sorted_names[0])
            self.update_plot(None)

    def update_plot(self, event):
        selected_name = self.combo_goalie.get()
        selected_style = self.combo_style.get()
        
        # Define your theme colors
        BG_COLOR = "#14213D"
        FG_COLOR = "#FCA311"

        ref_val = None
        raw_ref = self.entry_ref.get().strip()
        if raw_ref:
            try:
                ref_val = float(raw_ref)
            except ValueError:
                pass

        if not selected_name:
            return

        gid = self.name_to_id[selected_name]
        data = self.goalie_data[gid]

        shots = data.get("shotsAgainst", [])
        saves = data.get("saves", [])
        goals = data.get("goalsAgainst", [])
        games = list(range(1, len(shots) + 1))

        # 1. Clear and set the Figure background
        self.figure.clear()
        self.figure.set_facecolor(BG_COLOR)

        # 2. Call the drawing styles
        if selected_style == "Trend (Combined)":
            self.draw_trend_combined(games, shots, saves, goals, selected_name)
        elif selected_style == "Trend (Separate)":
            self.draw_trend_separate(games, shots, saves, goals, selected_name)
        elif selected_style == "Gaussian Distribution":
            self.draw_gaussian(shots, saves, goals, selected_name, ref_val)

        # 3. Apply the theme to all axes created by the draw functions
        for ax in self.figure.get_axes():
            ax.set_facecolor(BG_COLOR)
            ax.tick_params(colors=FG_COLOR, which='both')
            ax.xaxis.label.set_color(FG_COLOR)
            ax.yaxis.label.set_color(FG_COLOR)
            ax.title.set_color(FG_COLOR)
            
            # Color the spines (borders)
            for spine in ax.spines.values():
                spine.set_edgecolor(FG_COLOR)
                
            # If there is a legend, color it too
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(BG_COLOR)
                legend.get_frame().set_edgecolor(FG_COLOR)
                for text in legend.get_texts():
                    text.set_color(FG_COLOR)

        self.canvas.draw()

    def draw_trend_combined(self, games, shots, saves, goals, name):
        ax = self.figure.add_subplot(111)
        ax.plot(games, shots, marker='o', linestyle='-', label='Shots', color='blue')
        ax.plot(games, saves, marker='x', linestyle='--', label='Saves', color='green')
        ax.plot(games, goals, marker='s', linestyle='-', label='Goals', color='red')
        ax.set_title(f"Performance Stats (Combined): {name}")
        ax.set_xlabel("Game Number")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True)

    def draw_trend_separate(self, games, shots, saves, goals, name):
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312)
        ax3 = self.figure.add_subplot(313)

        ax1.plot(games, shots, marker='o', color='blue')
        ax1.set_title(f"Shots Trend: {name}")
        ax1.grid(True)
        ax2.plot(games, saves, marker='x', color='green')
        ax2.set_title("Saves Trend")
        ax2.grid(True)
        ax3.plot(games, goals, marker='s', color='red')
        ax3.set_title("Goals Trend")
        ax3.set_xlabel("Game Number")
        ax3.grid(True)
        self.figure.tight_layout()

    def draw_gaussian(self, shots, saves, goals, name, ref_val):
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312)
        ax3 = self.figure.add_subplot(313)
        self.plot_single_gaussian(ax1, shots, "Shots Against", "blue", ref_val)
        self.plot_single_gaussian(ax2, saves, "Saves", "green", ref_val)
        self.plot_single_gaussian(ax3, goals, "Goals Against", "red", ref_val)
        ax3.set_xlabel("Count per Game")
        self.figure.tight_layout()

    def plot_single_gaussian(self, ax, data, metric_name, color, ref_val):
        if not data or len(data) < 2:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
            ax.set_title(metric_name)
            return
        x_data = np.array(data)
        mu = np.mean(x_data)
        sigma = np.std(x_data)
        ax.hist(x_data, bins='auto', density=True, alpha=0.4, color=color, edgecolor=FG_COLOR, label='Freq')
        xmin, xmax = ax.get_xlim()
        if ref_val is not None:
            xmin = min(xmin, ref_val - 2)
            xmax = max(xmax, ref_val + 2)
        x = np.linspace(xmin, xmax, 100)
        p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        ax.plot(x, p, linewidth=2, label=f'$\\mu={mu:.1f}$', color=FG_COLOR)
        line_height = max(p) * 1.1
        ax.vlines(mu, 0, line_height, color=FG_COLOR, linestyle='dashed', alpha=0.5)
        for s in [-2, -1, 1, 2]:
            ax.vlines(mu + (s * sigma), 0, line_height, color='green', linestyle=':', alpha=0.5)
        if ref_val is not None:
            ax.vlines(ref_val, 0, line_height, color='magenta', linestyle='-', linewidth=2, label=f'Ref: {ref_val}')
            pct_below = np.mean(x_data < ref_val) * 100
            pct_above = np.mean(x_data > ref_val) * 100
            ax.text(ref_val, line_height * 0.9, f" Below {pct_below:.1f}% ", ha='right', color='magenta', fontweight='bold', fontsize=9)
            ax.text(ref_val, line_height * 0.9, f" Above {pct_above:.1f}% ", ha='left', color='magenta', fontweight='bold', fontsize=9)
        ax.set_title(f"{metric_name}")
        ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, line_height * 1.15)


# =============================================================================
# VIEW 2: PLAYER PERFORMANCE ANALYZER
# =============================================================================
class PlayerView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.SaveDir = "dump"
        self.controller = controller
        
        self.player_data = {}
        self.sorted_names = []
        self.load_data()
        self.setup_ui()

    def load_data(self):
        fname = f"player_analysis_Uni{datetime.now().date()}.json"
        file_path = os.path.join(self.SaveDir, fname)

        raw_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    raw_data = json.load(f)
            except Exception as e:
                print(f"Error loading player data: {e}")
        else:
            messagebox.showerror("ERROR", "FILE MISSING PLEASE DUMP DATA BEFORE USING")

        for pid, data in raw_data.items():
            name = data.get("Name", "Unknown")
            goals_arr = data.get("goals", [])
            assists_arr = data.get("assists", [])
            shots_arr = data.get("shots", [])
            points_arr = []
            length = min(len(goals_arr), len(assists_arr)) 
            for i in range(length):
                points_arr.append(goals_arr[i] + assists_arr[i])
                
            self.player_data[name] = {
                "Goals": np.array(goals_arr),
                "Assists": np.array(assists_arr),
                "Points": np.array(points_arr),
                "Shots": np.array(shots_arr)
            }
            self.sorted_names.append(name)
        self.sorted_names.sort()

    def setup_ui(self):
        top_frame = tk.Frame(self, pady=10, bg=AC_COLOR)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(top_frame, text="PLAYER ANALYSIS", bg=AC_MINOR, fg="#555", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)

        # 1. Player Selection
        tk.Label(top_frame, text="Player:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(10, 5))
        self.combo_player = ttk.Combobox(top_frame, values=self.sorted_names, state="readonly", width=20)
        self.combo_player.pack(side=tk.LEFT, padx=5)
        self.combo_player.bind("<<ComboboxSelected>>", self.update_plot)
        if self.sorted_names:
            self.combo_player.current(0)

        # 2. Metric Selection
        tk.Label(top_frame, text="Metric:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.metrics = ["Points", "Goals", "Assists", "Shots"]
        self.combo_metric = ttk.Combobox(top_frame, values=self.metrics, state="readonly", width=10)
        self.combo_metric.pack(side=tk.LEFT, padx=5)
        self.combo_metric.current(0) 
        self.combo_metric.bind("<<ComboboxSelected>>", self.update_plot)

        # 3. User Input
        tk.Label(top_frame, text="Compare Val:", bg=AC_MINOR, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.entry_ref_val = tk.Entry(top_frame, width=8)
        self.entry_ref_val.pack(side=tk.LEFT, padx=5)
        self.entry_ref_val.bind("<Return>", self.update_plot)

        self.btn_update = tk.Button(top_frame, text="Update", command=self.update_plot, bg="#dddddd")
        self.btn_update.pack(side=tk.LEFT, padx=10)

        # Main Plot Area
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.update_plot()

    def update_plot(self, event=None):
        player_name = self.combo_player.get()
        selected_metric = self.combo_metric.get()

        if not player_name or player_name not in self.player_data:
            return

        self.ax.clear()
        game_values = self.player_data[player_name][selected_metric]
        
        if len(game_values) == 0:
            self.ax.text(0.5, 0.5, "No data available.", ha='center', va='center')
            self.canvas.draw()
            return

        mu = np.mean(game_values)
        sigma = np.std(game_values)
        min_val = min(game_values)
        max_val = max(game_values)
        
        x_start = min_val - 2
        x_end = max_val + 3
        if sigma > 0:
            x_start = mu - 4*sigma
            x_end = mu + 4*sigma
        x_range = np.linspace(x_start, x_end, 1000)

        bins = np.arange(np.floor(x_start), np.ceil(x_end) + 1) - 0.5
        self.ax.hist(game_values, bins=bins, density=True, alpha=0.2, color='orange', edgecolor='black', label='Actual Game Frequency')

        line_height = 0
        if sigma > 0:
            y_pdf = stats.norm.pdf(x_range, mu, sigma)
            self.ax.plot(x_range, y_pdf, color='#007acc', linewidth=3, label='Standard Distribution')
            self.ax.fill_between(x_range, y_pdf, color='#007acc', alpha=0.1)
            line_height = np.max(y_pdf) * 1.1
        else:
            line_height = 1.0
            self.ax.axvline(mu, color='#007acc', linewidth=3, linestyle='--', label='Constant Performance')
        
        self.ax.set_ylim(0, line_height)
        self.ax.vlines(mu, 0, line_height, color=FG_COLOR, linestyle='dashed', alpha=0.8, label='Mean')
        
        if sigma > 0:
            styles = {1: (':', 0.6), 2: (':', 0.3)}
            for s in [1, 2]:
                ls, alpha = styles[s]
                self.ax.vlines(mu + (s * sigma), 0, line_height * 0.9, color='gray', linestyle=ls, alpha=alpha)
                self.ax.vlines(mu - (s * sigma), 0, line_height * 0.9, color='gray', linestyle=ls, alpha=alpha)

        user_val_str = self.entry_ref_val.get()
        if user_val_str and sigma > 0:
            try:
                ref_val = float(user_val_str)
                self.ax.vlines(ref_val, 0, line_height, color='magenta', linewidth=3, label=f'Ref: {ref_val}')
                pct_below = stats.norm.cdf(ref_val, mu, sigma) * 100
                pct_above = 100 - pct_below
                self.ax.text(ref_val, line_height * 0.85, f"  ABOVE: {pct_above:.1f}%\n  (Likelihood > {ref_val})", color='magenta', fontweight='bold', ha='left')
                self.ax.text(ref_val, line_height * 0.85, f"BELOW: {pct_below:.1f}%  \n(Likelihood < {ref_val})  ", color='magenta', fontweight='bold', ha='right')
            except ValueError:
                pass

        self.ax.set_title(f"{player_name}: {selected_metric} per Game Distribution", fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel(f"{selected_metric} Count", fontsize=11)
        self.ax.set_ylabel("Probability Density", fontsize=11)
        self.ax.legend(loc='upper right', frameon=True)
        self.ax.grid(True, linestyle='--', alpha=0.4)
        
        stats_text = (f"Mean: {mu:.2f}\nStd Dev: {sigma:.2f}\nGames: {len(game_values)}")
        self.ax.text(0.02, 0.95, stats_text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self.canvas.draw()


# =============================================================================
# VIEW 3: TEAM GOALS VISUALIZER
# =============================================================================
class TeamGoalsView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.SaveDir = "dump"
        self.controller = controller
        
        self.raw_data = {}
        self.teams_list = []
        
        # --- GUI Layout ---
        control_frame = tk.Frame(self, pady=10, padx=10, bg=AC_COLOR)
        control_frame.pack(fill="x")
        
        tk.Label(control_frame, text="TEAM ANALYSIS", bg=AC_MINOR, fg="#555", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)

        self.btn_load = tk.Button(control_frame, text="Load JSON File", command=self.load_file, bg="#ddd")
        self.btn_load.pack(side="left", padx=10)

        tk.Label(control_frame, text="Select Team:", bg=AC_MINOR).pack(side="left", padx=5)
        self.team_var = tk.StringVar()
        self.team_combo = ttk.Combobox(control_frame, textvariable=self.team_var, state="disabled")
        self.team_combo.pack(side="left", padx=5)
        self.team_combo.bind("<<ComboboxSelected>>", self.on_team_select)

        self.plot_frame = tk.Frame(self, bg="white")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5)) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.auto_load_default()

    def auto_load_default(self):
        default_file = os.path.join(self.SaveDir, f"team_analysis_Uni{datetime.now().date()}.json")
        if os.path.exists(default_file):
            try:
                with open(default_file, 'r', encoding='utf-8') as f:
                    self.process_json(json.load(f))
            except:
                messagebox.showerror("ERROR", "FILE MISSING PLEASE DUMP DATA BEFORE USING")

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.process_json(data)
        except Exception as e:
            messagebox.showerror("Error", f"FILE NOT FOUND, PLEASE DUMP DATA")

    def process_json(self, data):
        self.raw_data = data
        self.teams_list = sorted(list(data.keys()))
        self.team_combo['values'] = self.teams_list
        self.team_combo['state'] = "readonly"
        if self.teams_list:
            self.team_combo.current(0)
            self.on_team_select(None)

    def on_team_select(self, event):
        team = self.team_var.get()
        if not team or team not in self.raw_data:
            return

        team_data = self.raw_data[team].get("date-data", [])
        if not team_data:
            return

        df = pd.DataFrame(team_data)
        max_p1 = df['period1Goals'].max()
        if pd.isna(max_p1): max_p1 = 0
        thresholds = [x + 0.5 for x in range(int(max_p1) + 1)]
        over_probs_list = []
        for t in thresholds:
            prob = (df['period1Goals'] > t).mean() * 100
            over_probs_list.append(prob)
        p1_probs = pd.Series(over_probs_list, index=thresholds)

        p1_p2 = pd.crosstab(index=df['period1Goals'], columns=df['period2Goals'], normalize='index') * 100
        df['goals_entering_3rd'] = df['period1Goals'] + df['period2Goals']
        p1p2_p3 = pd.crosstab(index=df['goals_entering_3rd'], columns=df['period3Goals'], normalize='index') * 100
        self.draw_charts(team, p1_probs, p1_p2, p1p2_p3)

    def draw_charts(self, team_name, p1_probs, matrix1, matrix2):
        self.axs[0].clear()
        self.axs[1].clear()
        self.axs[2].clear()

        sns.barplot(x=p1_probs.index, y=p1_probs.values, ax=self.axs[0], color="skyblue")
        self.axs[0].set_title(f"{team_name}: Probability > X Goals (Period 1)")
        self.axs[0].set_ylabel("Probability (%)")
        self.axs[0].set_xlabel("Goal Threshold")
        for i, v in enumerate(p1_probs.values):
            self.axs[0].text(i, v + 1, f"{v:.1f}%", ha='center', color=FG_COLOR, fontsize=9)

        sns.heatmap(matrix1, annot=True, fmt=".0f", cmap="Blues", ax=self.axs[1], cbar=False)
        self.axs[1].set_title(f"P1 Goals vs P2 Goals (%)")
        self.axs[1].set_ylabel("P1 Goals")
        self.axs[1].set_xlabel("P2 Goals")

        sns.heatmap(matrix2, annot=True, fmt=".0f", cmap="Greens", ax=self.axs[2], cbar=False)
        self.axs[2].set_title(f"Entering 3rd vs P3 Goals (%)")
        self.axs[2].set_ylabel("Total (P1+P2)")
        self.axs[2].set_xlabel("P3 Goals")

        self.fig.tight_layout()
        self.canvas.draw()

class data_dump(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.SaveDir = "dump"
        self.controller = controller
        
        self.raw_data = {}
        self.teams_list = []
        
        # --- GUI Layout ---
        control_frame = tk.Frame(self, pady=10, padx=10, bg=BG_COLOR)
        control_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(control_frame, text="Data Dump", bg=AC_MINOR, fg="#555", font=("Arial", 12, "bold")).pack(side=tk.TOP, pady=10)

        self.btn_load = tk.Button(control_frame, text="Click to Dump Data", command=self.dump_data, bg="#ddd")
        self.btn_load.pack(side=tk.TOP, pady=10)

        self.warning = "This dump will take multiple minutes to complete.\n" + \
        "You will see a pop-up telling you when it is done dumping.\n" + \
        "In the meanwhile, please make some tea or scroll your phone until completed.\n\n" + \
        "YOU MUST DUMP EVERY SINGLE DAY FOR THIS TO WORK. PLAN AHEAD.\n" + \
        "THIS PROGRAM WILL FREEZE AND NOT RESPOND DURING DUMPING. THIS IS NORMAL.\n"
        "THIS MAY NOT APPLY IN FUTURE ITERATIONS BUT FOR NOW IT DOES.\n\n\n"  + \
        "Once you have dumped those files, click on 'Hockey Views' in the top left to switch to a different view and view the data."

        self.info_label = tk.Label(control_frame, text=self.warning)
        self.info_label.pack(side=tk.TOP, pady=10)
    
    def dump_data(self):

        self.TeamData = {}
        self.Roster = {}

        if not os.path.exists(self.SaveDir):
            os.makedirs(self.SaveDir)

        #
        # General Variable Initializations
        #

        if datetime.now().date().month < 6:
            self.season = f"{int(datetime.now().date().year)-1}{int(datetime.now().date().year)}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"
        else:
            self.season = f"{int(datetime.now().date().year)}{int(datetime.now().date().year)+1}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"
        self.EndDate = (datetime.now().date() - timedelta(days=1))
        
        # Load NHL Client API
        self.client = NHLClient(debug=True)

        #Load in all the possible players via the rosters
        for team in tcl.TeamThreeCodes:
            self.Roster[team] = self.client.teams.team_roster(team, self.season)
            self.TeamData[team] = {
                "period1Goals" : 0,
                "period2Goals" : 0,
                "period3Goals" : 0,
                "totalGoals" : 0,
                "date-data" : [],
            }
            time.sleep(0.1)

        self.RetrievalDate = datetime.strptime(self.StartDate, "%Y-%m-%d").date()

        while self.RetrievalDate <= self.EndDate:

            self.DateStr = self.StartDate

            self.DailySchedule = self.client.schedule.daily_schedule(date=str(self.RetrievalDate))

            # Handle API response structure
            self.GamesList = self.DailySchedule.get("games", [])

            for game in self.GamesList:

                # Identify Winner and Loser (for internal logic)
                self.HomeTeam = game["homeTeam"]
                self.AwayTeam = game["awayTeam"]

                #Make containers inside object if not present
                if not self.HomeTeam["abbrev"] in self.TeamData:
                    self.TeamData[self.HomeTeam["abbrev"]] = {}
                if not self.AwayTeam["abbrev"] in self.TeamData:
                    self.TeamData[self.AwayTeam["abbrev"]] = {}

                # Get game data from PlayByPlay
                # Only way to get Offensive, Hits and Goalie Stats all at once.
                self.TempHomeData, self.TempAwayData = self.PlayByPlay(game["id"])

                # Accumulate stats in dictionary
                for key in self.TeamData[self.HomeTeam["abbrev"]].items():
                    key = key[0]
                    if not key == "date-data" and not key == "outcome":
                        self.TeamData[self.HomeTeam["abbrev"]][key] += self.TempHomeData[key]
                for key in self.TeamData[self.AwayTeam["abbrev"]].items():
                    key = key[0]
                    if not key == "date-data" and not key == "outcome":
                        self.TeamData[self.AwayTeam["abbrev"]][key] += self.TempAwayData[key]

                # Save the date data for later
                # Useful for predictive models
                self.TeamData[self.HomeTeam["abbrev"]]["date-data"].append(self.TempHomeData)
                self.TeamData[self.AwayTeam["abbrev"]]["date-data"].append(self.TempAwayData)

            self.RetrievalDate += timedelta(days=1)

        if not os.path.exists(self.SaveDir):
            os.makedirs(f"{self.SaveDir}")

        # Save the compiled list
        self.OutputFile = f'team_analysis_Uni{datetime.now().date()}.json'
        with open(os.path.join(self.SaveDir, self.OutputFile), 'w', encoding='utf-8') as f:
            json.dump(self.TeamData, f, indent=4, ensure_ascii=False)

        self.roster = {}
        self.player_data = {}
        self.goalie_record = {}

        if datetime.now().date().month < 6:
            self.season = f"{int(datetime.now().date().year)-1}{int(datetime.now().date().year)}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"
        else:
            self.season = f"{int(datetime.now().date().year)}{int(datetime.now().date().year)+1}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"

        for team in tcl.TeamThreeCodes:
            self.roster[team] = self.client.teams.team_roster(team, "current")
            for player in self.roster[team]["goalies"]:
                self.player_data[player["id"]] = {
                    "game" : self.client.stats.player_game_log(player_id=player["id"], season_id=self.season, game_type=2),
                    "name" : f"{player["firstName"]["default"]} {player["lastName"]["default"]}"
                }

        for player in self.player_data:
            self.goalie_record[player] = {
                "Name" : self.player_data[player]["name"],
                "shotsAgainst" : [],
                "goalsAgainst" : [],
                "saves" : [],
            }
            for game in self.player_data[player]["game"]:
                self.goalie_record[player]["shotsAgainst"].append(game.get("shotsAgainst", 0))
                self.goalie_record[player]["goalsAgainst"].append(game.get("goalsAgainst", 0))
                self.goalie_record[player]["saves"].append((game.get("shotsAgainst", 0)-game.get("goalsAgainst", 0)))
        
        # Save the compiled list
        output_filename = f'goalie_analysis_Uni{datetime.now().date()}.json'
        with open(os.path.join(self.SaveDir, output_filename), 'w', encoding='utf-8') as f:
            json.dump(self.goalie_record, f, indent=4, ensure_ascii=False)

        print(f"--- Analysis Complete. Data saved to {output_filename} ---")

        self.roster = {}
        self.player_data = {}
        self.player_record = {}

        self.position = ["forwards","defensemen"]

        if datetime.now().date().month < 6:
            self.season = f"{int(datetime.now().date().year)-1}{int(datetime.now().date().year)}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"
        else:
            self.season = f"{int(datetime.now().date().year)}{int(datetime.now().date().year)+1}"
            self.StartDate = str(datetime.now().date().year-1) + "-10-01"

        for team in tcl.TeamThreeCodes:
            self.roster[team] = self.client.teams.team_roster(team, "current")
            for pos in self.position:
                for player in self.roster[team][pos]:
                    self.player_data[player["id"]] = {
                        "game" : self.client.stats.player_game_log(player_id=player["id"], season_id=self.season, game_type=2),
                        "name" : f"{player["firstName"]["default"]} {player["lastName"]["default"]}"
                    }

        for player in self.player_data:
            self.player_record[player] = {
                "Name" : self.player_data[player]["name"],
                "shots" : [],
                "goals" : [],
                "assists" : [],
            }
            for game in self.player_data[player]["game"]:
                self.player_record[player]["shots"].append(game.get("shots", 0))
                self.player_record[player]["goals"].append(game.get("goals", 0))
                self.player_record[player]["assists"].append(game.get("assists", 0))
        
        # Save the compiled list
        output_filename = f'player_analysis_Uni{datetime.now().date()}.json'
        with open(os.path.join(self.SaveDir, output_filename), 'w', encoding='utf-8') as f:
            json.dump(self.player_record, f, indent=4, ensure_ascii=False)

        messagebox.OK("Yay", "Task Completed Successfully!")

    def PlayByPlay(self, game_id):

        #
        # Initialize the stats
        #

        HomeTeamStats = {            
                "period1Goals" : 0,
                "period2Goals" : 0,
                "period3Goals" : 0,
                "totalGoals" : 0,
        }

        AwayTeamStats = {
                "period1Goals" : 0,
                "period2Goals" : 0,
                "period3Goals" : 0,
                "totalGoals" : 0,
        }
        
        #
        # Obtain the play-by-play
        #

        PlayByPlay = self.client.game_center.play_by_play(game_id)

        HomeTeam = PlayByPlay["homeTeam"]["id"]
        AwayTeam = PlayByPlay["awayTeam"]["id"]

        for play in PlayByPlay["plays"]:
            match play["typeCode"]:
                case 505:
                    match play["periodDescriptor"]["number"]:
                        case 1:
                            # Goal Code
                            if play["details"]["eventOwnerTeamId"] == HomeTeam:
                                HomeTeamStats["period1Goals"]+=1
                            else:
                                AwayTeamStats["period1Goals"]+=1
                        case 2:
                            # Goal Code
                            if play["details"]["eventOwnerTeamId"] == HomeTeam:
                                HomeTeamStats["period2Goals"]+=1
                            else:
                                AwayTeamStats["period2Goals"]+=1
                        case 3:
                            # Goal Code
                            if play["details"]["eventOwnerTeamId"] == HomeTeam:
                                HomeTeamStats["period3Goals"]+=1
                            else:
                                AwayTeamStats["period3Goals"]+=1
        
        #Return the calculated stats
        return HomeTeamStats, AwayTeamStats


# =============================================================================
# MAIN CONTAINER APP
# =============================================================================
class HockeySuiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hockey Analysis Suite 2026")
        self.geometry("1200x900")
        self.SaveDir = "dump"

        # 1. Create the Menu Bar
        self.menu_bar = tk.Menu(self, bg="#000000", fg="#FCA311")
        self.config(menu=self.menu_bar)

        # 2. Add "Views" Menu
        views_menu = tk.Menu(self.menu_bar, tearoff=0, bg="#000000", fg="#FCA311")
        self.menu_bar.add_cascade(label="Hockey Views", menu=views_menu)
        
        views_menu.add_command(label="Goalie Performance", command=lambda: self.show_frame(GoalieView))
        views_menu.add_command(label="Player Analysis", command=lambda: self.show_frame(PlayerView))
        views_menu.add_command(label="Team Goal Breakdown", command=lambda: self.show_frame(TeamGoalsView))
        views_menu.add_command(label="Data Dump", command=lambda: self.show_frame(data_dump))
        views_menu.add_separator()
        views_menu.add_command(label="Exit", command=self.quit)

        # 3. Container for Stacked Frames
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # 4. Initialize Frames
        self.frames = {}
        for F in (GoalieView, PlayerView, TeamGoalsView, data_dump):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # 5. Show Default
        self.show_frame(data_dump)

    def show_frame(self, context_class):
        frame = self.frames[context_class]
        frame.tkraise()

    def on_closing(self):
        # 1. Stop the GUI loop
        self.destroy()
        # 2. Force-kill the Python process (essential for .exe files)
        sys.exit()

if __name__ == "__main__":
    app = HockeySuiteApp()
    pywinstyles.change_header_color(app, color="#000000")
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()