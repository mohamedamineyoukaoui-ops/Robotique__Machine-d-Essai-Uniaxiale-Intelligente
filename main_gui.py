import tkinter as tk
from tkinter import ttk, messagebox, filedialog as fd
import serial
import serial.tools.list_ports
import threading
import time
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import joblib

# =============================================================================
# THEME CONFIGURATION
# =============================================================================

class ModernTheme:
    """Thème interface"""
    
    # Couleurs principales
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_CARD = "#16213e"
    BG_CARD_HOVER = "#1e293b"
    
    # Accents
    ACCENT_PRIMARY = "#0f3460"
    ACCENT_SUCCESS = "#10b981"
    ACCENT_WARNING = "#f59e0b"
    ACCENT_DANGER = "#ef4444"
    ACCENT_INFO = "#3b82f6"
    
    # Texte
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_MUTED = "#94a3b8"
    
    # Boutons
    BTN_PRIMARY = "#0f3460"
    BTN_PRIMARY_HOVER = "#1e4976"
    BTN_SUCCESS = "#10b981"
    BTN_SUCCESS_HOVER = "#34d399"
    BTN_DANGER = "#ef4444"
    BTN_DANGER_HOVER = "#f87171"
    BTN_SECONDARY = "#334155"
    BTN_SECONDARY_HOVER = "#475569"
    
    # Borders
    BORDER_COLOR = "#475569"
    
    @staticmethod
    def apply_button_style(btn, style="primary"):
        """Applique un style aux boutons"""
        pass  # Style simple sans hover

# =============================================================================
# ALGORITHME DE TRAITEMENT (Lissage + Régression)
# =============================================================================

class MaterialAnalyzer:
    """Analyseur unique avec lissage et régression"""
    
    @staticmethod
    def lissage_moyenne_glissante(y, window_size=7):
        """Lisse le signal par moyenne glissante"""
        if window_size < 3:
            return np.array(y, dtype=float)
        
        y = np.array(y, dtype=float)
        n = len(y)
        y_lisse = np.zeros_like(y)
        
        demi = window_size // 2
        for i in range(n):
            i_min = max(0, i - demi)
            i_max = min(n, i + demi + 1)
            y_lisse[i] = np.mean(y[i_min:i_max])
        
        return y_lisse

    @staticmethod
    def find_yield_strength(strain, stress, E, offset=0.002):
        """Calcule limite élasticité avec offset"""
        strain = np.array(strain, dtype=float)
        stress = np.array(stress, dtype=float)
        idx = np.argsort(strain)
        strain = strain[idx]
        stress = stress[idx]
        
        sigma_offset = E * (strain - offset)
        diff = stress - sigma_offset
        
        yield_index = None
        i_max = int(np.argmax(stress))
        for i in range(0, i_max + 1):
            if diff[i] < 0:
                yield_index = i
                break
        
        if yield_index is None:
            return None, None
        
        if yield_index > 0:
            x1, y1 = strain[yield_index - 1], diff[yield_index - 1]
            x2, y2 = strain[yield_index], diff[yield_index]
            eps_yield = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            sigma_yield = (stress[yield_index - 1] +
                          (eps_yield - x1) * (stress[yield_index] - stress[yield_index - 1]) / (x2 - x1))
        else:
            eps_yield = strain[yield_index]
            sigma_yield = stress[yield_index]
        
        return sigma_yield, eps_yield

    @staticmethod
    def analyze(strain, stress, window_size=7):
        """Analyse complète du matériau"""
        strain = np.array(strain, dtype=float)
        stress = np.array(stress, dtype=float)
        idx = np.argsort(strain)
        strain_sorted = strain[idx]
        stress_sorted = stress[idx]
        
        # Lissage
        stress_smooth = MaterialAnalyzer.lissage_moyenne_glissante(stress_sorted, window_size)
        
        # Zone élastique
        eps0 = strain_sorted[0]
        eps = strain_sorted - eps0
        max_strain = eps[-1]
        
        if max_strain >= 0.005:
            elastic_threshold = 0.002
        else:
            elastic_threshold = 0.2 * max_strain
        
        mask = eps <= elastic_threshold
        if np.sum(mask) < 2:
            mask = np.zeros_like(eps, dtype=bool)
            mask[:min(len(eps), 5)] = True
        
        # Régression linéaire
        slope, intercept = np.polyfit(eps[mask], stress_smooth[mask], 1)
        
        sigma_pred = slope * eps[mask] + intercept
        ss_res = np.sum((stress_smooth[mask] - sigma_pred)**2)
        ss_tot = np.sum((stress_smooth[mask] - np.mean(stress_smooth[mask]))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        
        # Propriétés mécaniques
        E = abs(slope)
        Su = np.max(stress_sorted)
        Sy, _ = MaterialAnalyzer.find_yield_strength(strain_sorted, stress_sorted, slope)
        if Sy is None:
            Sy = 0.8 * Su
        
        return {
            "E": E,
            "Sy": Sy,
            "Su": Su,
            "r2": r2,
            "intercept": intercept,
            "smooth_stress": stress_smooth,
            "stress_sorted": stress_sorted,
            "strain_sorted": strain_sorted,
            "elastic_strain": strain_sorted[mask],
            "elastic_stress": stress_smooth[mask],
            "n_points": np.sum(mask)
        }

# =============================================================================
# PRÉDICTEUR ML (avec fallback simulé)
# =============================================================================

class MLPredictor:
    def __init__(self):
        self.model = None
        self.has_model = False
        self.le = None
        # Load only the required model at the specified absolute path.
        model_path = r"C:\Users\simoo\OneDrive\Bureau\interface_robotique\rf_material_classifier.pkl"
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle manquant: {model_path}")
            loaded = joblib.load(model_path)
            # If joblib returned a dict, try to extract common keys
            if isinstance(loaded, dict):
                # attempt to find nested model object
                model_candidates = ['model', 'estimator', 'clf', 'classifier', 'pipeline', 'rf', 'est']
                found = False
                for k in model_candidates:
                    if k in loaded and hasattr(loaded[k], 'predict'):
                        self.model = loaded[k]
                        found = True
                        break

                # if label encoder packaged inside the dict
                for lekey in ('label_encoder', 'le', 'encoder'):
                    if lekey in loaded:
                        self.le = loaded[lekey]
                        break

                if not found:
                    # as a fallback, if any dict value has predict, pick it
                    for v in loaded.values():
                        if hasattr(v, 'predict'):
                            self.model = v
                            found = True
                            break
                    if not found:
                        raise RuntimeError('Le fichier pkl contient un dict mais aucun objet avec .predict n\'a été trouvé')
            else:
                self.model = loaded

            # try loading label encoder next to model file (optional) if not found in dict
            if self.le is None:
                le_path = os.path.join(os.path.dirname(model_path), 'label_encoder.pkl')
                if os.path.exists(le_path):
                    try:
                        self.le = joblib.load(le_path)
                    except Exception:
                        self.le = None

            # ensure loaded object has predict
            if not hasattr(self.model, 'predict'):
                raise RuntimeError(f"Objet chargé ne contient pas de méthode 'predict' (type={type(self.model)})")

            self.has_model = True
            print(f"✅ Modèle ML chargé: {model_path} (type: {type(self.model)})")
        except Exception as e:
            self.has_model = False
            # show an error so the user knows the model is required
            try:
                messagebox.showerror('ML Model', f"Échec du chargement du modèle ML:\n{e}\n\nPlacez rf_material_classifier.pkl à {model_path}.")
            except Exception:
                print(f"Erreur chargement modèle ML: {e}")

    def predict(self, E, Sy, Su):
        """Prédiction du matériau"""
        if not self.has_model:
            # Model is required — inform user and return empty string
            try:
                messagebox.showerror('ML Model', 'Modèle ML non chargé — impossible de prédire.')
            except Exception:
                print('Modèle ML non chargé — impossible de prédire.')
            return ''
        # Perform prediction using the loaded model; be robust to output formats
        try:
            features = np.array([[Su, Sy, E]], dtype=float)
            raw_pred = self.model.predict(features)
        except Exception as e:
            # raise so caller can handle and display to user
            raise RuntimeError(f"Erreur durant la prédiction ML: {e}")

        # Extract scalar prediction
        try:
            pred0 = raw_pred[0] if hasattr(raw_pred, '__iter__') else raw_pred
        except Exception:
            pred0 = raw_pred

        # If label encoder is provided, try to inverse transform
        if self.le is not None:
            try:
                lbl = self.le.inverse_transform([pred0])[0]
                return str(lbl)
            except Exception:
                pass

        # If model exposes classes_, and pred0 is an index, try map
        if hasattr(self.model, 'classes_'):
            try:
                classes = getattr(self.model, 'classes_')
                # if pred0 matches a class label directly, return it
                if pred0 in classes:
                    return str(pred0)
            except Exception:
                pass

        # Fallback: return string representation
        try:
            return str(pred0)
        except Exception:
            return ''

# =============================================================================
# INTERFACE MODERNE
# =============================================================================

class ModernAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Machine d'essai Uniaxiale - Analyse de Matériaux")
        self.root.geometry("1500x950")
        self.root.configure(bg=ModernTheme.BG_DARK)
        self.root.minsize(1200, 800)
        
        # Configure ttk styles
        self.setup_styles()
        
        # Variables
        self.raw_data_strain = []
        self.raw_data_stress = []
        self.user_zoomed = False
        self.is_running = False
        self.serial_obj = None
        self.ml_predictor = MLPredictor()
        self.start_time = 0
        self.data_received = False
        self.analysis_results = None
        
        self.setup_ui()
        self.scan_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configure les styles ttk modernes"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Combobox style
        style.configure('Modern.TCombobox',
                       fieldbackground=ModernTheme.BG_CARD,
                       background=ModernTheme.BG_CARD,
                       foreground=ModernTheme.TEXT_PRIMARY,
                       arrowcolor=ModernTheme.TEXT_PRIMARY,
                       bordercolor=ModernTheme.BORDER_COLOR,
                       lightcolor=ModernTheme.BG_CARD,
                       darkcolor=ModernTheme.BG_CARD)
        
        style.map('Modern.TCombobox',
                 fieldbackground=[('readonly', ModernTheme.BG_CARD)],
                 selectbackground=[('readonly', ModernTheme.ACCENT_PRIMARY)],
                 selectforeground=[('readonly', ModernTheme.TEXT_PRIMARY)])

    def setup_ui(self):
        # === HEADER BAR ===
        header = tk.Frame(self.root, bg=ModernTheme.BG_MEDIUM, height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        # Header content frame
        header_content = tk.Frame(header, bg=ModernTheme.BG_MEDIUM)
        header_content.pack(fill=tk.BOTH, expand=True, padx=30)
        
        # Logo and title
        title_frame = tk.Frame(header_content, bg=ModernTheme.BG_MEDIUM)
        title_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Icon circle
        icon_canvas = tk.Canvas(title_frame, width=50, height=50, 
                               bg=ModernTheme.BG_MEDIUM, highlightthickness=0)
        icon_canvas.pack(side=tk.LEFT, padx=(0, 15), pady=15)
        icon_canvas.create_oval(5, 5, 45, 45, fill=ModernTheme.ACCENT_PRIMARY, outline="")
        icon_canvas.create_text(25, 25, text="⚙", font=("Segoe UI Emoji", 18), fill="white")
        
        title_text_frame = tk.Frame(title_frame, bg=ModernTheme.BG_MEDIUM)
        title_text_frame.pack(side=tk.LEFT, fill=tk.Y, pady=15)
        
        tk.Label(title_text_frame, text="Test de Compression Uniaxiale", 
                font=("Segoe UI", 18, "bold"), bg=ModernTheme.BG_MEDIUM, 
                fg=ModernTheme.TEXT_PRIMARY).pack(anchor=tk.W)
        tk.Label(title_text_frame, text="Analyse et identification de matériaux", 
                font=("Segoe UI", 11), bg=ModernTheme.BG_MEDIUM, 
                fg=ModernTheme.TEXT_SECONDARY).pack(anchor=tk.W)
        
        # Status indicator (right side of header)
        status_frame = tk.Frame(header_content, bg=ModernTheme.BG_MEDIUM)
        status_frame.pack(side=tk.RIGHT, fill=tk.Y, pady=20)
        
        self.status_dot = tk.Canvas(status_frame, width=12, height=12, 
                                   bg=ModernTheme.BG_MEDIUM, highlightthickness=0)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self.status_dot.create_oval(2, 2, 10, 10, fill=ModernTheme.ACCENT_DANGER, 
                                   outline="", tags="dot")
        
        self.lbl_status = tk.Label(status_frame, text="Déconnecté", 
                                  font=("Segoe UI", 11, "bold"), 
                                  bg=ModernTheme.BG_MEDIUM, fg=ModernTheme.ACCENT_DANGER)
        self.lbl_status.pack(side=tk.LEFT)
        
        # === CONTROL PANEL ===
        control_wrapper = tk.Frame(self.root, bg=ModernTheme.BG_DARK)
        control_wrapper.pack(fill=tk.X, padx=30, pady=(15, 10))
        
        control_panel = tk.Frame(control_wrapper, bg=ModernTheme.BG_CARD)
        control_panel.pack(fill=tk.X)
        
        control_inner = tk.Frame(control_panel, bg=ModernTheme.BG_CARD)
        control_inner.pack(fill=tk.X, padx=20, pady=15)
        
        # Connection section
        conn_frame = tk.Frame(control_inner, bg=ModernTheme.BG_CARD)
        conn_frame.pack(side=tk.LEFT)
        
        # Port COM
        port_frame = tk.Frame(conn_frame, bg=ModernTheme.BG_CARD)
        port_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(port_frame, text="PORT COM", font=("Segoe UI", 9, "bold"), 
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED).pack(anchor=tk.W)
        
        port_input_frame = tk.Frame(port_frame, bg=ModernTheme.BG_CARD)
        port_input_frame.pack(anchor=tk.W, pady=(3, 0))
        
        self.combo_ports = ttk.Combobox(port_input_frame, width=12, font=("Segoe UI", 11),
                                        style='Modern.TCombobox')
        self.combo_ports.pack(side=tk.LEFT)
        
        btn_refresh = tk.Button(port_input_frame, text="↻", command=self.scan_ports, 
                               bg=ModernTheme.BTN_SECONDARY, fg="white", 
                               font=("Segoe UI", 12, "bold"), width=3, 
                               relief=tk.FLAT, cursor="hand2", bd=0)
        btn_refresh.pack(side=tk.LEFT, padx=(5, 0))
        ModernTheme.apply_button_style(btn_refresh, "secondary")
        
        # Baudrate
        baud_frame = tk.Frame(conn_frame, bg=ModernTheme.BG_CARD)
        baud_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(baud_frame, text="BAUDRATE", font=("Segoe UI", 9, "bold"), 
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED).pack(anchor=tk.W)
        
        self.combo_baud = ttk.Combobox(baud_frame, values=[115200, 9600, 57600], 
                                       width=10, font=("Segoe UI", 11),
                                       style='Modern.TCombobox')
        self.combo_baud.current(0)
        self.combo_baud.pack(anchor=tk.W, pady=(3, 0))
        
        # Points counter
        pts_frame = tk.Frame(conn_frame, bg=ModernTheme.BG_CARD)
        pts_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        tk.Label(pts_frame, text="POINTS", font=("Segoe UI", 9, "bold"), 
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED).pack(anchor=tk.W)
        
        self.lbl_count = tk.Label(pts_frame, text="0", font=("Segoe UI", 16, "bold"), 
                                 bg=ModernTheme.BG_CARD, fg=ModernTheme.ACCENT_INFO)
        self.lbl_count.pack(anchor=tk.W)
        
        # Spacer
        tk.Frame(control_inner, bg=ModernTheme.BG_CARD).pack(side=tk.LEFT, expand=True)
        
        # Action buttons
        btn_frame = tk.Frame(control_inner, bg=ModernTheme.BG_CARD)
        btn_frame.pack(side=tk.RIGHT)
        
        self.btn_start = tk.Button(btn_frame, text="▶  DÉMARRER", command=self.start_acquisition,
                                  bg=ModernTheme.BTN_SUCCESS, fg="white", 
                                  font=("Segoe UI", 12, "bold"),
                                  relief=tk.FLAT, padx=25, pady=10, cursor="hand2", bd=0)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        ModernTheme.apply_button_style(self.btn_start, "success")
        
        self.btn_stop = tk.Button(btn_frame, text="⏹  ANALYSER", command=self.stop_and_analyze,
                                 bg=ModernTheme.BTN_DANGER, fg="white", 
                                 font=("Segoe UI", 12, "bold"),
                                 relief=tk.FLAT, padx=25, pady=10, state=tk.DISABLED, 
                                 cursor="hand2", bd=0)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_dezoom = tk.Button(btn_frame, text="⤢  Réinitialiser vue", command=self.dezoom,
                                   bg=ModernTheme.BTN_SECONDARY, fg="white", 
                                   font=("Segoe UI", 11),
                                   relief=tk.FLAT, padx=15, pady=10, cursor="hand2", bd=0)
        self.btn_dezoom.pack(side=tk.LEFT, padx=5)
        ModernTheme.apply_button_style(self.btn_dezoom, "secondary")
        
        # Curve mode toggle
        mode_frame = tk.Frame(btn_frame, bg=ModernTheme.BG_CARD)
        mode_frame.pack(side=tk.LEFT, padx=(20, 5))
        
        self.curve_mode = tk.StringVar(value='lisse')
        
        rb_lisse = tk.Radiobutton(mode_frame, text='Lissé', variable=self.curve_mode, 
                                 value='lisse', bg=ModernTheme.BG_CARD, 
                                 fg=ModernTheme.TEXT_SECONDARY, 
                                 selectcolor=ModernTheme.BG_DARK,
                                 activebackground=ModernTheme.BG_CARD,
                                 activeforeground=ModernTheme.TEXT_PRIMARY,
                                 font=("Segoe UI", 10),
                                 command=self.update_curve_display)
        rb_lisse.pack(side=tk.LEFT, padx=3)
        
        rb_orig = tk.Radiobutton(mode_frame, text='Original', variable=self.curve_mode, 
                                value='orig', bg=ModernTheme.BG_CARD, 
                                fg=ModernTheme.TEXT_SECONDARY,
                                selectcolor=ModernTheme.BG_DARK,
                                activebackground=ModernTheme.BG_CARD,
                                activeforeground=ModernTheme.TEXT_PRIMARY,
                                font=("Segoe UI", 10),
                                command=self.update_curve_display)
        rb_orig.pack(side=tk.LEFT, padx=3)
        
        # Fullscreen toggle
        self.is_fullscreen = False
        self.btn_fullscreen = tk.Button(btn_frame, text="⛶", command=self.toggle_fullscreen,
                                       bg=ModernTheme.BTN_SECONDARY, fg="white", 
                                       font=("Segoe UI", 14),
                                       relief=tk.FLAT, width=3, cursor="hand2", bd=0)
        self.btn_fullscreen.pack(side=tk.LEFT, padx=(10, 0))
        ModernTheme.apply_button_style(self.btn_fullscreen, "secondary")
        
        # === MAIN CONTENT ===
        content = tk.Frame(self.root, bg=ModernTheme.BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 20))
        
        # === GRAPH (Left) ===
        graph_wrapper = tk.Frame(content, bg=ModernTheme.BG_CARD)
        graph_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        graph_frame = tk.Frame(graph_wrapper, bg=ModernTheme.BG_CARD)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor(ModernTheme.BG_CARD)
        self.ax.set_facecolor('#1e1e32')
        
        self.line_data, = self.ax.plot([], [], 'o-', color='#60a5fa', linewidth=2.5, 
                                       markersize=4, label='Données expérimentales', 
                                       alpha=0.9, markeredgecolor='white', markeredgewidth=0.5)
        self.line_tangent, = self.ax.plot([], [], '--', color='#fbbf24', linewidth=3,
                                          label='Tangente élastique (E)', visible=False)
        
        self.ax.set_xlabel('Déformation ε', fontsize=13, color='white', fontweight='bold', labelpad=10)
        self.ax.set_ylabel('Contrainte σ (MPa)', fontsize=13, color='white', fontweight='bold', labelpad=10)
        self.ax.set_title('Courbe Contrainte-Déformation', fontsize=16, 
                         color='white', fontweight='bold', pad=20)
        self.ax.tick_params(colors='#a1a1aa', labelsize=10)
        self.ax.grid(True, alpha=0.15, color='#6366f1', linestyle='-', linewidth=0.5)
        self.ax.legend(loc='upper left', facecolor='#252542', edgecolor='#3f3f5a', 
                      labelcolor='white', fontsize=10, framealpha=0.95)
        
        for spine in self.ax.spines.values():
            spine.set_color('#3f3f5a')
            spine.set_linewidth(1.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Zoom handlers
        self.zoom_rect = None
        self.zoom_start = None
        self.zooming = False
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Instruction label
        instr = tk.Label(graph_frame, text="🔍 Glisser pour zoomer  •  Double-clic pour réinitialiser",
                        bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED, 
                        font=("Segoe UI", 10))
        instr.pack(side=tk.BOTTOM, pady=(0, 10))
        
        # === RESULTS PANEL (Right) ===
        results_wrapper = tk.Frame(content, bg=ModernTheme.BG_CARD, width=380)
        results_wrapper.pack(side=tk.RIGHT, fill=tk.Y)
        results_wrapper.pack_propagate(False)
        
        results_panel = tk.Frame(results_wrapper, bg=ModernTheme.BG_CARD)
        results_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with icon
        title_frame = tk.Frame(results_panel, bg=ModernTheme.BG_CARD)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(title_frame, text="📊", font=("Segoe UI Emoji", 20),
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT)
        tk.Label(title_frame, text="RÉSULTATS", font=("Segoe UI", 18, "bold"),
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_PRIMARY).pack(side=tk.LEFT, padx=(10, 0))
        
        # Material prediction card (featured)
        prediction_card = tk.Frame(results_panel, bg=ModernTheme.ACCENT_PRIMARY)
        prediction_card.pack(fill=tk.X, pady=(0, 20))
        
        prediction_inner = tk.Frame(prediction_card, bg='#1e1b4b')
        prediction_inner.pack(fill=tk.BOTH, padx=3, pady=3)
        
        tk.Label(prediction_inner, text="🎯 MATÉRIAU DÉTECTÉ", font=("Segoe UI", 10, "bold"),
                bg='#1e1b4b', fg=ModernTheme.TEXT_MUTED).pack(pady=(15, 5))
        
        self.lbl_prediction = tk.Label(prediction_inner, text="---", 
                                      font=("Segoe UI", 24, "bold"), bg='#1e1b4b', 
                                      fg=ModernTheme.ACCENT_SUCCESS, wraplength=300)
        self.lbl_prediction.pack(pady=(0, 15))
        
        # Properties cards
        props_frame = tk.Frame(results_panel, bg=ModernTheme.BG_CARD)
        props_frame.pack(fill=tk.BOTH, expand=True)
        
        # Module de Young
        card_E = self.create_property_card(props_frame, "E", "Module de Young", "MPa", "📐")
        card_E.pack(fill=tk.X, pady=6)
        
        # Limite élastique
        card_Sy = self.create_property_card(props_frame, "Sy", "Limite élastique", "MPa", "📏")
        card_Sy.pack(fill=tk.X, pady=6)
        
        # Résistance
        card_Su = self.create_property_card(props_frame, "Su", "Résistance maximale", "MPa", "💪")
        card_Su.pack(fill=tk.X, pady=6)
        
        # R²
        card_r2 = self.create_property_card(props_frame, "R²", "Qualité régression", "", "✓")
        card_r2.pack(fill=tk.X, pady=6)

        # Spacer
        tk.Frame(results_panel, bg=ModernTheme.BG_CARD, height=15).pack(fill=tk.X)

        # Export section with header
        export_section = tk.Frame(results_panel, bg=ModernTheme.BG_CARD)
        export_section.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        tk.Label(export_section, text="━━  EXPORTATION  ━━", font=("Segoe UI", 10, "bold"),
                bg=ModernTheme.BG_CARD, fg=ModernTheme.TEXT_MUTED).pack(pady=(0, 12))

        # Export buttons row
        export_btn_frame = tk.Frame(export_section, bg=ModernTheme.BG_CARD)
        export_btn_frame.pack(fill=tk.X, pady=(0, 5))

        self.btn_export_csv = tk.Button(export_btn_frame, text="📄  Exporter CSV", command=self.export_csv,
                                       bg=ModernTheme.BTN_SECONDARY, fg="white", 
                                       font=("Segoe UI", 11, "bold"),
                                       relief=tk.FLAT, padx=15, pady=12, cursor="hand2", bd=0)
        self.btn_export_csv.pack(side=tk.LEFT, padx=(0, 8), expand=True, fill=tk.X)
        ModernTheme.apply_button_style(self.btn_export_csv, "secondary")

        self.btn_export_png = tk.Button(export_btn_frame, text="📷  Exporter PNG", command=self.export_png,
                                       bg=ModernTheme.BTN_SECONDARY, fg="white", 
                                       font=("Segoe UI", 11, "bold"),
                                       relief=tk.FLAT, padx=15, pady=12, cursor="hand2", bd=0)
        self.btn_export_png.pack(side=tk.LEFT, padx=0, expand=True, fill=tk.X)
        ModernTheme.apply_button_style(self.btn_export_png, "secondary")

        # Database button - very prominent green button
        self.btn_confirm_db = tk.Button(export_section, text="💾  SAUVEGARDER DANS LA BASE", 
                                       command=self.confirm_add_to_db,
                                       bg="#10b981", fg="white", 
                                       font=("Segoe UI", 12, "bold"),
                                       relief=tk.RAISED, pady=14, cursor="hand2", bd=2,
                                       activebackground="#34d399", activeforeground="white")
        self.btn_confirm_db.pack(fill=tk.X, pady=(15, 10))
        ModernTheme.apply_button_style(self.btn_confirm_db, "primary")

    def create_property_card(self, parent, prop_id, label, unit, icon=""):
        """Crée une carte de propriété moderne avec icône"""
        card = tk.Frame(parent, bg=ModernTheme.BG_DARK)
        
        # Icon
        if icon:
            tk.Label(card, text=icon, font=("Segoe UI Emoji", 14), 
                    bg=ModernTheme.BG_DARK, fg=ModernTheme.TEXT_MUTED).pack(side=tk.LEFT, padx=(12, 8))
        
        # Content
        content_frame = tk.Frame(card, bg=ModernTheme.BG_DARK)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=12)
        
        tk.Label(content_frame, text=label, font=("Segoe UI", 10), 
                bg=ModernTheme.BG_DARK, fg=ModernTheme.TEXT_MUTED).pack(anchor=tk.W)
        
        value_frame = tk.Frame(content_frame, bg=ModernTheme.BG_DARK)
        value_frame.pack(anchor=tk.W)
        
        value_label = tk.Label(value_frame, text="---", font=("Segoe UI", 20, "bold"),
                              bg=ModernTheme.BG_DARK, fg=ModernTheme.TEXT_PRIMARY)
        value_label.pack(side=tk.LEFT)
        
        if unit:
            tk.Label(value_frame, text=f" {unit}", font=("Segoe UI", 11), 
                    bg=ModernTheme.BG_DARK, fg=ModernTheme.TEXT_MUTED).pack(side=tk.LEFT, pady=(5, 0))
        
        # Store reference
        setattr(self, f"lbl_{prop_id.lower().replace('²', '2')}", value_label)
        
        return card
    
    def update_status(self, text, color):
        """Met à jour l'indicateur de status avec animation"""
        self.lbl_status.config(text=text, fg=color)
        self.status_dot.delete("dot")
        self.status_dot.create_oval(2, 2, 10, 10, fill=color, outline="", tags="dot")

    def scan_ports(self):
        """Scan des ports COM"""
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        self.combo_ports['values'] = port_list
        if port_list:
            self.combo_ports.current(0)
            print(f"✅ Ports: {', '.join(port_list)}")

    def start_acquisition(self):
        """Démarrage acquisition"""
        port = self.combo_ports.get()
        baud = self.combo_baud.get()
        
        if not port:
            messagebox.showerror("Erreur", "Sélectionnez un port COM")
            return

        if self.serial_obj and self.serial_obj.is_open:
            try:
                self.serial_obj.close()
                time.sleep(0.5)
            except:
                pass

        try:
            print(f"🔌 Connexion: {port} @ {baud} bauds...")
            self.serial_obj = serial.Serial(port, int(baud), timeout=1)
            time.sleep(2)
            self.serial_obj.reset_input_buffer()
            
            self.is_running = True
            self.start_time = time.time()
            self.data_received = False
            self.raw_data_strain = []
            self.raw_data_stress = []
            self.analysis_results = None
            
            # Reset UI
            self.line_data.set_data([], [])
            self.line_tangent.set_visible(False)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.lbl_status.config(text="● Connecté", fg="#10b981")
            self.lbl_prediction.config(text="Acquisition...", fg="#3b82f6")
            
            # Reset results
            self.lbl_e.config(text="---")
            self.lbl_sy.config(text="---")
            self.lbl_su.config(text="---")
            self.lbl_r2.config(text="---")
            
            # Threads
            threading.Thread(target=self.read_serial_data, daemon=True).start()
            self.animate_plot()

        except Exception as e:
            if "PermissionError" in str(e) or "Access" in str(e):
                messagebox.showerror("Port Occupé", 
                    f"❌ Port {port} occupé!\n\nFermez Arduino IDE et réessayez.")
            else:
                messagebox.showerror("Erreur", f"Connexion impossible:\n{e}")

    def read_serial_data(self):
        """Lecture série"""
        print("🔌 Thread lecture démarré")
        
        while self.is_running and self.serial_obj.is_open:
            try:
                line = self.serial_obj.readline().decode('utf-8').strip()
                
                if line:
                    if line.startswith('#'):
                        print(f"📝 {line}")
                        
                        # Détection automatique de fin de test
                        if "FIN DU TEST" in line or "FIN" in line.upper():
                            print("✅ Fin de test détectée - Analyse automatique...")
                            # Attendre 500ms pour être sûr que toutes les données sont arrivées
                            time.sleep(0.5)
                            # Déclencher l'analyse automatiquement
                            self.root.after(100, self.auto_analyze)
                            break
                        
                        continue
                    
                    parts = line.split(',')
                    if len(parts) == 2:
                        strain_val = float(parts[0])
                        stress_val = float(parts[1])
                        
                        self.raw_data_strain.append(strain_val)
                        self.raw_data_stress.append(stress_val)
                        self.data_received = True
                        
                        if len(self.raw_data_strain) == 1:
                            print(f"✅ Première donnée: ε={strain_val:.6f}, σ={stress_val:.2f}")
                        elif len(self.raw_data_strain) == 50:
                            print(f"📊 50 points reçus - Courbe se forme...")
                        elif len(self.raw_data_strain) == 100:
                            print(f"📊 100 points reçus - Données suffisantes pour analyse")
                            
            except ValueError:
                pass
            except Exception as e:
                print(f"❌ Erreur: {e}")
                break
        
        print("🔌 Thread terminé")

    def animate_plot(self):
        """Animation temps réel"""
        if self.is_running:
            n_points = len(self.raw_data_strain)
            self.lbl_count.config(text=f"{n_points} pts")
            
            # Timeout check
            elapsed = time.time() - self.start_time
            if not self.data_received and elapsed > 5:
                self.is_running = False
                if self.serial_obj:
                    self.serial_obj.close()
                self.btn_start.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                self.lbl_status.config(text="● Timeout", fg="#ef4444")
                
                messagebox.showerror("Timeout", 
                    "❌ Aucune donnée après 5 secondes!\n\n"
                    "Vérifiez:\n"
                    "• Baudrate = 115200\n"
                    "• Code Arduino téléversé\n"
                    "• Port COM correct\n"
                    "• Câble USB branché\n\n"
                    "💡 Ouvrez le Moniteur Série Arduino\n"
                    "pour vérifier l'envoi des données.")
                return
            
            # Update status
            if n_points > 0:
                if n_points >= 100:
                    self.lbl_status.config(text="● Réception OK", fg="#10b981")
                else:
                    self.lbl_status.config(text="● Réception", fg="#3b82f6")
            else:
                self.lbl_status.config(text="● Attente...", fg="#f59e0b")
            
            # Update graph
            if n_points > 1:
                try:
                    mode = self.curve_mode.get()
                except Exception:
                    mode = 'lisse'

                if mode == 'lisse':
                    try:
                        arr_x = np.array(self.raw_data_strain)
                        arr_y = np.array(self.raw_data_stress)
                        if len(arr_x) > 1:
                            idx = np.argsort(arr_x)
                            xs = arr_x[idx]
                            ys = arr_y[idx]
                            ys_s = MaterialAnalyzer.lissage_moyenne_glissante(ys, window_size=7)
                            self.line_data.set_data(xs, ys_s)
                        else:
                            self.line_data.set_data(self.raw_data_strain, self.raw_data_stress)
                    except Exception:
                        self.line_data.set_data(self.raw_data_strain, self.raw_data_stress)
                else:
                    self.line_data.set_data(self.raw_data_strain, self.raw_data_stress)

                # Ne pas ré-autoscaler si l'utilisateur a zoomé manuellement
                if not getattr(self, 'user_zoomed', False):
                    self.ax.relim()
                    self.ax.autoscale_view()
            
            self.canvas.draw_idle()
            self.root.after(100, self.animate_plot)

    def stop_and_analyze(self):
        """Arrêt et analyse (manuel)"""
        self.perform_analysis()
    
    def auto_analyze(self):
        """Analyse automatique (déclenchée par Arduino)"""
        print("🤖 Déclenchement analyse automatique...")
        self.lbl_prediction.config(text="Analyse auto...", fg="#f59e0b")
        self.perform_analysis()
    
    def perform_analysis(self):
        """Effectue l'analyse des données"""
        self.is_running = False
        if self.serial_obj and self.serial_obj.is_open:
            self.serial_obj.close()

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="● Analyse...", fg="#f59e0b")

        strain_arr = np.array(self.raw_data_strain)
        stress_arr = np.array(self.raw_data_stress)

        if len(strain_arr) < 10:
            messagebox.showwarning("Données insuffisantes", 
                                 "Minimum 10 points requis!")
            self.lbl_status.config(text="● Déconnecté", fg="#ef4444")
            return

        print("\n" + "="*50)
        print("🔬 ANALYSE EN COURS")
        print("="*50)

        # Analyse
        results = MaterialAnalyzer.analyze(strain_arr, stress_arr, window_size=7)
        self.analysis_results = results

        # Update results
        self.lbl_e.config(text=f"{results['E']:.0f}")
        self.lbl_sy.config(text=f"{results['Sy']:.0f}")
        self.lbl_su.config(text=f"{results['Su']:.0f}")
        self.lbl_r2.config(text=f"{results['r2']:.4f}")

        # ML Prediction (handle errors and empty outputs)
        try:
            prediction = self.ml_predictor.predict(results['E'], results['Sy'], results['Su'])
            if prediction is None or str(prediction).strip() == '':
                raise RuntimeError('Prédiction introuvable ou vide')
            self.lbl_prediction.config(text=prediction, fg="#10b981")
        except Exception as e:
            # Show error and write a visible error label so user knows prediction failed
            err_msg = str(e)
            try:
                self.lbl_prediction.config(text='Erreur ML', fg="#ef4444")
            except Exception:
                pass
            try:
                messagebox.showerror('Prédiction ML', f"Erreur lors de la prédiction:\n{err_msg}")
            except Exception:
                print(f"Erreur lors de la prédiction: {err_msg}")

        # Draw tangent (always attempt to draw results)
        try:
            self.draw_tangent(results)
        except Exception:
            pass

        self.lbl_status.config(text="● Terminé", fg="#10b981")
        
        # Affichage console résumé
        print(f"\n📊 RÉSUMÉ DES RÉSULTATS:")
        print(f"   Module de Young (E)  : {results['E']:.0f} MPa")
        print(f"   Limite élastique (Sy): {results['Sy']:.0f} MPa")
        print(f"   Résistance max (Su)  : {results['Su']:.0f} MPa")
        print(f"   Qualité (R²)         : {results['r2']:.4f}")
        print(f"   Points analysés      : {results['n_points']}/{len(strain_arr)}")
        print(f"\n🤖 MATÉRIAU DÉTECTÉ: {prediction}")
        print("="*50 + "\n")

    def draw_tangent(self, results):
        """Dessine la tangente élastique"""
        # Données complètes (respecter le mode d'affichage)
        try:
            mode = self.curve_mode.get()
        except Exception:
            mode = 'lisse'

        if mode == 'lisse':
            ydata = results.get('smooth_stress', results.get('stress_sorted'))
        else:
            ydata = results.get('stress_sorted', results.get('smooth_stress'))

        self.line_data.set_data(results['strain_sorted'], ydata)
        
        # Tangente
        eps_tangent = results['elastic_strain']
        sigma_tangent = results['E'] * (eps_tangent - results['strain_sorted'][0]) + results['intercept']
        
        self.line_tangent.set_data(eps_tangent, sigma_tangent)
        self.line_tangent.set_visible(True)
        
        # Adjust view (ne pas écraser le zoom utilisateur)
        if not getattr(self, 'user_zoomed', False):
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas.draw()

    def update_curve_display(self):
        """Re-affiche la courbe selon le mode sélectionné (original / lissé)."""
        # If we have analysis results, prefer those sorted arrays
        try:
            mode = self.curve_mode.get()
        except Exception:
            mode = 'lisse'

        if self.analysis_results is not None:
            xs = self.analysis_results.get('strain_sorted')
            if mode == 'lisse':
                ys = self.analysis_results.get('smooth_stress')
            else:
                ys = self.analysis_results.get('stress_sorted', self.analysis_results.get('smooth_stress'))
            if xs is not None and ys is not None:
                self.line_data.set_data(xs, ys)
        else:
            # Use live/raw data
            if mode == 'lisse' and len(self.raw_data_strain) > 1:
                try:
                    arr_x = np.array(self.raw_data_strain)
                    arr_y = np.array(self.raw_data_stress)
                    idx = np.argsort(arr_x)
                    xs = arr_x[idx]
                    ys_s = MaterialAnalyzer.lissage_moyenne_glissante(arr_y[idx], window_size=7)
                    self.line_data.set_data(xs, ys_s)
                except Exception:
                    self.line_data.set_data(self.raw_data_strain, self.raw_data_stress)
            else:
                self.line_data.set_data(self.raw_data_strain, self.raw_data_stress)

        if not getattr(self, 'user_zoomed', False):
            try:
                self.ax.relim()
                self.ax.autoscale_view()
            except Exception:
                pass
        self.canvas.draw_idle()

    def on_button_press(self, event):
        """Start rectangle zoom on left mouse press inside axes."""
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.zoom_start = (event.xdata, event.ydata)
        self.zooming = True
        # create rectangle patch
        self.zoom_rect = Rectangle((event.xdata, event.ydata), 0, 0,
                                   linewidth=1, edgecolor='white', facecolor='white', alpha=0.2)
        self.ax.add_patch(self.zoom_rect)
        self.canvas.draw_idle()

    def on_motion(self, event):
        """Update rectangle while dragging."""
        if not self.zooming or self.zoom_rect is None:
            return
        if event.inaxes != self.ax:
            return
        x0, y0 = self.zoom_start
        x1, y1 = event.xdata, event.ydata
        if x1 is None or y1 is None:
            return
        xmin, ymin = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        self.zoom_rect.set_xy((xmin, ymin))
        self.zoom_rect.set_width(width)
        self.zoom_rect.set_height(height)
        self.canvas.draw_idle()

    def on_release(self, event):
        """Finish zoom: set axes limits to rectangle and remove patch."""
        if not self.zooming:
            return
        self.zooming = False
        if self.zoom_rect is None or self.zoom_start is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            # remove rectangle
            try:
                self.zoom_rect.remove()
            except Exception:
                pass
            self.zoom_rect = None
            self.canvas.draw_idle()
            return

        x0, y0 = self.zoom_start
        x1, y1 = event.xdata, event.ydata
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        # small drag -> ignore
        if abs(xmax - xmin) < 1e-12 or abs(ymax - ymin) < 1e-12:
            try:
                self.zoom_rect.remove()
            except Exception:
                pass
            self.zoom_rect = None
            self.canvas.draw_idle()
            return

        # apply zoom and mark user_zoomed
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.user_zoomed = True

        try:
            self.zoom_rect.remove()
        except Exception:
            pass
        self.zoom_rect = None
        self.canvas.draw_idle()

    def on_click(self, event):
        """Double-clic: réinitialise le zoom (autoscale)."""
        if getattr(event, 'dblclick', False):
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()
            # clear user zoom flag
            self.user_zoomed = False

    def reset_view(self):
        """Reset axes to autoscale and clear zoom flag."""
        try:
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()
            self.user_zoomed = False
        except Exception:
            pass

    def toggle_fullscreen(self):
        """Toggle fullscreen state for the main window."""
        self.is_fullscreen = not self.is_fullscreen
        try:
            self.root.attributes("-fullscreen", self.is_fullscreen)
        except Exception:
            # fallback for platforms not supporting attribute
            if self.is_fullscreen:
                self.root.state('zoomed')
            else:
                self.root.state('normal')

    def export_png(self):
        """Export the current figure as PNG into the predefined images folder."""
        try:
            base = r"C:\Users\simoo\OneDrive\Bureau\exports"
            img_dir = os.path.join(base, 'images')
            os.makedirs(img_dir, exist_ok=True)

            now = datetime.now().strftime('%d-%m-%Y_%H-%M')
            pred = self.lbl_prediction.cget('text') if hasattr(self, 'lbl_prediction') else 'unknown'
            pred_clean = str(pred).replace(' ', '_').replace('/', '_')
            filename = f"{now}_{pred_clean}.png"
            path = os.path.join(img_dir, filename)

            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor())
            messagebox.showinfo('Export', f'PNG sauvegardé: {path}')
        except Exception as e:
            messagebox.showerror('Export PNG', f'Erreur: {e}')

    def export_csv(self):
        """Export current data (strain, stress) to CSV into predefined csv folder."""
        try:
            if len(self.raw_data_strain) == 0:
                messagebox.showwarning('Export CSV', 'Aucune donnée à exporter')
                return

            base = r"C:\Users\simoo\OneDrive\Bureau\exports"
            csv_dir = os.path.join(base, 'csv')
            os.makedirs(csv_dir, exist_ok=True)

            now = datetime.now().strftime('%d-%m-%Y_%H-%M')
            pred = self.lbl_prediction.cget('text') if hasattr(self, 'lbl_prediction') else 'unknown'
            pred_clean = str(pred).replace(' ', '_').replace('/', '_')
            filename = f"{now}_{pred_clean}.csv"
            path = os.path.join(csv_dir, filename)

            import csv
            # Choose data depending on current curve mode and available analysis
            if self.analysis_results is not None and self.curve_mode.get() == 'lisse':
                xs = self.analysis_results.get('strain_sorted')
                ys = self.analysis_results.get('smooth_stress')
            elif self.analysis_results is not None and self.curve_mode.get() == 'orig':
                xs = self.analysis_results.get('strain_sorted')
                ys = self.analysis_results.get('stress_sorted', self.raw_data_stress)
            else:
                xs = self.raw_data_strain
                ys = self.raw_data_stress

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['strain', 'stress'])
                for s, t in zip(xs, ys):
                    writer.writerow([s, t])

            messagebox.showinfo('Export', f'CSV sauvegardé: {path}')
        except Exception as e:
            messagebox.showerror('Export CSV', f'Erreur: {e}')

    def get_database_path(self):
        """Return the path to the central results database CSV (create base folder if needed)."""
        base = r"C:\Users\simoo\OneDrive\Bureau\exports"
        os.makedirs(base, exist_ok=True)
        db_path = os.path.join(base, 'database_results.csv')
        return db_path

    def ensure_database(self):
        """Create database CSV with header if it does not exist."""
        path = self.get_database_path()
        if not os.path.exists(path):
            try:
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'E', 'Sy', 'Su', 'prediction'])
            except Exception as e:
                messagebox.showerror('Database', f'Erreur création database: {e}')
        return path

    def confirm_add_to_db(self):
        """Append current analysis results to central CSV database after confirmation."""
        try:
            if self.analysis_results is None:
                messagebox.showwarning('Database', 'Aucun résultat analysé à ajouter')
                return

            # Ensure DB exists
            db = self.ensure_database()

            ts = datetime.now().strftime('%d/%m/%Y_%H:%M')
            E = f"{self.analysis_results.get('E'):.0f}" if self.analysis_results.get('E') is not None else ''
            Sy = f"{self.analysis_results.get('Sy'):.0f}" if self.analysis_results.get('Sy') is not None else ''
            Su = f"{self.analysis_results.get('Su'):.0f}" if self.analysis_results.get('Su') is not None else ''
            pred = self.lbl_prediction.cget('text') if hasattr(self, 'lbl_prediction') else ''

            import csv
            with open(db, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([ts, E, Sy, Su, pred])

            messagebox.showinfo('Database', f'Résultats ajoutés: {db}')
        except Exception as e:
            messagebox.showerror('Database', f'Erreur ajout à la database: {e}')

    def dezoom(self, factor=1.5):
        """Dézoome incrémental centré sur la vue actuelle (factor > 1 agrandit la fenêtre)."""
        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
        except Exception:
            return

        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        halfx = 0.5 * (x1 - x0) * factor
        halfy = 0.5 * (y1 - y0) * factor

        self.ax.set_xlim(cx - halfx, cx + halfx)
        self.ax.set_ylim(cy - halfy, cy + halfy)
        # mark as user zoomed so autoscale doesn't override
        self.user_zoomed = True
        self.canvas.draw_idle()

    def on_closing(self):
        """Fermeture propre"""
        self.is_running = False
        if self.serial_obj and self.serial_obj.is_open:
            try:
                self.serial_obj.close()
            except:
                pass
        self.root.destroy()

# =============================================================================
# LANCEMENT
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernAnalysisApp(root)
    root.mainloop()
