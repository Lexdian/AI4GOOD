#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network + Tkinter GUI from scratch (no ML libs)
- Classification (multi-class) using softmax + cross-entropy
- Hidden layers with ReLU
- CSV loader, label encoding, min-max normalization
- Train/test split, simple metrics
- Single-sample prediction UI

Usage:
    python nn_gui_from_scratch.py
"""

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# -------------------------------
# Utility: matrix & vector ops
# -------------------------------

def zeros(shape):
    if isinstance(shape, int):
        return [0.0 for _ in range(shape)]
    rows, cols = shape
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def rand_matrix(rows: int, cols: int, scale: float = 0.01) -> List[List[float]]:
    return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]

def rand_vector(n: int, scale: float = 0.01) -> List[float]:
    return [(random.random() * 2 - 1) * scale for _ in range(n)]

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def matvec(W: List[List[float]], x: List[float]) -> List[float]:
    return [dot(row, x) for row in W]

def vecadd(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def vecsub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]

def vecscale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]

def outer(a: List[float], b: List[float]) -> List[List[float]]:
    return [[x * y for y in b] for x in a]

def transpose(M: List[List[float]]) -> List[List[float]]:
    if not M:
        return []
    return [list(col) for col in zip(*M)]

def relu(v: List[float]) -> List[float]:
    return [x if x > 0 else 0.0 for x in v]

def relu_deriv(v: List[float]) -> List[float]:
    return [1.0 if x > 0 else 0.0 for x in v]

def softmax(v: List[float]) -> List[float]:
    # numerically stable softmax
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]

# -------------------------------
# Data handling
# -------------------------------

@dataclass
class Dataset:
    X: List[List[float]]
    y: List[int]
    classes_: List[Any]
    x_min: List[float]
    x_max: List[float]

def read_csv(path: str) -> List[List[str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not all(cell.strip() == "" for cell in row):
                rows.append(row)
    return rows

def try_float(x: str) -> Optional[float]:
    try:
        return float(x.replace(",", "."))
    except:
        return None

def encode_labels(labels: List[str]) -> Tuple[List[int], List[str]]:
    uniq = []
    index = {}
    y_encoded = []
    for lab in labels:
        if lab not in index:
            index[lab] = len(uniq)
            uniq.append(lab)
        y_encoded.append(index[lab])
    return y_encoded, uniq

def minmax_scale_fit(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not X:
        return [], []
    nfeat = len(X[0])
    mins = [float("inf")] * nfeat
    maxs = [float("-inf")] * nfeat
    for row in X:
        for j, val in enumerate(row):
            if val < mins[j]: mins[j] = val
            if val > maxs[j]: maxs[j] = val
    # avoid zero span
    for j in range(nfeat):
        if mins[j] == float("inf"):
            mins[j] = 0.0
        if maxs[j] == float("-inf"):
            maxs[j] = 1.0
        if abs(maxs[j] - mins[j]) < 1e-12:
            maxs[j] = mins[j] + 1.0
    return mins, maxs

def minmax_scale_transform(X: List[List[float]], mins: List[float], maxs: List[float]) -> List[List[float]]:
    Xn = []
    for row in X:
        Xn.append([(row[j] - mins[j]) / (maxs[j] - mins[j]) for j in range(len(row))])
    return Xn

def train_test_split(X, y, test_size=0.2, seed=42):
    random.Random(seed).shuffle(X)
    # Keep pairing with y
    pairs = list(zip(X, y))
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    nt = int(n * (1 - test_size))
    train = pairs[:nt]
    test = pairs[nt:]
    Xtr, ytr = zip(*train) if train else ([], [])
    Xte, yte = zip(*test) if test else ([], [])
    return list(Xtr), list(ytr), list(Xte), list(yte)

def load_dataset(csv_path: str, target_col: int, normalize: bool = True) -> Dataset:
    rows = read_csv(csv_path)
    if not rows:
        raise ValueError("CSV vazio ou não lido.")
    # heuristics: if header contains non-numeric in target, still ok
    # Try detect header by trying to parse first row numbers
    def is_header(row):
        return any(try_float(c) is None for c in row)
    header = None
    if is_header(rows[0]):
        header = rows[0]
        rows = rows[1:]

    X_raw = []
    y_raw = []
    for r in rows:
        if target_col < 0 or target_col >= len(r):
            raise ValueError(f"Coluna alvo {target_col} fora do intervalo.")
        y_raw.append(r[target_col])
        feats = [c for j, c in enumerate(r) if j != target_col]
        feats_f = []
        ok = True
        for c in feats:
            v = try_float(c)
            if v is None:
                ok = False
                break
            feats_f.append(v)
        if ok:
            X_raw.append(feats_f)
        else:
            # skip rows with non-numeric features
            continue

    if not X_raw:
        raise ValueError("Nenhuma linha com features numéricas válidas.")

    y_enc, classes = encode_labels(y_raw[:len(X_raw)])

    x_min, x_max = minmax_scale_fit(X_raw) if normalize else ([0.0]*len(X_raw[0]), [1.0]*len(X_raw[0]))
    X_use = minmax_scale_transform(X_raw, x_min, x_max) if normalize else X_raw
    return Dataset(X_use, y_enc, classes, x_min, x_max)

# -------------------------------
# Neural Network (MLP) from scratch
# -------------------------------

@dataclass
class MLPConfig:
    input_dim: int
    hidden_layers: List[int]
    output_dim: int
    lr: float = 0.05
    seed: int = 42
    l2: float = 0.0

class MLP:
    def __init__(self, cfg: MLPConfig):
        random.seed(cfg.seed)
        self.cfg = cfg
        layer_dims = [cfg.input_dim] + cfg.hidden_layers + [cfg.output_dim]
        self.W: List[List[List[float]]] = []
        self.b: List[List[float]] = []
        for i in range(len(layer_dims)-1):
            self.W.append(rand_matrix(layer_dims[i+1], layer_dims[i], scale=math.sqrt(2.0 / layer_dims[i])))
            self.b.append(rand_vector(layer_dims[i+1], scale=0.0))

    def forward(self, x: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        # returns (zs, activations) per layer; activations includes input as first
        a = x
        activations = [a]
        zs = []
        for li in range(len(self.W)):
            z = vecadd(matvec(self.W[li], a), self.b[li])
            zs.append(z)
            if li < len(self.W) - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)
        return zs, activations

    def backward(self, x: List[float], y_onehot: List[float]) -> Tuple[List[List[List[float]]], List[List[float]]]:
        zs, activations = self.forward(x)
        grads_W = [zeros((len(self.W[i]), len(self.W[i][0]))) for i in range(len(self.W))]
        grads_b = [zeros(len(self.b[i])) for i in range(len(self.b))]

        # output layer gradient: softmax + CE => delta = y_hat - y
        y_hat = activations[-1]
        delta = vecsub(y_hat, y_onehot)

        # grad for last layer
        a_prev = activations[-2]
        grads_W[-1] = outer(delta, a_prev)
        grads_b[-1] = delta[:]

        # propagate backwards through hidden layers
        for li in range(len(self.W)-2, -1, -1):
            W_next_T = transpose(self.W[li+1])  # shape: (n_curr, n_next)
            delta_next = grads_b[li+1]          # length = n_next

            # delta_k = sum_j W[k][j] * delta_next_j   (k = unidade da camada atual)
            delta = [
                sum(W_next_T[k][j] * delta_next[j] for j in range(len(delta_next)))
                for k in range(len(W_next_T))
            ]

            # aplica ReLU' no z da camada atual
            relud = relu_deriv(zs[li])
            delta = [delta[k] * relud[k] for k in range(len(relud))]

            a_prev = activations[li]
            grads_W[li] = outer(delta, a_prev)
            grads_b[li] = delta[:]

        return grads_W, grads_b

    def apply_grads(self, gW, gB, batch_size: int):
        # SGD update with optional L2
        for li in range(len(self.W)):
            for r in range(len(self.W[li])):
                for c in range(len(self.W[li][0])):
                    grad = gW[li][r][c] / max(1, batch_size) + self.cfg.l2 * self.W[li][r][c]
                    self.W[li][r][c] -= self.cfg.lr * grad
            for r in range(len(self.b[li])):
                gradb = gB[li][r] / max(1, batch_size)
                self.b[li][r] -= self.cfg.lr * gradb

    def predict_proba(self, x: List[float]) -> List[float]:
        _, acts = self.forward(x)
        return acts[-1]

    def predict(self, x: List[float]) -> int:
        probs = self.predict_proba(x)
        return max(range(len(probs)), key=lambda i: probs[i])

# -------------------------------
# Training helpers
# -------------------------------

def one_hot(y: int, K: int) -> List[float]:
    v = [0.0]*K
    v[y] = 1.0
    return v

def accuracy(model: MLP, X: List[List[float]], y: List[int]) -> float:
    if not X:
        return 0.0
    correct = 0
    for xi, yi in zip(X, y):
        pred = model.predict(xi)
        if pred == yi:
            correct += 1
    return correct / len(X)

def train(model: MLP, X: List[List[float]], y: List[int], epochs: int = 50, batch_size: int = 16, log=None):
    N = len(X)
    K = len(set(y))
    idxs = list(range(N))
    for ep in range(1, epochs+1):
        random.shuffle(idxs)
        gW_acc = [zeros((len(model.W[i]), len(model.W[i][0]))) for i in range(len(model.W))]
        gB_acc = [zeros(len(model.b[i])) for i in range(len(model.b))]
        batch_count = 0
        for i, idx in enumerate(idxs):
            xi = X[idx]
            yi = y[idx]
            yi_oh = one_hot(yi, K)
            gW, gB = model.backward(xi, yi_oh)
            # accumulate
            for li in range(len(gW_acc)):
                for r in range(len(gW_acc[li])):
                    for c in range(len(gW_acc[li][0])):
                        gW_acc[li][r][c] += gW[li][r][c]
                for r in range(len(gB_acc[li])):
                    gB_acc[li][r] += gB[li][r]
            batch_count += 1
            if batch_count == batch_size or i == N-1:
                model.apply_grads(gW_acc, gB_acc, batch_count)
                # reset accumulators
                gW_acc = [zeros((len(model.W[i]), len(model.W[i][0]))) for i in range(len(model.W))]
                gB_acc = [zeros(len(model.b[i])) for i in range(len(model.b))]
                batch_count = 0
        if log:
            log(f"Época {ep}/{epochs} concluída.")

# -------------------------------
# Saving / Loading model
# -------------------------------

def save_model(path: str, model: MLP, meta: Dict[str, Any]):
    payload = {
        "W": model.W,
        "b": model.b,
        "cfg": {
            "input_dim": model.cfg.input_dim,
            "hidden_layers": model.cfg.hidden_layers,
            "output_dim": model.cfg.output_dim,
            "lr": model.cfg.lr,
            "seed": model.cfg.seed,
            "l2": model.cfg.l2,
        },
        "meta": meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_model(path: str) -> Tuple[MLP, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = MLPConfig(
        input_dim=payload["cfg"]["input_dim"],
        hidden_layers=payload["cfg"]["hidden_layers"],
        output_dim=payload["cfg"]["output_dim"],
        lr=payload["cfg"]["lr"],
        seed=payload["cfg"]["seed"],
        l2=payload["cfg"]["l2"],
    )
    model = MLP(cfg)
    model.W = payload["W"]
    model.b = payload["b"]
    return model, payload.get("meta", {})

# -------------------------------
# GUI (Tkinter)
# -------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rede Neural do Zero (CSV)")
        self.geometry("900x680")

        # State
        self.csv_path = tk.StringVar(value="heart.csv")
        self.target_col = tk.IntVar(value=-1)  # -1 means "last column"
        self.normalize = tk.BooleanVar(value=True)
        self.epochs = tk.IntVar(value=60)
        self.lr = tk.DoubleVar(value=0.05)
        self.batch = tk.IntVar(value=16)
        self.hidden = tk.StringVar(value="32,16")
        self.test_size_pct = tk.IntVar(value=20)
        self.status = tk.StringVar(value="Pronto.")
        self.model: Optional[MLP] = None
        self.data: Optional[Dataset] = None
        self.Xtr: List[List[float]] = []
        self.Xte: List[List[float]] = []
        self.ytr: List[int] = []
        self.yte: List[int] = []
        self.meta: Dict[str, Any] = {}

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        frm_top = ttk.LabelFrame(self, text="Dados")
        frm_top.pack(fill="x", **pad)

        ttk.Label(frm_top, text="CSV:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.csv_path, width=48).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(frm_top, text="Procurar...", command=self.browse_csv).grid(row=0, column=2, **pad)

        ttk.Label(frm_top, text="Coluna alvo (índice, -1=última):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm_top, textvariable=self.target_col, width=10).grid(row=1, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm_top, text="Normalizar (min-max)", variable=self.normalize).grid(row=1, column=2, sticky="w", **pad)

        ttk.Button(frm_top, text="Pré-visualizar", command=self.preview_csv).grid(row=2, column=0, **pad)
        ttk.Button(frm_top, text="Carregar", command=self.load_data).grid(row=2, column=1, **pad)

        frm_nn = ttk.LabelFrame(self, text="Treinamento")
        frm_nn.pack(fill="x", **pad)

        ttk.Label(frm_nn, text="Hidden layers (ex: 32,16):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm_nn, textvariable=self.hidden, width=18).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frm_nn, text="Épocas:").grid(row=0, column=2, sticky="e", **pad)
        ttk.Entry(frm_nn, textvariable=self.epochs, width=10).grid(row=0, column=3, **pad)

        ttk.Label(frm_nn, text="LR:").grid(row=0, column=4, sticky="e", **pad)
        ttk.Entry(frm_nn, textvariable=self.lr, width=10).grid(row=0, column=5, **pad)

        ttk.Label(frm_nn, text="Batch:").grid(row=0, column=6, sticky="e", **pad)
        ttk.Entry(frm_nn, textvariable=self.batch, width=10).grid(row=0, column=7, **pad)

        ttk.Label(frm_nn, text="Teste %:").grid(row=0, column=8, sticky="e", **pad)
        ttk.Entry(frm_nn, textvariable=self.test_size_pct, width=10).grid(row=0, column=9, **pad)

        ttk.Button(frm_nn, text="Treinar", command=self.train_model).grid(row=1, column=0, **pad)
        ttk.Button(frm_nn, text="Salvar modelo", command=self.save_model_ui).grid(row=1, column=1, **pad)
        ttk.Button(frm_nn, text="Carregar modelo", command=self.load_model_ui).grid(row=1, column=2, **pad)

        frm_pred = ttk.LabelFrame(self, text="Predição Pontual")
        # --- Visualização da Rede ---
        frm_viz = ttk.LabelFrame(self, text="Visualização da Rede (estrutura e pesos)")
        frm_viz.pack(fill="both", expand=False, **pad)
        self.canvas_net = tk.Canvas(frm_viz, width=860, height=360, bg="#0f0f0f", highlightthickness=0)
        self.canvas_net.pack(fill="x", expand=False, **pad)
    
        frm_pred.pack(fill="x", **pad)
        self.sample_entry = ttk.Entry(frm_pred, width=90)
        self.sample_entry.grid(row=0, column=0, columnspan=6, sticky="we", **pad)
        ttk.Label(frm_pred, text="Valores separados por vírgula (somente features, na mesma ordem do CSV).").grid(row=1, column=0, columnspan=6, sticky="w", **pad)
        ttk.Button(frm_pred, text="Prever", command=self.predict_one).grid(row=0, column=6, **pad)

        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(frm_log, height=16)
        self.txt.pack(fill="both", expand=True, **pad)

        frm_status = ttk.Frame(self)
        frm_status.pack(fill="x")
        ttk.Label(frm_status, textvariable=self.status).pack(side="left", padx=8, pady=4)

    
    def _current_architecture(self):
        # Try to infer the layer sizes
        if self.model is not None:
            # From trained model
            sizes = [self.model.cfg.input_dim] + self.model.cfg.hidden_layers + [self.model.cfg.output_dim]
            return sizes
        # Fallback: from loaded data + text field "hidden" + classes
        if self.data is not None:
            try:
                hidden_layers = [int(x.strip()) for x in self.hidden.get().split(",") if x.strip()]
            except Exception:
                hidden_layers = [32, 16]
            input_dim = len(self.data.X[0]) if self.data and self.data.X else 0
            output_dim = len(set(self.data.y)) if self.data and self.data.y else 1
            if input_dim and output_dim:
                return [input_dim] + (hidden_layers if hidden_layers else [32,16]) + [output_dim]
        return []

    def render_network(self, model=None):
        # Draw a simple MLP diagram with nodes and weighted edges
        if not hasattr(self, "canvas_net"):
            return
        cnv = self.canvas_net
        cnv.delete("all")
        arch = self._current_architecture()
        if not arch:
            cnv.create_text(430, 180, fill="#ddd", text="Carregue os dados e/ou treine o modelo para ver a rede.")
            return

        width = int(cnv["width"]); height = int(cnv["height"])
        left_pad, right_pad, top_pad, bottom_pad = 50, 50, 30, 30
        L = len(arch)
        max_nodes = max(arch)
        # positions per layer
        layer_x = [left_pad + i*( (width-left_pad-right_pad)/max(1,(L-1)) ) for i in range(L)]
        y_spacings = []
        node_pos = []  # list of ( (x,y) for each node in layer )
        for i, n in enumerate(arch):
            if n == 1:
                ys = [height/2]
            else:
                usable_h = height - top_pad - bottom_pad
                step = usable_h / (n-1)
                ys = [top_pad + j*step for j in range(n)]
            x = layer_x[i]
            node_pos.append([(x, y) for y in ys])

        # Background labels
        for li, n in enumerate(arch):
            label = "Entrada" if li == 0 else ("Saída" if li == L-1 else f"Oculta {li}")
            cnv.create_text(layer_x[li], 15, fill="#aaa", text=f"{label} ({n})")

        # Edge drawing — if model provided, color by weight sign and thickness by magnitude
        def edge_style(w, maxabs):
            if maxabs <= 0: maxabs = 1.0
            t = max(1, int(1 + 3*abs(w)/maxabs))  # thickness 1..4
            # colors: red para negativo, azul para positivo
            col = "#d9534f" if w < 0 else "#5bc0de"
            return col, t

        # Determine weights max abs per layer for scaling
        W = model.W if model else (self.model.W if self.model else None)
        if W is None:
            # draw neutral edges
            for li in range(L-1):
                for i,(x1,y1) in enumerate(node_pos[li]):
                    for j,(x2,y2) in enumerate(node_pos[li+1]):
                        cnv.create_line(x1+8,y1, x2-8,y2, fill="#555", width=1)
        else:
            for li in range(L-1):
                wmat = W[li]
                maxabs = max((abs(w) for row in wmat for w in row), default=1.0)
                for j,(x2,y2) in enumerate(node_pos[li+1]):
                    for i,(x1,y1) in enumerate(node_pos[li]):
                        w = wmat[j][i] if j < len(wmat) and i < len(wmat[0]) else 0.0
                        col, thick = edge_style(w, maxabs)
                        cnv.create_line(x1+8,y1, x2-8,y2, fill=col, width=thick)

        # Draw nodes
        r = 7
        for li in range(L):
            for (x,y) in node_pos[li]:
                cnv.create_oval(x-r, y-r, x+r, y+r, outline="#eee", fill="#222")

        # Legend
        cnv.create_rectangle(10, height-28, 240, height-10, outline="#777")
        cnv.create_text(20, height-19, anchor="w", fill="#bbb",
                        text="Arestas: azul=pesos positivos, vermelho=negativos; espessura ~ |peso|")

    def log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.status.set(msg)
        self.update_idletasks()

    def browse_csv(self):
        path = filedialog.askopenfilename(title="Escolher CSV", filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if path:
            self.csv_path.set(path)

    def preview_csv(self):
        try:
            rows = read_csv(self.csv_path.get())
            preview = "\n".join([", ".join(r[:10]) for r in rows[:10]])
            messagebox.showinfo("Prévia (primeiras linhas)", preview if preview else "(vazio)")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def load_data(self):
        try:
            path = self.csv_path.get()
            tcol = self.target_col.get()
            if tcol == -1:
                # assume last column
                # need to inspect first row to get width
                rows = read_csv(path)
                width = len(rows[0]) if rows else 0
                tcol = width - 1
                self.target_col.set(tcol)

            self.data = load_dataset(path, tcol, normalize=self.normalize.get())
            test_size = max(0.05, min(0.95, self.test_size_pct.get() / 100.0))
            Xtr, ytr, Xte, yte = train_test_split(self.data.X, self.data.y, test_size=test_size, seed=42)
            self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte

            self.meta = {
                "csv_path": path,
                "target_col": tcol,
                "normalize": self.normalize.get(),
                "classes": self.data.classes_,
                "x_min": self.data.x_min,
                "x_max": self.data.x_max,
            }
            self.log(f"Dados carregados. Features={len(self.data.X[0])}, Classes={len(self.data.classes_)}. Train={len(Xtr)}, Test={len(Xte)}"); self.render_network(None)
        except Exception as e:
            messagebox.showerror("Erro ao carregar dados", str(e))

    
    def _epoch_log(self, msg: str):
        self.log(msg)
        try:
            if msg.startswith("Época"):
                # update viz every 5 epochs (non-blocking)
                parts = msg.split()
                # Época N/M ...
                if len(parts) >= 2 and "/" in parts[1]:
                    num = int(parts[1].split("/")[0])
                    if num % 5 == 0 and self.model is not None:
                        self.render_network(self.model)
                        self.update_idletasks()
        except Exception:
            pass
    def train_model(self):
        if not self.Xtr:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro.")
            return
        try:
            hidden_layers = [int(x.strip()) for x in self.hidden.get().split(",") if x.strip()]
            cfg = MLPConfig(
                input_dim=len(self.Xtr[0]),
                hidden_layers=hidden_layers if hidden_layers else [32, 16],
                output_dim=len(set(self.ytr)),
                lr=self.lr.get(),
                seed=42,
                l2=0.0,
            )
            self.model = MLP(cfg)
            start = time.time()
            train(self.model, self.Xtr, self.ytr, epochs=self.epochs.get(), batch_size=self.batch.get(), log=self._epoch_log)
            dur = time.time() - start
            acc_tr = accuracy(self.model, self.Xtr, self.ytr)
            acc_te = accuracy(self.model, self.Xte, self.yte) if self.Xte else 0.0
            self.log(f"Treino concluído em {dur:.1f}s. Acurácia: train={acc_tr*100:.2f}% | test={acc_te*100:.2f}%"); self.render_network(self.model)
        except Exception as e:
            messagebox.showerror("Erro no treino", str(e))

    def predict_one(self):
        if not self.model or not self.data:
            messagebox.showwarning("Aviso", "Treine ou carregue um modelo primeiro.")
            return
        try:
            raw = self.sample_entry.get().strip()
            if not raw:
                messagebox.showwarning("Aviso", "Informe os valores das features separados por vírgula.")
                return
            vals = [float(x.strip().replace(",", ".")) for x in raw.split(",")]
            if len(vals) != self.model.cfg.input_dim:
                messagebox.showwarning("Aviso", f"Esperado {self.model.cfg.input_dim} valores, recebido {len(vals)}.")
                return
            # apply same scaling if needed
            if self.meta.get("normalize", True):
                mins = self.meta["x_min"]
                maxs = self.meta["x_max"]
                vals = [(vals[j] - mins[j]) / (maxs[j] - mins[j]) for j in range(len(vals))]
            probs = self.model.predict_proba(vals)
            pred = max(range(len(probs)), key=lambda i: probs[i])
            cls = self.meta.get("classes", [])
            label = cls[pred] if pred < len(cls) else str(pred)
            out = f"Predição: {label} | Probabilidades: " + ", ".join(f"{(cls[i] if i < len(cls) else i)}={probs[i]*100:.1f}%" for i in range(len(probs)))
            self.log(out)
        except Exception as e:
            messagebox.showerror("Erro na predição", str(e))

    def save_model_ui(self):
        if not self.model:
            messagebox.showwarning("Aviso", "Não há modelo para salvar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], title="Salvar modelo")
        if not path:
            return
        try:
            save_model(path, self.model, self.meta)
            self.log(f"Modelo salvo em: {path}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    def load_model_ui(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")], title="Carregar modelo")
        if not path:
            return
        try:
            model, meta = load_model(path)
            self.model = model
            self.meta = meta
            self.log(f"Modelo carregado. Input={model.cfg.input_dim}, Output={model.cfg.output_dim}."); self.render_network(self.model)
        except Exception as e:
            messagebox.showerror("Erro ao carregar", str(e))

def main():
    random.seed(42)
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
