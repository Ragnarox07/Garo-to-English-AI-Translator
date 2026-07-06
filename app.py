 #!/usr/bin/env python3

import os
import re
import json
import math
import datetime
import csv
from typing import List, Dict

import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# =========================================================
# ✏   CONFIG — edit these three lines to connect your data
# =========================================================

MODEL_DIR = "saved_garo_translator"

# ── OPTION A: local file (default) ━━━━━━━━━━━━━━━━━━━━━━
# MAIN_DATA = "test1.csv"

# ── OPTION B: public Google Sheet ━━━━━━━━━━━━━━━━━━━━━━
# 1. Open your Google Sheet → File → Share → "Anyone with the link can view"
# 2. Copy the Sheet ID from the URL:
#    https://docs.google.com/spreadsheets/d/  <<<SHEET_ID>>>  /edit
# 3. Copy the gid (grid ID) from the URL (e.g., if it's the first sheet, it's usually 0).
# 4. Paste the ID and gid below and uncomment the two lines:
#
# GSHEET_ID   = "YOUR_SHEET_ID_HERE"
MAIN_DATA   = f"https://docs.google.com/spreadsheets/d/1k-ucejVa3xuhSZfFU6ZaSael0DxzkPOW7X_Zsq7ekWM/export?format=csv&gid=0" # Verify sharing settings and gid

# ── CONTRIBUTION STORAGE ━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTION A: save to a local CSV (default)
CONTRIBUTION_FILE = "https://docs.google.com/spreadsheets/d/1rays7kXL-SrAWbB4o2YlInAyNmRH_xtYkvpTAM9488Q/edit?usp=sharing"

# OPTION B: write to a private Google Sheet via gspread
# 1. Create a Google Cloud service account (console.cloud.google.com)
#    → Enable "Google Sheets API"
#    → Create service account → download key as "service_account.json"
# 2. Share your Contributions Google Sheet with the service account email.
# 3. Set USE_GSPREAD_CONTRIBUTE = True and fill in the sheet name below.
USE_GSPREAD_CONTRIBUTE = True
GSPREAD_CREDENTIALS_FILE = "service_account.json"   # path to your key file
CONTRIBUTION_SHEET_NAME = "Contributions"            # tab name in the sheet

# ── FEEDBACK STORAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEEDBACK_FILE = "https://docs.google.com/spreadsheets/d/1oG5als56WOnrQWi298z4NSrEcYmR5FG47aZvL7nQFrE/edit?usp=sharing"
USE_GSPREAD_FEEDBACK = True          # set True + fill in name to use Sheets
FEEDBACK_SHEET_NAME = "Feedback"

# =========================================================
# MODEL PARAMS (must match what you trained with)
# =========================================================
EMBED_SIZE = 128
NHEAD = 4
NUM_LAYERS = 2
FFN_HID = 256
DROPOUT = 0.1
MAX_LEN = 50
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)

# =========================================================
# TEXT UTILS
# =========================================================

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z]+(?:[.'\u00b7][a-zA-Z]+)?|[0-9]+|[?.!,]", text)


def encode(text: str, vocab: Dict[str, int], max_len: int = MAX_LEN) -> List[int]:
    ids = [vocab["<bos>"]]
    ids += [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]
    ids.append(vocab["<eos>"])
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = vocab["<eos>"]
    return ids


def decode(ids: List[int], inv_vocab: Dict[int, str]) -> str:
    words = []
    for idx in ids:
        tok = inv_vocab.get(int(idx), "<unk>")
        if tok == "<eos>":
            break
        if tok not in ["<pad>", "<bos>", "<unk>"]:
            words.append(tok)
    return " ".join(words)

# =========================================================
# MODEL DEFINITION
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GaroTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, EMBED_SIZE)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, EMBED_SIZE)
        self.pos = PositionalEncoding(EMBED_SIZE)
        self.transformer = nn.Transformer(
            d_model=EMBED_SIZE, nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FFN_HID, dropout=DROPOUT, batch_first=True
        )
        self.out = nn.Linear(EMBED_SIZE, tgt_vocab_size)

    def forward(self, src, tgt):
        src = self.pos(self.src_emb(src) * math.sqrt(EMBED_SIZE))
        tgt = self.pos(self.tgt_emb(tgt) * math.sqrt(EMBED_SIZE))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)
        result = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.out(result)

# =========================================================
# LOAD MODEL (once at startup)
# =========================================================

print("Loading Garo Transformer model…")

with open(os.path.join(MODEL_DIR, "src_vocab.json"), "r", encoding="utf-8") as f:
    src_vocab = json.load(f)
with open(os.path.join(MODEL_DIR, "tgt_vocab.json"), "r", encoding="utf-8") as f:
    tgt_vocab = json.load(f)
with open(os.path.join(MODEL_DIR, "inv_tgt_vocab.json"), "r", encoding="utf-8") as f:
    inv_tgt_vocab = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(MODEL_DIR, "exact_memory.json"), "r", encoding="utf-8") as f:
    exact_memory = json.load(f)
with open(os.path.join(MODEL_DIR, "alternatives.json"), "r", encoding="utf-8") as f:
    alternatives = json.load(f)

model = GaroTransformer(len(src_vocab), len(tgt_vocab)).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE))
model.eval()

print(f"Model loaded on {DEVICE}. Exact memory: {len(exact_memory)} phrases.")

# =========================================================
# TRANSLATION LOGIC
# =========================================================

def ai_translate(text):
    src = torch.tensor([encode(text, src_vocab)]).to(DEVICE)
    output = [tgt_vocab["<bos>"]]
    for _ in range(MAX_LEN):
        tgt = torch.tensor([output]).to(DEVICE)
        with torch.no_grad():
            logits = model(src, tgt)
        next_token = logits[:, -1, :].argmax(-1).item()
        if next_token == tgt_vocab["<eos>"]:
            break
        output.append(next_token)
    return decode(output, inv_tgt_vocab)


def safe_ai_translate(text):
    try:
        result = ai_translate(text)
    except Exception as e:
        print("AI fallback error:", e)
        return None

    bad = {"anga", "dada", "na.a", "mi"}

    if not result.strip():
        return None

    if result.strip() in bad and len(text.split()) > 1:
        return None

    if len(result.split()) < max(1, len(text.split()) // 2):
        return None

    return result


def name_pattern_translate(text):
    match = re.match(r"^my name is\s+(.+)$", text.strip(), flags=re.IGNORECASE)
    if match:
        return f"Angni biming {match.group(1).strip()}"
    return None


def smart_phrase_translate(text):
    tokens = re.split(r"\b(and|but|or)\b", text)
    final = []
    any_match = False
    for part in tokens:
        part = part.strip()
        if not part:
            continue
        if part in exact_memory:
            final.append(exact_memory[part])
            any_match = True
            continue
        words = part.split()
        i = 0
        while i < len(words):
            matched = False
            for size in range(min(6, len(words) - i), 0, -1):
                chunk = " ".join(words[i:i + size])
                if chunk in exact_memory:
                    final.append(exact_memory[chunk])
                    i += size
                    matched = True
                    any_match = True
                    break
            if not matched:
                final.append(words[i])
                i += 1
    return " ".join(final), any_match


def translate(text: str) -> dict:
    original = text.strip()

    # 1. Name pattern
    name_result = name_pattern_translate(original)
    if name_result:
        return {"result": name_result, "match_type": "name pattern"}

    normalized = normalize_text(original)

    # 2. Exact memory match
    if normalized in exact_memory:
        return {"result": exact_memory[normalized], "match_type": "exact match"}

    # 3. Smart phrase / chunk match
    phrase_result, any_match = smart_phrase_translate(normalized)
    if any_match:
        return {"result": phrase_result, "match_type": "phrase match"}

    # 4. AI transformer fallback
    ai_result = safe_ai_translate(normalized)
    if ai_result:
        return {"result": ai_result, "match_type": "AI translation"}

    return {
        "result": "",
        "match_type": "no match",
        "message": "Not in the phrasebook yet. Please use Contribute to add this translation!"
    }

# =========================================================
# GSPREAD HELPERS (only used if USE_GSPREAD_* = True)
# =========================================================

def _get_gspread_sheet(sheet_name: str):
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(GSPREAD_CREDENTIALS_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open(sheet_name).sheet1


def append_to_sheet(sheet_name: str, row: list):
    sheet = _get_gspread_sheet(sheet_name)
    sheet.append_row(row)

# =========================================================
# CONTRIBUTION + FEEDBACK WRITERS
# =========================================================

def save_contribution(source: str, target: str):
    source = normalize_text(source)
    target = normalize_text(target)
    timestamp = datetime.datetime.utcnow().isoformat()
    row = [source, target, "no", timestamp]

    if USE_GSPREAD_CONTRIBUTE:
        append_to_sheet(CONTRIBUTION_SHEET_NAME, row)
    else:
        write_header = not os.path.exists(CONTRIBUTION_FILE)
        with open(CONTRIBUTION_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["source", "target", "verified", "timestamp"])
            w.writerow(row)


def save_feedback(rating: int, message: str):
    timestamp = datetime.datetime.utcnow().isoformat()
    row = [rating, message, timestamp]

    if USE_GSPREAD_FEEDBACK:
        append_to_sheet(FEEDBACK_SHEET_NAME, row)
    else:
        write_header = not os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["rating", "message", "timestamp"])
            w.writerow(row)


print(translate("sometime"))

# =========================================================
# FLASK HOSTING SECTION (Appended at Bottom)
# =========================================================
import threading
import time
import urllib.request

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "garo-translator.html")

@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = translate(text)
    return jsonify(result)

def run_flask():
    # Using port 5006 to ensure it's free
    app.run(host='0.0.0.0', port=5006, debug=False, use_reloader=False)

# Start Flask in background thread
threading.Thread(target=run_flask, daemon=True).start()

# Fetch Public IP and Start Tunnel
time.sleep(2)
try:
    endpoint_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
    print(f"\nEndpoint IP (Tunnel Password): {endpoint_ip}")
except:
    print("\nCould not fetch external IP.")

# Execute localtunnel shell command
!npx localtunnel --port 5006