"""
main_fs_patch.py
----------------
This file documents the FOUR edits required in main_fs.py to integrate
RNN, GRU, and CNN models. Apply them in order.

HOW TO APPLY
------------
Search for the exact string shown under FIND and replace it with the
string shown under REPLACE. Each edit is self-contained.
"""

# ══════════════════════════════════════════════════════════════════════════════
# EDIT 1 — Register RNN/GRU/CNN as univariate (subset-independent) models
# Location: ~line 87
# ══════════════════════════════════════════════════════════════════════════════

EDIT_1_FIND = '''UNIVARIATE_MODELS = {"arima", "ets", "naive", "lstm"}'''

EDIT_1_REPLACE = '''UNIVARIATE_MODELS = {"arima", "ets", "naive", "lstm", "rnn", "gru", "cnn"}'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 2 — Add runner functions for RNN, GRU, CNN
# Location: immediately after the _run_lstm function (~line 164)
# ══════════════════════════════════════════════════════════════════════════════

EDIT_2_FIND = '''def _run_arima(raw_train, raw_test):'''

EDIT_2_REPLACE = '''def _run_rnn(raw_train, raw_test, raw_split, raw):
    from models.rnn_model import RNNForecaster
    m = RNNForecaster(lookback=48, horizon=24, epochs=30, patience=5)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - 48): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "RNN", train_time_s=m.train_time_)


def _run_gru(raw_train, raw_test, raw_split, raw):
    from models.gru_model import GRUForecaster
    m = GRUForecaster(lookback=48, horizon=24, epochs=30, patience=5)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - 48): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "GRU", train_time_s=m.train_time_)


def _run_cnn(raw_train, raw_test, raw_split, raw):
    from models.cnn_model import CNNForecaster
    m = CNNForecaster(lookback=48, horizon=24, epochs=30, patience=5)
    m.fit(raw_train)
    preds = np.array([
        m._predict_one(raw[max(0, raw_split + i - 48): raw_split + i])
        for i in range(len(raw_test))
    ])
    return evaluate(raw_test, preds, "CNN", train_time_s=m.train_time_)


def _run_arima(raw_train, raw_test):'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 3 — Add RNN/GRU/CNN to the model dispatcher in run_model_on_subset
# Location: inside run_model_on_subset, after the lstm branch (~line 503)
# ══════════════════════════════════════════════════════════════════════════════

EDIT_3_FIND = '''        elif model_key == "arima": return _run_arima(raw_train, raw_test)'''

EDIT_3_REPLACE = '''        elif model_key == "arima": return _run_arima(raw_train, raw_test)
        elif model_key == "rnn":   return _run_rnn(raw_train, raw_test,
                                                    raw_split, raw)
        elif model_key == "gru":   return _run_gru(raw_train, raw_test,
                                                    raw_split, raw)
        elif model_key == "cnn":   return _run_cnn(raw_train, raw_test,
                                                    raw_split, raw)'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 4 — Register rnn/gru/cnn as valid model keys in the CLI validator
# Location: inside main(), ~line 954
# ══════════════════════════════════════════════════════════════════════════════

EDIT_4_FIND = '''    valid_keys = {"xgb","lgb","rf","lstm","arima",
                  "gam","ridge","lasso","enet","svr","mlp","ets","naive"}'''

EDIT_4_REPLACE = '''    valid_keys = {"xgb","lgb","rf","lstm","arima",
                  "gam","ridge","lasso","enet","svr","mlp","ets","naive",
                  "rnn","gru","cnn"}'''


# ══════════════════════════════════════════════════════════════════════════════
# EDIT 5 — (Optional) Update the docstring model list at the top of the file
# Location: ~line 24, inside the module docstring
# ══════════════════════════════════════════════════════════════════════════════

EDIT_5_FIND = '''  naive    — Naïve baseline (lag-1: ŷ_t = y_{t-1})'''

EDIT_5_REPLACE = '''  naive    — Naïve baseline (lag-1: ŷ_t = y_{t-1})
  rnn      — Vanilla RNN (SimpleRNN, lookback=48h)
  gru      — Gated Recurrent Unit (lookback=48h)
  cnn      — 1D CNN with dilated causal convolutions (lookback=48h)'''


# ══════════════════════════════════════════════════════════════════════════════
# RUN COMMAND (after all edits applied)
# ══════════════════════════════════════════════════════════════════════════════
#
#   python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \
#                     --models rnn gru cnn
#
# --skip-fs is safe: RNN/GRU/CNN are univariate — they never consume
# the feature subsets, so the saved fs_summary.csv is not involved.
#
# To run all deep learning models together:
#   python main_fs.py --csv GEFCom2014_clean.csv --skip-fs \
#                     --models lstm rnn gru cnn
#
# ══════════════════════════════════════════════════════════════════════════════
