# app.py
import streamlit as st

# ---------- state ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "budget" not in st.session_state:
    st.session_state.budget = 100
if "cards" not in st.session_state:
    st.session_state.cards = [
        {"idx": i, "color": ("blue" if i < 26 else "red"), "used": False, "signal": None}
        for i in range(52)
    ]
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

# ---------- login ----------
if st.session_state.user is None:
    st.title("Login")
    name = st.text_input("Player name")
    if st.button("Enter"):
        st.session_state.user = name.strip() or "player"
        st.rerun()
    st.stop()

# ---------- header ----------
st.title("Card Signals")
st.subheader(f"Budget: £{st.session_state.budget}")

# ---------- styles (card = the button). Scoped to wrapper to avoid affecting sidebar buttons ----------
st.markdown("""
<style>
.cardwrap{
  width:120px; height:170px; border:2px solid #ffffff; border-radius:12px;
  display:flex; align-items:center; justify-content:center; margin:auto;
}
.cardwrap.blue{ background:#2b6cb0; }   /* blue */
.cardwrap.red { background:#c53030; }   /* red  */
.cardwrap.used{ opacity:.35; filter:grayscale(30%); }
/* make the inner Streamlit button invisible and full-size so the wrapper provides the color */
.cardwrap div[data-testid^="baseButton"] > button,
.cardwrap .stButton button{
  width:100%; height:100%; background:transparent; border:none; box-shadow:none;
  font-size:0; color:transparent; line-height:0;
}
.cardwrap div[data-testid^="baseButton"] > button:focus,
.cardwrap .stButton button:focus{ outline:none; }
</style>
""", unsafe_allow_html=True)

# ---------- grid (cards are clickable buttons) ----------
cols_per_row = 5
rows = (len(st.session_state.cards) + cols_per_row - 1) // cols_per_row
for r in range(rows):
    cols = st.columns(cols_per_row, gap="small")
    for c in range(cols_per_row):
        i = r * cols_per_row + c
        if i >= len(st.session_state.cards):
            cols[c].empty()
            continue
        card = st.session_state.cards[i]
        colour_cls = "blue" if card["color"] == "blue" else "red"
        used_cls = " used" if card["used"] else ""
        cols[c].markdown(f'<div class="cardwrap {colour_cls}{used_cls}">', unsafe_allow_html=True)
        if cols[c].button(" ", key=f"card_{i}", disabled=card["used"]):
            st.session_state.selected_idx = i
            st.rerun()
        cols[c].markdown("</div>", unsafe_allow_html=True)

# ---------- sidebar: signal selection and budget deduction ----------
sid = st.sidebar
sid.header("Signal picker")
if st.session_state.selected_idx is None:
    sid.info("Click a card.")
else:
    i = st.session_state.selected_idx
    card = st.session_state.cards[i]
    sid.write(f"Card #{i+1} • {card['color']}")
    sig = sid.radio("Signal", [1, 2], horizontal=True, label_visibility="collapsed")
    a, b = sid.columns(2)
    if a.button("Confirm"):
        if st.session_state.budget < 5:
            sid.error("Insufficient budget.")
        else:
            st.session_state.cards[i]["signal"] = int(sig)
            st.session_state.cards[i]["used"] = True
            st.session_state.budget -= 5
            st.session_state.selected_idx = None
            st.rerun()
    if b.button("Cancel"):
        st.session_state.selected_idx = None
        st.rerun()
