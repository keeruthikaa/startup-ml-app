import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="StartupLens India", page_icon="🔭",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
# CSS  — fixes ALL colour issues including funding input
# ══════════════════════════════════════════════════════════════════
st.markdown("""<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;}

/* ─ labels (bold white) ─ */
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] label,
label{color:#F0F4FF!important;font-size:1rem!important;
      font-weight:800!important;letter-spacing:.03em;}

/* ─ selectbox text ─ */
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div{color:#FFFFFF!important;font-weight:700!important;}

/* ─ number input  ← THE KEY FIX ─ */
div[data-testid="stNumberInput"] input,
input[type="number"]{
    color:#FFFFFF!important;background:#1A2040!important;
    font-weight:800!important;font-size:1.1rem!important;
    border:2px solid #4F63D2!important;border-radius:8px!important;
    padding:8px 12px!important;}

/* ─ metrics ─ */
div[data-testid="stMetricLabel"] p{color:#A0B0FF!important;font-weight:700!important;}
div[data-testid="stMetricValue"] div{color:#7DF9C8!important;font-weight:800!important;font-size:1.8rem!important;}

/* ─ sidebar ─ */
section[data-testid="stSidebar"]{background:#060D26!important;}
section[data-testid="stSidebar"] *{color:#E0E8FF!important;}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input{
    background:#0D1640!important;border:2px solid #4F63D2!important;
    color:#FFFFFF!important;font-weight:800!important;}

/* ─ buttons ─ */
div.stButton>button[kind="primary"]{
    font-size:1.1rem!important;font-weight:800!important;
    background:linear-gradient(135deg,#4F63D2,#7C3AED)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    padding:.7rem 1.5rem!important;letter-spacing:.04em;}

/* ─ tabs ─ */
button[data-baseweb="tab"]{font-weight:700!important;font-size:.95rem!important;}

/* ─ general text ─ */
p,li,.stMarkdown p{color:#D0DBFF!important;}
h1,h2,h3{color:#FFFFFF!important;}
div[data-testid="stAlert"] p{font-weight:600!important;}
details summary p{font-weight:700!important;color:#A0B0FF!important;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
MODEL_DIR = "models"
TIER1 = {"Bangalore","Bengaluru","Mumbai","Delhi","New Delhi","Chennai","Hyderabad",
         "Kolkata","Gurgaon","Gurugram","Noida","Andheri","Chembur","Kormangala",
         "Bengaluru and Gurugram","Mumbai/Bengaluru"}
TIER2 = {"Pune","Ahmedabad","Jaipur","Chandigarh","Indore","Lucknow","Nagpur","Kochi",
         "Coimbatore","Bhopal","Vadodara","Surat","Trivandrum","Bhubneswar",
         "Goa","Panaji","Faridabad","Haryana","Karnataka","Kerala","Taramani"}
TIER3 = {"Amritsar","Gwalior","Varanasi","Kanpur","Rourkela","Jodhpur","Udaipur",
         "Gaya","Udupi","Belgaum","Missourie","Burnsville","Tulangan","Nairobi"}
HIGH_FUND_INDUSTRIES = {"FinTech","E-commerce","E-Commerce","Finance",
                         "Consumer Internet","Technology","Logistics"}
PALETTE = ["#4F63D2","#7C3AED","#06B6D4","#10B981","#F59E0B","#EF4444","#EC4899","#8B5CF6"]

def get_tier(city):
    if city in TIER1: return "Tier 1"
    if city in TIER2: return "Tier 2"
    if city in TIER3: return "Tier 3"
    if any(x in city for x in ["USA","US","SFO","California","Boston",
                                "New York","Singapore","Palo Alto"]): return "International"
    return "Other"

def tier_num(city):
    return {"Tier 1":1,"Tier 2":2,"Tier 3":3,"International":0,"Other":4}[get_tier(city)]

def tier_color(tier):
    return {"Tier 1":"#4F63D2","Tier 2":"#06B6D4","Tier 3":"#10B981",
            "International":"#F59E0B","Other":"#6B7280"}.get(tier,"#6B7280")

def fmt_inr(v):
    if v>=1e7:  return f"₹{v/1e7:.1f}Cr"
    if v>=1e5:  return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔬 Loading AI models…")
def load_models():
    files = ["models.pkl","ridge_reg.pkl","feature_cols.pkl","raw_stats.pkl",
             "city_avg_fund.pkl","ind_avg_fund.pkl","city_counts.pkl",
             "ind_counts.pkl","city_success_rt.pkl"]
    if not all(os.path.exists(os.path.join(MODEL_DIR,f)) for f in files):
        st.error("❌ Run `python train.py` first to generate model files.")
        st.stop()
    loaded = {}
    for f in files:
        with open(os.path.join(MODEL_DIR,f),"rb") as fh:
            loaded[f.replace(".pkl","")] = pickle.load(fh)
    return loaded

data = load_models()
models_dict   = data["models"]           # dict of 5 classifiers
ridge_reg     = data["ridge_reg"]
FEATURE_COLS  = data["feature_cols"]     # 12 feature names (excl. funding_amount)
raw_stats     = data["raw_stats"]
city_avg_fund = data["city_avg_fund"]
ind_avg_fund  = data["ind_avg_fund"]
city_counts   = data["city_counts"]
ind_counts    = data["ind_counts"]
city_success_rt = data["city_success_rt"]

# dropdown lists from stats
cities     = sorted(raw_stats["city_df"]["city_name"].tolist())
industries = sorted(raw_stats["ind_df"]["industry_name"].tolist())

# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDER  (13 engineered predictors)
# ══════════════════════════════════════════════════════════════════
def build_features(city, industry, amount, success_for_reg=0):
    c_avg  = city_avg_fund.get(city, np.mean(list(city_avg_fund.values())))
    i_avg  = ind_avg_fund.get(industry, np.mean(list(ind_avg_fund.values())))
    c_cnt  = city_counts.get(city, 1)
    i_cnt  = ind_counts.get(industry, 1)
    c_sr   = city_success_rt.get(city, 0.015)

    row = {
        "log_funding":          np.log1p(amount),
        "funding_bucket":       float(pd.cut([amount],
                                    bins=[0,500_000,2_000_000,10_000_000,
                                          50_000_000,500_000_000,float("inf")],
                                    labels=[0,1,2,3,4,5])[0]),
        "city_tier":            tier_num(city),
        "is_tier1_city":        int(city in TIER1),
        "is_international":     int(any(x in city for x in ["USA","US","SFO","California",
                                        "Boston","New York","Singapore","Palo Alto"])),
        "city_startup_density": c_cnt,
        "city_avg_funding":     c_avg,
        "city_success_rate":    c_sr,
        "is_hot_industry":      int(industry in HIGH_FUND_INDUSTRIES),
        "industry_density":     i_cnt,
        "industry_avg_funding": i_avg,
        "fund_vs_industry_avg": amount / (i_avg + 1),
    }
    X_clf = pd.DataFrame([row])[FEATURE_COLS]

    row_reg = dict(row)
    row_reg["Success"] = success_for_reg
    X_reg = pd.DataFrame([row_reg])[FEATURE_COLS + ["Success"]]
    return X_clf, X_reg, c_avg, i_avg, c_sr

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔭 StartupLens India")
    st.markdown("*AI-Powered Startup Intelligence*")
    st.divider()

    st.markdown("### ⚙️ Your Startup")
    sel_city = st.selectbox("📍 City / Location", cities,
                            index=cities.index("Mumbai") if "Mumbai" in cities else 0)
    sel_ind  = st.selectbox("🏭 Industry Vertical", industries,
                            index=industries.index("E-commerce") if "E-commerce" in industries else 0)
    sel_amt  = st.number_input("💰 Funding Amount (INR)",
                               min_value=0.0, value=5_000_000.0,
                               step=500_000.0, format="%.0f")

    tier = get_tier(sel_city)
    badge = {"Tier 1":"🔵","Tier 2":"🟢","Tier 3":"🟡","International":"🌐","Other":"⚪"}
    st.markdown(f"**City Tier:** {badge[tier]} `{tier}`")
    st.divider()

    run = st.button("🔮 Run Full Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("5 Models · 13 Features · 2,071 Startups")

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#060D26,#1a1f4e);
     border-radius:14px;padding:28px 36px;margin-bottom:18px;
     border:1px solid #2a3060;'>
  <h1 style='color:#fff;margin:0;font-size:2.2rem;'>🔭 StartupLens India</h1>
  <p style='color:#A0B0FF;font-size:1.05rem;margin:8px 0 0;'>
     AI-Powered Startup Success Intelligence · 5 Models · 13 Engineered Predictors
  </p>
</div>""", unsafe_allow_html=True)

# KPIs
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("📊 Startups",       f"{raw_stats['total']:,}")
k2.metric("✅ Success Rate",    f"{raw_stats['overall_success']*100:.1f}%")
k3.metric("💰 Avg Funding",    fmt_inr(raw_stats['avg_fund_overall']))
k4.metric("🏙️ Cities",         str(len(raw_stats['city_df'])))
k5.metric("🏭 Industries",     str(len(raw_stats['ind_df'])))
st.divider()

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔮 Prediction Engine",
    "📐 Feature Explorer",
    "🏙️ City & Tier Analysis",
    "🏭 Industry Deep Dive",
    "📊 Market Overview",
])

# ─────────────────────────────────────────────────────────────────
# TAB 1  PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🎯 Multi-Model Prediction Engine")
    if not run:
        st.info("👈 Fill details in the sidebar and click **Run Full Analysis**.")
    else:
        X_clf, X_reg_dummy, c_avg, i_avg, c_sr = build_features(sel_city, sel_ind, sel_amt)

        # ── Run all 5 classifiers ──
        results = {}
        for name, mdl in models_dict.items():
            prob = float(mdl.predict_proba(X_clf)[0][1])
            pred = int(mdl.predict(X_clf)[0])
            results[name] = {"prob": prob, "pred": pred}

        ensemble = np.mean([v["prob"] for v in results.values()])
        ensemble_pred = int(ensemble >= 0.5)
        _, X_reg, _, _, _ = build_features(sel_city, sel_ind, sel_amt,
                                           success_for_reg=ensemble_pred)
        pred_amt = max(float(ridge_reg.predict(X_reg)[0]), 0)

        tier_row = raw_stats["tier_df"][raw_stats["tier_df"]["tier"]==tier]
        t_sr   = float(tier_row["avg_success_rate"].values[0]) if len(tier_row) else 0.015
        t_fund = float(tier_row["Amount in USD"].values[0]) if len(tier_row) else c_avg

        # verdict
        if ensemble>=0.60:   vc,vi,vt="#10B981","🟢","HIGH POTENTIAL"
        elif ensemble>=0.35: vc,vi,vt="#F59E0B","🟡","MODERATE POTENTIAL"
        else:                vc,vi,vt="#EF4444","🔴","NEEDS IMPROVEMENT"

        # ── Headline ──
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#060D26,#111c3a);
             border:2px solid {vc};border-radius:12px;padding:22px 28px;margin-bottom:18px;'>
          <div style='display:flex;align-items:center;gap:16px;'>
            <div style='font-size:3rem'>{vi}</div>
            <div>
              <div style='color:{vc};font-size:1.6rem;font-weight:800'>{vt}</div>
              <div style='color:#A0B0FF;font-size:.95rem;margin-top:4px'>
                Ensemble Score: <b style='color:#fff'>{ensemble*100:.1f}%</b>
                &nbsp;|&nbsp; Tier: <b style='color:#fff'>{tier}</b>
                &nbsp;|&nbsp; Predicted Funding: <b style='color:#7DF9C8'>{fmt_inr(pred_amt)}</b>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── 5 Model metric cards ──
        st.markdown("#### 📊 All 5 Model Predictions")
        cols5 = st.columns(5)
        model_colors = ["#4F63D2","#7C3AED","#06B6D4","#10B981","#F59E0B"]
        for i,(name,res) in enumerate(results.items()):
            icon = "✅" if res["pred"]==1 else "⚠️"
            cols5[i].metric(f"{icon} {name}", f"{res['prob']*100:.1f}%",
                            "Success" if res["pred"]==1 else "Fail")

        st.divider()

        # ── Gauge + Model bar ──
        g1,g2 = st.columns(2)

        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ensemble*100,
                delta={"reference":t_sr*100,"suffix":"%",
                       "increasing":{"color":"#10B981"},"decreasing":{"color":"#EF4444"}},
                title={"text":"Ensemble Score","font":{"color":"#A0B0FF","size":15}},
                number={"suffix":"%","font":{"color":"#fff","size":40}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#4F63D2"},
                       "bar":{"color":vc,"thickness":0.28},
                       "bgcolor":"#1a2040","bordercolor":"#2a3060",
                       "steps":[{"range":[0,35],"color":"#2d1515"},
                                {"range":[35,60],"color":"#2d2a10"},
                                {"range":[60,100],"color":"#102d1f"}],
                       "threshold":{"line":{"color":"#fff","width":3},"value":t_sr*100}},
            ))
            fig_g.update_layout(paper_bgcolor="#060D26",plot_bgcolor="#060D26",
                                font_color="#fff",height=300,margin=dict(t=60,b=10,l=30,r=30))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"White line = {tier} benchmark ({t_sr*100:.1f}%)")

        with g2:
            names  = list(results.keys())
            probs  = [results[n]["prob"]*100 for n in names]
            colors = [model_colors[i] for i in range(len(names))]
            fig_mb = go.Figure(go.Bar(
                x=names, y=probs,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
                textfont=dict(color="#fff",size=12),
            ))
            fig_mb.add_hline(y=t_sr*100, line_dash="dash", line_color="#fff",
                             annotation_text=f"{tier} avg",
                             annotation_font_color="#fff")
            fig_mb.update_layout(
                title=dict(text="🧠 Model-by-Model Comparison",font=dict(color="#A0B0FF",size=14)),
                paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=300,
                yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",title="%",range=[0,110]),
                xaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20))
            st.plotly_chart(fig_mb, use_container_width=True)

        # ── Radar ──
        st.markdown("#### 🕸️ Multi-Dimensional Profile vs. Tier Benchmark")
        cats = ["LR Score","RF Score","GB Score","SVM Score","KNN Score"]
        your_vals = [results["Logistic Regression"]["prob"],
                     results["Random Forest"]["prob"],
                     results["Gradient Boosting"]["prob"],
                     results["SVM"]["prob"],
                     results["KNN"]["prob"]]
        bench_vals = [t_sr]*5

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=your_vals+[your_vals[0]],
                                         theta=cats+[cats[0]],fill="toself",
                                         name="Your Startup",line_color="#4F63D2",
                                         fillcolor="rgba(79,99,210,0.3)"))
        fig_r.add_trace(go.Scatterpolar(r=bench_vals+[bench_vals[0]],
                                         theta=cats+[cats[0]],fill="toself",
                                         name=f"{tier} Benchmark",line_color="#06B6D4",
                                         fillcolor="rgba(6,182,212,0.15)"))
        fig_r.update_layout(
            polar=dict(bgcolor="#060D26",
                       radialaxis=dict(range=[0,1],gridcolor="#2a3060",color="#A0B0FF"),
                       angularaxis=dict(color="#D0DBFF")),
            paper_bgcolor="#060D26",font_color="#fff",height=380,
            legend=dict(bgcolor="#060D26"),margin=dict(t=30,b=30))
        st.plotly_chart(fig_r, use_container_width=True)

        # ── Funding comparison ──
        st.markdown("#### 💰 Funding Context")
        fc1,fc2 = st.columns(2)
        with fc1:
            fl = ["Your Input","Model Prediction",f"{tier} Avg","Industry Avg","City Avg"]
            fv = [sel_amt, pred_amt, t_fund, i_avg, c_avg]
            fig_fc = go.Figure(go.Bar(
                x=fl, y=fv,
                marker_color=["#4F63D2","#7C3AED","#06B6D4","#10B981","#F59E0B"],
                text=[fmt_inr(v) for v in fv],
                textposition="outside",textfont=dict(color="#fff",size=11),
            ))
            fig_fc.update_layout(
                title=dict(text="💰 Funding Comparison",font=dict(color="#A0B0FF",size=14)),
                paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=320,
                yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=","),
                xaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20))
            st.plotly_chart(fig_fc, use_container_width=True)

        with fc2:
            # Confidence donut
            fig_do = go.Figure(go.Pie(
                values=[ensemble*100, 100-ensemble*100],
                labels=["Success","Non-success"],
                hole=0.6,
                marker=dict(colors=[vc,"#1a2040"],line=dict(color="#060D26",width=2)),
                textfont=dict(color="#fff"),
                showlegend=False,
            ))
            fig_do.add_annotation(text=f"{ensemble*100:.1f}%",x=0.5,y=0.5,
                                   font=dict(size=28,color="#fff"),showarrow=False)
            fig_do.update_layout(
                title=dict(text="🎯 Ensemble Confidence",font=dict(color="#A0B0FF",size=14)),
                paper_bgcolor="#060D26",font_color="#fff",height=320,
                margin=dict(t=50,b=20))
            st.plotly_chart(fig_do, use_container_width=True)

        # ── Smart Recommendations ──
        st.markdown("#### 💡 AI-Powered Strategic Recommendations")
        recs=[]

        if sel_amt < 500_000:
            recs.append(("🔴","Critical: Funding Too Low",
                f"₹{sel_amt:,.0f} is below the minimum viable threshold. "
                f"The {tier} average is {fmt_inr(t_fund)}. Most investors expect at least ₹5L+.","#2d1515","#EF4444"))
        elif sel_amt < t_fund*0.5:
            recs.append(("⚠️","Below-Average Funding",
                f"Your ask ({fmt_inr(sel_amt)}) is below the {tier} average ({fmt_inr(t_fund)}). "
                "Consider raising more or clearly justifying lean operations.","#2d2a10","#F59E0B"))
        elif sel_amt > t_fund*5:
            recs.append(("💡","Ambitious Funding Target",
                f"Your target is {sel_amt/t_fund:.1f}× the {tier} average. "
                "Strong revenue traction, IP, or unit economics will be essential.","#101a3a","#4F63D2"))
        else:
            recs.append(("✅","Funding Well-Positioned",
                f"Your funding ({fmt_inr(sel_amt)}) aligns well with {tier} norms ({fmt_inr(t_fund)}).","#102d1f","#10B981"))

        if tier=="Tier 3":
            recs.append(("📍","City Tier Disadvantage",
                "Tier 3 cities receive <5% of India's VC deal flow. "
                "Consider a co-headquarters or registered office in Bangalore/Mumbai.","#2d1515","#EF4444"))
        elif tier=="Tier 1":
            recs.append(("🏙️","Prime Startup Ecosystem",
                f"Tier 1 cities account for 82% of Indian startup funding. "
                f"Your city ({sel_city}) is in the strongest ecosystem tier.","#102d1f","#10B981"))
        elif tier=="International":
            recs.append(("🌐","Global Positioning",
                "International presence signals global ambition. "
                "Ensure compliance with FEMA/RBI regulations for cross-border funding.","#101a3a","#4F63D2"))

        if sel_ind in HIGH_FUND_INDUSTRIES:
            recs.append(("🔥","Hot Industry Sector",
                f"{sel_ind} is among India's highest-funded sectors. "
                f"Industry average: {fmt_inr(i_avg)}. "
                "Investor competition is high — differentiation is critical.","#102d1f","#10B981"))
        else:
            recs.append(("🏭","Niche Industry",
                f"{sel_ind[:40]} averages {fmt_inr(i_avg)} in funding. "
                "Niche sectors may face longer fundraising timelines but less competition.","#101a3a","#4F63D2"))

        if ensemble<0.35:
            recs.append(("🚨","Low Success Probability",
                "Focus on: (1) Product-market fit evidence, (2) 3–6 months runway, "
                "(3) At least 2 paying customers before approaching VCs.","#2d1515","#EF4444"))
        elif ensemble>=0.60:
            recs.append(("🚀","Strong Success Outlook",
                "Prioritize investor deck quality, due-diligence readiness, "
                "and leverage your tier + industry advantages.","#102d1f","#10B981"))

        cols_r = st.columns(2)
        for i,(icon,title,body,bg,border) in enumerate(recs):
            with cols_r[i%2]:
                st.markdown(f"""
                <div style='background:{bg};border-left:4px solid {border};
                     border-radius:8px;padding:14px 16px;margin:6px 0;'>
                  <div style='font-weight:800;color:#fff;font-size:.97rem;'>{icon} {title}</div>
                  <div style='color:#C0CFFF;font-size:.87rem;margin-top:5px;'>{body}</div>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TAB 2  FEATURE EXPLORER
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📐 13 Engineered Predictors — Explained")

    feat_info = [
        ("1","💰 Funding Amount",          "Raw INR funding amount entered by user","Input"),
        ("2","📈 Log Funding",              "log(1+amount) — removes skew from large outliers","Transformation"),
        ("3","🪣 Funding Bucket",           "Categorical: <5L / 5L-20L / 20L-1Cr / 1Cr-5Cr / 5Cr-50Cr / 50Cr+","Categorisation"),
        ("4","🏙️ City Tier Number",         "1=Tier1, 2=Tier2, 3=Tier3, 0=International, 4=Other","City Feature"),
        ("5","🔵 Is Tier 1 City",           "Binary flag — is the city in top startup ecosystems?","City Feature"),
        ("6","🌐 Is International City",    "Binary flag — does the city have an international component?","City Feature"),
        ("7","🌆 City Startup Density",     "How many startups are in the same city (ecosystem depth)","City Feature"),
        ("8","💵 City Avg Funding",         "Historical average funding raised in that city","City Feature"),
        ("9","✅ City Success Rate",        "Historical success rate of startups in that city","City Feature"),
        ("10","🔥 Is Hot Industry",         "Binary — is the industry in top 7 highest-funded sectors?","Industry Feature"),
        ("11","📊 Industry Density",        "How many startups operate in the same industry","Industry Feature"),
        ("12","💵 Industry Avg Funding",    "Historical average funding in the same industry","Industry Feature"),
        ("13","📐 Fund vs Industry Avg",    "Ratio of user's funding to industry average (relative position)","Ratio Feature"),
    ]

    type_colors = {"Input":"#4F63D2","Transformation":"#7C3AED","Categorisation":"#06B6D4",
                   "City Feature":"#10B981","Industry Feature":"#F59E0B","Ratio Feature":"#EC4899"}

    frows = [feat_info[i:i+3] for i in range(0,len(feat_info),3)]
    for row in frows:
        cols_f = st.columns(3)
        for j,(num,name,desc,ftype) in enumerate(row):
            with cols_f[j]:
                st.markdown(f"""
                <div style='background:#0D1640;border:1px solid {type_colors[ftype]};
                     border-radius:10px;padding:14px;min-height:110px;margin:4px 0;'>
                  <div style='font-size:1.5rem;font-weight:800;color:{type_colors[ftype]};'>#{num}</div>
                  <div style='color:#fff;font-weight:700;font-size:.95rem;margin:4px 0;'>{name}</div>
                  <div style='color:#A0B0FF;font-size:.82rem;'>{desc}</div>
                  <div style='margin-top:8px;'>
                    <span style='background:{type_colors[ftype]}33;color:{type_colors[ftype]};
                         font-size:.75rem;font-weight:700;padding:2px 8px;border-radius:4px;'>
                      {ftype}
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # Feature importance chart
    fi = raw_stats["feat_imp"]
    fig_fi = go.Figure(go.Bar(
        x=fi.values[::-1],
        y=fi.index[::-1],
        orientation="h",
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(fi))],
        text=[f"{v:.4f}" for v in fi.values[::-1]],
        textposition="outside",textfont=dict(color="#fff",size=11),
    ))
    fig_fi.update_layout(
        title=dict(text="🌲 Feature Importance (Random Forest)",font=dict(color="#A0B0FF",size=15)),
        paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=400,
        xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF"),
        yaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20,l=10,r=80))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Model comparison table
    st.markdown("#### 🤖 5 Models — What They Each Do")
    model_info = [
        ["Logistic Regression","Linear decision boundary","Fast, interpretable, good baseline","Low complexity data"],
        ["Random Forest",     "Ensemble of 300 decision trees","Handles non-linearity, robust","Feature importance"],
        ["Gradient Boosting", "Sequential trees correcting errors","Very high accuracy","Imbalanced classes"],
        ["SVM",               "Hyperplane margin maximiser","Works well in high-dim space","Clear class margin"],
        ["KNN",               "K nearest-neighbour voting","Non-parametric, instance-based","Local pattern data"],
    ]
    df_models = pd.DataFrame(model_info,
                             columns=["Model","Mechanism","Strength","Best For"])
    st.dataframe(df_models.style.set_properties(**{
        "background-color":"#0D1640","color":"white","border":"1px solid #2a3060"
    }), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────
# TAB 3  CITY & TIER
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🏙️ City & Tier Intelligence")
    city_df = raw_stats["city_df"]
    tier_df = raw_stats["tier_df"]

    # Tier summary cards
    t_cols = st.columns(len(tier_df))
    for i,row in tier_df.iterrows():
        with t_cols[i]:
            tc = tier_color(row["tier"])
            st.markdown(f"""
            <div style='background:#060D26;border:2px solid {tc};border-radius:12px;
                 padding:16px;text-align:center;'>
              <div style='color:{tc};font-weight:800;font-size:1rem;'>{row["tier"]}</div>
              <div style='color:#fff;font-size:1.6rem;font-weight:700;'>{row["total_startups"]:,}</div>
              <div style='color:#A0B0FF;font-size:.8rem;'>startups</div>
              <div style='color:#7DF9C8;font-weight:700;margin-top:6px;'>{fmt_inr(row["Amount in USD"])}</div>
              <div style='color:#A0B0FF;font-size:.8rem;'>avg funding</div>
              <div style='color:#F59E0B;font-weight:700;'>{row["avg_success_rate"]*100:.1f}%</div>
              <div style='color:#A0B0FF;font-size:.8rem;'>success rate</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Tier bar charts
    tc1,tc2 = st.columns(2)
    with tc1:
        fig_tf = go.Figure(go.Bar(
            x=tier_df["tier"],y=tier_df["Amount in USD"],
            marker_color=[tier_color(t) for t in tier_df["tier"]],
            text=[fmt_inr(v) for v in tier_df["Amount in USD"]],
            textposition="outside",textfont=dict(color="#fff",size=11)))
        fig_tf.update_layout(
            title=dict(text="💰 Avg Funding by City Tier",font=dict(color="#A0B0FF",size=14)),
            paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=320,
            yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF"),
            xaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20))
        st.plotly_chart(fig_tf, use_container_width=True)

    with tc2:
        fig_ts = go.Figure(go.Bar(
            x=tier_df["tier"],y=tier_df["avg_success_rate"]*100,
            marker_color=[tier_color(t) for t in tier_df["tier"]],
            text=[f"{v*100:.1f}%" for v in tier_df["avg_success_rate"]],
            textposition="outside",textfont=dict(color="#fff",size=11)))
        fig_ts.update_layout(
            title=dict(text="✅ Success Rate by Tier",font=dict(color="#A0B0FF",size=14)),
            paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=320,
            yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",title="%"),
            xaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20))
        st.plotly_chart(fig_ts, use_container_width=True)

    # Bubble chart — all cities
    top20 = city_df.nlargest(20,"count").copy()
    fig_b = px.scatter(
        top20, x="Amount in USD", y="Success",
        size="count", color="tier",
        hover_name="city_name",
        text=top20["city_name"].str[:12],
        color_discrete_map={"Tier 1":"#4F63D2","Tier 2":"#06B6D4","Tier 3":"#10B981",
                             "International":"#F59E0B","Other":"#6B7280"},
        size_max=55,
        labels={"Amount in USD":"Avg Funding","Success":"Success Rate","count":"# Startups"},
        title="🗺️ City Intelligence Map (size = # startups, colour = tier)")
    fig_b.update_traces(textposition="top center",textfont=dict(color="#fff",size=9))
    fig_b.update_layout(paper_bgcolor="#0
    60D26",plot_bgcolor="#060D26",font_color="#fff",
     height=460,margin=dict(t=60,b=40,l=40,r=20),
 xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=","),
                        yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=".1%"),
                        legend=dict(bgcolor="#060D26",bordercolor="#2a3060"))
    st.plotly_chart(fig_b, use_container_width=True)

    # Top 10 horizontal bars
    top10 = city_df.nlargest(10,"count").sort_values("count")
    fig_h = go.Figure(go.Bar(
        y=top10["city_name"],x=top10["count"],orientation="h",
        marker_color=[tier_color(t) for t in top10["tier"]],
        text=top10["count"],textposition="outside",textfont=dict(color="#fff",size=11)))
    fig_h.update_layout(
        title=dict(text="🏙️ Top 10 Cities by Startup Volume",font=dict(color="#A0B0FF",size=14)),
        paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=360,
        xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF"),
        yaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=30,l=10,r=60))
    st.plotly_chart(fig_h, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# TAB 4  INDUSTRY DEEP DIVE
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🏭 Industry Deep Dive")
    ind_df = raw_stats["ind_df"].copy()
    n = st.slider("Industries to show", 5, 20, 12)

    top_f = ind_df.nlargest(n,"Amount in USD")
    fig_if = go.Figure(go.Bar(
        x=top_f["Amount in USD"],y=[s[:35] for s in top_f["Industry Vertical"]],
        orientation="h",
        marker_color=PALETTE*(n//len(PALETTE)+1),
        text=[fmt_inr(v) for v in top_f["Amount in USD"]],
        textposition="outside",textfont=dict(color="#fff",size=10)))
    fig_if.update_layout(
        title=dict(text=f"💰 Top {n} Industries by Avg Funding",font=dict(color="#A0B0FF",size=14)),
        paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=440,
        xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=","),
        yaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20,l=10,r=90),bargap=0.25)
    st.plotly_chart(fig_if, use_container_width=True)

    ia1,ia2 = st.columns(2)
    with ia1:
        top_c = ind_df.nlargest(n,"count").sort_values("count")
        fig_ic = go.Figure(go.Bar(
            y=[s[:30] for s in top_c["Industry Vertical"]],x=top_c["count"],
            orientation="h",marker_color="#4F63D2",
            text=top_c["count"],textposition="outside",textfont=dict(color="#fff",size=10)))
        fig_ic.update_layout(
            title=dict(text=f"📊 Top {n} by Volume",font=dict(color="#A0B0FF",size=14)),
            paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=400,
            xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF"),
            yaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=20,l=10,r=60))
        st.plotly_chart(fig_ic, use_container_width=True)

    with ia2:
        top_t = ind_df.nlargest(20,"count")
        fig_tree = px.treemap(top_t,path=["Industry Vertical"],values="count",
                              color="Amount in USD",
                              color_continuous_scale=["#0D1128","#4F63D2","#7C3AED","#EC4899"],
                              title="🗂️ Industry Treemap")
        fig_tree.update_layout(paper_bgcolor="#060D26",font_color="#fff",height=400,
                                margin=dict(t=50,b=10))
        fig_tree.update_traces(textfont=dict(color="#fff"))
        st.plotly_chart(fig_tree, use_container_width=True)

    # Scatter: funding vs success
    top_sc = ind_df.nlargest(30,"count")
    fig_sc = px.scatter(
        top_sc,x="Amount in USD",y="Success",size="count",color="Amount in USD",
        text=[s[:18] for s in top_sc["Industry Vertical"]],
        color_continuous_scale=["#4F63D2","#7C3AED","#EC4899"],
        labels={"Amount in USD":"Avg Funding","Success":"Success Rate","count":"# Startups"},
        title="🔬 Industry Landscape — Funding vs. Success Rate",size_max=50)
    fig_sc.update_traces(textposition="top center",textfont=dict(color="#fff",size=9))
    fig_sc.update_layout(paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",
                         height=440,margin=dict(t=60,b=40),
                         xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=","),
                         yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=".1%"))
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# TAB 5  MARKET OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📊 Indian Startup Market Overview")
    bd = raw_stats["bucket_df"]

    o1,o2 = st.columns(2)
    with o1:
        fig_bk = go.Figure(go.Bar(
            x=bd["bucket"].astype(str),y=bd["count"],
            marker_color=PALETTE[:len(bd)],
            text=bd["count"],textposition="outside",textfont=dict(color="#fff",size=11)))
        fig_bk.update_layout(
            title=dict(text="📦 Startups by Funding Bucket",font=dict(color="#A0B0FF",size=14)),
            paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=320,
            xaxis=dict(color="#D0DBFF"),
            yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF"),margin=dict(t=50,b=30))
        st.plotly_chart(fig_bk, use_container_width=True)

    with o2:
        fig_bsr = go.Figure(go.Bar(
            x=bd["bucket"].astype(str),y=bd["Success"]*100,
            marker_color=["#10B981" if v>0.02 else "#4F63D2" for v in bd["Success"]],
            text=[f"{v*100:.1f}%" for v in bd["Success"]],
            textposition="outside",textfont=dict(color="#fff",size=11)))
        fig_bsr.update_layout(
            title=dict(text="✅ Success Rate by Funding Bucket",font=dict(color="#A0B0FF",size=14)),
            paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=320,
            xaxis=dict(color="#D0DBFF"),
            yaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",title="%"),
            margin=dict(t=50,b=30))
        st.plotly_chart(fig_bsr, use_container_width=True)

    # Top cities by avg funding
    top_cf = raw_stats["city_df"].nlargest(10,"Amount in USD").sort_values("Amount in USD")
    fig_cf = go.Figure(go.Bar(
        y=top_cf["city_name"],x=top_cf["Amount in USD"],orientation="h",
        marker_color=[tier_color(t) for t in top_cf["tier"]],
        text=[fmt_inr(v) for v in top_cf["Amount in USD"]],
        textposition="outside",textfont=dict(color="#fff",size=11)))
    fig_cf.update_layout(
        title=dict(text="💰 Top 10 Cities by Avg Funding per Startup",font=dict(color="#A0B0FF",size=14)),
        paper_bgcolor="#060D26",plot_bgcolor="#060D26",font_color="#fff",height=380,
        xaxis=dict(showgrid=True,gridcolor="#1e2a50",color="#A0B0FF",tickformat=","),
        yaxis=dict(color="#D0DBFF"),margin=dict(t=50,b=30,l=10,r=90))
    st.plotly_chart(fig_cf, use_container_width=True)

    # Tier pie
    fig_pie = go.Figure(go.Pie(
        labels=tier_df["tier"],values=tier_df["total_startups"],hole=0.45,
        marker=dict(colors=[tier_color(t) for t in tier_df["tier"]],
                    line=dict(color="#060D26",width=2)),
        textfont=dict(color="#fff",size=13),showlegend=True))
    fig_pie.add_annotation(text="Tiers",x=0.5,y=0.5,
                            font=dict(size=16,color="#fff"),showarrow=False)
    fig_pie.update_layout(
        title=dict(text="🗺️ Startup Distribution by City Tier",font=dict(color="#A0B0FF",size=14)),
        paper_bgcolor="#060D26",font_color="#fff",height=380,
        legend=dict(bgcolor="#060D26",bordercolor="#2a3060",font=dict(color="#fff")),
        margin=dict(t=60,b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Platform Summary")
    st.markdown("""
| Component | Detail |
|---|---|
| Dataset | Indian Startup Funding — 2,071 startups |
| Classifiers | Logistic Regression · Random Forest · Gradient Boosting · SVM · KNN |
| Regressor | Ridge Regression (predict funding amount) |
| Predictors | **13 engineered features** from 3 raw inputs |
| City Tiers | Tier 1 (Bangalore/Mumbai/Delhi) · Tier 2 (Pune/Jaipur…) · Tier 3 (smaller cities) |
| Charts | Gauge · Bar · Bubble · Radar · Donut · Treemap · Scatter · Pie · Feature Importance |
| Predictions | Success probability (5 models + ensemble) · Funding estimate · Smart recommendations |
""")
