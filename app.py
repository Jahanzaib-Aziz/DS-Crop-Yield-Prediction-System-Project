# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ==============================
# MAX YIELD OPTIMIZATION FUNCTION
# ==============================
def find_best_input(model, features, df, n_trials=500):
    best_input = None
    best_output = -1

    for _ in range(n_trials):
        sample = {}
        for col in features:
            min_val = df[col].min()
            max_val = df[col].max()
            sample[col] = np.random.uniform(min_val, max_val)

        sample_df = pd.DataFrame([sample])
        pred = model.predict(sample_df)[0]

        if pred > best_output:
            best_output = pred
            best_input = sample

    return best_input, best_output


# ==============================
# PROFESSIONAL PDF GENERATOR
# ==============================
def generate_professional_pdf(pred, risk, input_data, model_name=None, df=None):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    # ---------- Color Palette ----------
    GREEN_DARK  = colors.HexColor("#1B5E20")
    GREEN_MID   = colors.HexColor("#2E7D32")
    GREEN_LIGHT = colors.HexColor("#A5D6A7")
    GREEN_PALE  = colors.HexColor("#E8F5E9")
    YELLOW      = colors.HexColor("#F9A825")
    RED         = colors.HexColor("#B71C1C")
    GREY_DARK   = colors.HexColor("#37474F")
    GREY_LIGHT  = colors.HexColor("#ECEFF1")
    WHITE       = colors.white

    risk_color = {
        "Low":    RED,
        "Medium": YELLOW,
        "High":   GREEN_MID,
    }.get(risk.split()[0], GREEN_MID)

    # ---------- Custom Styles ----------
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        fontSize=20,          # ← reduced from 26
        fontName="Helvetica-Bold",
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=0,         # ← changed from 4 to 0
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        fontSize=11,
        fontName="Helvetica",
        textColor=GREEN_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=GREEN_DARK,
        spaceBefore=14,
        spaceAfter=6,
    )
    normal = ParagraphStyle(
        "NormalCustom",
        fontSize=10,
        fontName="Helvetica",
        textColor=GREY_DARK,
        spaceAfter=4,
        leading=14,
    )
    footer_style = ParagraphStyle(
        "Footer",
        fontSize=8,
        fontName="Helvetica",
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    label_style = ParagraphStyle(
        "Label",
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=GREY_DARK,
    )
    value_style = ParagraphStyle(
        "Value",
        fontSize=9,
        fontName="Helvetica",
        textColor=GREY_DARK,
    )

    story = []
    W, _ = A4

    # ============================================================
    # HEADER BANNER
    # ============================================================
    header_data = [[
        Paragraph("AI Crop Yield Prediction Report", title_style),
    ]]
    header_table = Table(header_data, colWidths=[W - 3 * cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 20),   # ← increased
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),   # ← increased from 10
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0))   # ← change from 6 to 0

    # subtitle row (date + model name)
    now = datetime.now().strftime("%B %d, %Y  |  %H:%M")
    model_label = f"Model: {model_name}" if model_name else ""
    sub_data = [[
        Paragraph(f"Generated on {now}    {model_label}", subtitle_style),
    ]]
    sub_table = Table(sub_data, colWidths=[W - 3 * cm])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sub_table)      # ✅ append subtitle after creation
    story.append(Spacer(1, 14))

    # ============================================================
    # SUMMARY CARDS  (3-column row)
    # ============================================================
    story.append(Paragraph("Executive Summary", section_heading))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=8))

    col_w = (W - 3 * cm) / 3 - 4

    yield_card = Table([
        [Paragraph("PREDICTED YIELD", ParagraphStyle("cl", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
        [Paragraph(f"{pred:.1f}%",    ParagraphStyle("cv", fontSize=28, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    ], colWidths=[col_w])
    yield_card.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREEN_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))

    risk_card = Table([
        [Paragraph("RISK LEVEL", ParagraphStyle("cl2", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
        [Paragraph(risk,         ParagraphStyle("cv2", fontSize=18, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    ], colWidths=[col_w])
    risk_card.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), risk_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))

    status_text = "Excellent" if pred >= 66 else ("Moderate" if pred >= 33 else "Poor")
    status_card = Table([
        [Paragraph("CROP STATUS",  ParagraphStyle("cl3", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
        [Paragraph(status_text,    ParagraphStyle("cv3", fontSize=18, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER))],
    ], colWidths=[col_w])
    status_card.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GREY_DARK),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))

    outer = Table([[yield_card, risk_card, status_card]], colWidths=[col_w + 4, col_w + 4, col_w + 4])
    outer.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(outer)
    story.append(Spacer(1, 14))

    # ============================================================
    # INTERPRETATION
    # ============================================================
    story.append(Paragraph("Yield Interpretation", section_heading))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=6))

    key = risk.split()[0]
    interp_map = {
        "High":   (
            "The predicted yield of <b>{:.1f}%</b> falls in the <b>High</b> category (above 66%). "
            "Crop conditions are optimal. Maintain current farming practices and monitor for "
            "any late-season environmental changes."
        ),
        "Medium": (
            "The predicted yield of <b>{:.1f}%</b> falls in the <b>Medium</b> category (33% to 65%). "
            "Conditions are adequate but improvement is possible. Consider reviewing irrigation "
            "schedules, fertiliser application, and soil health."
        ),
        "Low":    (
            "The predicted yield of <b>{:.1f}%</b> falls in the <b>Low</b> category (below 33%). "
            "Immediate attention is recommended. Review soil nutrients, water availability, "
            "pest/disease pressure, and weather forecasts."
        ),
    }
    interp_text = interp_map.get(key, interp_map["Medium"]).format(pred)
    story.append(Paragraph(interp_text, normal))
    story.append(Spacer(1, 6))

    # ============================================================
    # YIELD GAUGE BAR
    # ============================================================
    fig_gauge, ax_gauge = plt.subplots(figsize=(6, 0.8))
    bar_color = "#2E7D32" if pred >= 66 else "#F9A825" if pred >= 33 else "#B71C1C"
    ax_gauge.barh([0], [100], color="#E8F5E9", height=0.5)
    ax_gauge.barh([0], [pred], color=bar_color, height=0.5)
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_yticks([])
    ax_gauge.set_xticks([0, 33, 66, 100])
    ax_gauge.set_xticklabels(["0%", "33%", "66%", "100%"], fontsize=8)
    ax_gauge.axvline(33, color="#F9A825", lw=1, ls="--")
    ax_gauge.axvline(66, color="#2E7D32", lw=1, ls="--")
    ax_gauge.set_title(f"Yield Gauge: {pred:.1f}%", fontsize=9, pad=4)
    ax_gauge.spines[["top", "right", "left"]].set_visible(False)
    fig_gauge.tight_layout()
    gauge_buf = io.BytesIO()
    fig_gauge.savefig(gauge_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_gauge)
    gauge_buf.seek(0)
    story.append(Image(gauge_buf, width=14 * cm, height=2.2 * cm))
    story.append(Spacer(1, 10))

    # ============================================================
    # INPUT PARAMETERS TABLE
    # ============================================================
    story.append(Paragraph("Input Parameters Used for Prediction", section_heading))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=6))

    if input_data:
        table_data = [[
            Paragraph("Parameter",      label_style),
            Paragraph("Value Provided", label_style),
            Paragraph("Dataset Min",    label_style),
            Paragraph("Dataset Max",    label_style),
            Paragraph("Dataset Mean",   label_style),
        ]]
        for col, val in input_data.items():
            if df is not None and col in df.columns:
                dmin  = f"{df[col].min():.3f}"
                dmax  = f"{df[col].max():.3f}"
                dmean = f"{df[col].mean():.3f}"
            else:
                dmin = dmax = dmean = "N/A"
            table_data.append([
                Paragraph(col,           value_style),
                Paragraph(f"{val:.3f}",  value_style),
                Paragraph(dmin,          value_style),
                Paragraph(dmax,          value_style),
                Paragraph(dmean,         value_style),
            ])

        param_table = Table(table_data, colWidths=[4.5*cm, 3*cm, 3*cm, 3*cm, 3*cm], repeatRows=1)
        param_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), GREEN_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 9),
            ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GREEN_PALE]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
            ("BOX",           (0, 0), (-1, -1), 0.8, GREEN_DARK),
        ]))
        story.append(param_table)
    else:
        story.append(Paragraph("No input parameters available.", normal))

    story.append(Spacer(1, 14))

    # ============================================================
    # YIELD DISTRIBUTION CHART
    # ============================================================
    if df is not None and "Yield" in df.columns:
        story.append(Paragraph("Dataset Yield Distribution", section_heading))
        story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=6))

        fig_hist, ax_hist = plt.subplots(figsize=(7, 3))
        ax_hist.hist(df["Yield"], bins=25, color="#2E7D32", edgecolor="white", alpha=0.85)
        ax_hist.axvline(pred, color="#B71C1C", lw=2, ls="--",
                        label=f"Your Prediction: {pred:.1f}%")
        ax_hist.axvline(df["Yield"].mean(), color="#F9A825", lw=1.5, ls=":",
                        label=f"Dataset Mean: {df['Yield'].mean():.1f}%")
        ax_hist.set_xlabel("Yield (%)", fontsize=9)
        ax_hist.set_ylabel("Frequency", fontsize=9)
        ax_hist.set_title("Yield Distribution Across Dataset", fontsize=10, pad=6)
        ax_hist.legend(fontsize=8)
        ax_hist.spines[["top", "right"]].set_visible(False)
        fig_hist.tight_layout()
        hist_buf = io.BytesIO()
        fig_hist.savefig(hist_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_hist)
        hist_buf.seek(0)
        story.append(Image(hist_buf, width=14 * cm, height=6 * cm))
        story.append(Spacer(1, 6))

        stats = df["Yield"].describe()
        stats_data = [
            [Paragraph("Statistic", label_style), Paragraph("Value", label_style)],
            ["Count",   f"{int(stats['count'])}"],
            ["Mean",    f"{stats['mean']:.2f}%"],
            ["Std Dev", f"{stats['std']:.2f}%"],
            ["Min",     f"{stats['min']:.2f}%"],
            ["25%",     f"{stats['25%']:.2f}%"],
            ["Median",  f"{stats['50%']:.2f}%"],
            ["75%",     f"{stats['75%']:.2f}%"],
            ["Max",     f"{stats['max']:.2f}%"],
        ]
        st_table = Table(stats_data, colWidths=[5*cm, 4*cm])
        st_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), GREEN_DARK),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GREEN_PALE]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#B0BEC5")),
            ("BOX",           (0, 0), (-1, -1), 0.8, GREEN_DARK),
        ]))
        story.append(st_table)
        story.append(Spacer(1, 14))

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================
    story.append(Paragraph("Recommendations", section_heading))
    story.append(HRFlowable(width="100%", thickness=1.5, color=GREEN_LIGHT, spaceAfter=6))

    recs = {
        "High": [
            "Maintain current input levels — conditions are optimal.",
            "Schedule a post-harvest soil test to preserve soil health.",
            "Consider crop rotation to avoid nutrient depletion.",
            "Document current practices for replication in future seasons.",
        ],
        "Medium": [
            "Review irrigation schedule and ensure consistent water supply.",
            "Conduct a soil nutrient analysis and adjust fertiliser accordingly.",
            "Monitor weather forecasts and adjust planting schedules.",
            "Consult an agronomist for targeted improvement strategies.",
        ],
        "Low": [
            "Urgently assess soil health and pH levels.",
            "Check for pest or disease pressure and apply appropriate treatment.",
            "Evaluate water stress — consider drip irrigation.",
            "Consider replanting with a more climate-resilient variety.",
            "Seek government or NGO agricultural support if available.",
        ],
    }

    rec_list = recs.get(key, recs["Medium"])
    rec_data = [[
        Paragraph("✔", ParagraphStyle("bullet", fontSize=10, fontName="Helvetica-Bold",
                                       textColor=GREEN_MID, alignment=TA_CENTER)),
        Paragraph(r, normal),
    ] for r in rec_list]

    rec_table = Table(rec_data, colWidths=[0.8*cm, W - 3*cm - 0.8*cm])
    rec_table.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, GREEN_PALE]),
        ("BOX",           (0, 0), (-1, -1), 0.6, GREEN_LIGHT),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.3, GREEN_LIGHT),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 16))

    # ============================================================
    # FOOTER
    # ============================================================
    story.append(HRFlowable(width="100%", thickness=0.8, color=GREEN_LIGHT, spaceAfter=6))
    story.append(Paragraph(
        "This report was generated automatically by the AI Crop Yield Prediction System. "
        "Predictions are based on historical data and machine learning models — always "
        "consult a qualified agronomist before making major farming decisions.",
        footer_style
    ))
    story.append(Paragraph(
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="🌾 AI Crop Yield Prediction System", layout="wide")

# ==============================
# SESSION STATE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None


# ==============================
# LOGIN
# ==============================
def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "Jahanzaib" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Login")


# ==============================
# DATA UPLOAD
# ==============================
def data_tab():
    st.header("📁 Upload Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = df.drop_duplicates()
        df = df.fillna(df.mean(numeric_only=True))
        remove_cols = ["Crop", "Crop_Year", "Season", "State",
                       "Min_Temp", "Max_Temp", "Production"]
        df = df.drop(remove_cols, axis=1, errors='ignore')
        for col in df.select_dtypes(include=np.number).columns:
            df = df[(df[col] >= df[col].quantile(0.01)) &
                    (df[col] <= df[col].quantile(0.99))]
        if "Yield" in df.columns:
            df["Yield"] = (df["Yield"] / df["Yield"].max()) * 100
        st.session_state.df = df
        st.subheader("✅ Cleaned Dataset")
        st.dataframe(df)
        st.download_button(
            "📥 Download Cleaned Dataset",
            df.to_csv(index=False),
            "cleaned_crop_data.csv"
        )


# ==============================
# TRAINING
# ==============================
def train_tab():
    st.header("🤖 Model Training")
    df = st.session_state.df
    if df is None:
        st.warning("Upload dataset first")
        return

    target = "Yield"
    X = df.drop(target, axis=1)
    y = df[target]
    st.session_state.features = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear":        LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = r2_score(y_test, pred)

    best_name  = max(results, key=results.get)
    best_model = models[best_name]
    best_model.fit(X, y)

    st.session_state.model           = best_model
    st.session_state.best_model_name = best_name

    st.subheader("📊 Model Comparison")
    st.bar_chart(results)
    st.success(f"Best Model: {best_name}")


# ==============================
# PREDICTION + PDF
# ==============================
def predict_tab():
    st.header("📊 Prediction")
    model = st.session_state.model
    df    = st.session_state.df

    if model is None or df is None:
        st.warning("Train model first")
        return

    st.subheader("Enter Custom Inputs")
    input_data = {}
    for col in st.session_state.features:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

    input_df = pd.DataFrame([input_data])

    pred = None
    risk = None

    # ==============================
    # PREDICTION BUTTON
    # ==============================
    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        pred = max(0, min(pred, 100))
        st.success(f"🌾 Predicted Yield: {pred:.2f}%")

        if pred < 33:
            risk = "Low ⚠️"
        elif pred < 66:
            risk = "Medium ⚖️"
        else:
            risk = "High ✅"

        st.info(f"Risk Level: {risk}")

        st.session_state.last_pred       = pred
        st.session_state.last_risk       = risk
        st.session_state.last_input_data = input_data

    # ==============================
    # FIND MAX YIELD BUTTON
    # ==============================
    if st.button("🚀 Find Maximum Yield"):
        best_input, best_output = find_best_input(
            st.session_state.model,
            st.session_state.features,
            st.session_state.df
        )
        st.success(f"Max Predicted Yield: {best_output:.2f}%")
        st.subheader("Best Input Values to Get Max. yield:")
        for k, v in best_input.items():
            st.write(f"{k}: {v:.2f}")

        # Feature Impact Graph
        st.subheader("📊 Feature Impact Simulation")
        feature_name = st.selectbox("Select Feature to Analyze", st.session_state.features)
        values = []
        for i in range(10):
            temp_df = input_df.copy()
            temp_df[feature_name] += i * 5
            pred_val = model.predict(temp_df)[0]
            values.append(pred_val)

        fig, ax = plt.subplots()
        ax.plot(values)
        ax.set_title(f"Impact of {feature_name} on Yield")
        st.pyplot(fig)

        # ==============================
        # PROFESSIONAL PDF DOWNLOAD
        # ==============================
        pred_for_pdf  = st.session_state.get("last_pred", best_output)
        risk_for_pdf  = st.session_state.get("last_risk", "High ✅")
        input_for_pdf = st.session_state.get("last_input_data", input_data)

        pdf_buffer = generate_professional_pdf(
            pred       = pred_for_pdf,
            risk       = risk_for_pdf,
            input_data = input_for_pdf,
            model_name = st.session_state.get("best_model_name"),
            df         = st.session_state.df,
        )

        st.download_button(
            "📄 Download Professional PDF Report",
            pdf_buffer,
            "crop_yield_report.pdf",
            mime="application/pdf"
        )


# ==============================
# ANALYSIS
# ==============================
def analysis_tab():
    st.header("📈 Analysis")
    df = st.session_state.df
    if df is None:
        st.warning("Upload dataset first")
        return

    fig, ax = plt.subplots()
    ax.hist(df["Yield"], bins=20)
    st.pyplot(fig)
    st.write(df.describe())


# ==============================
# MAIN APP
# ==============================
if not st.session_state.logged_in:
    login()
else:
    st.sidebar.title("🌾 AI Crop Yield Prediction System")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 10px;'>
            <p style='color: red`; font-size: 13px; margin:0;'>Conceptualized, Designed & Developed by:</p>
            <p style='color: brown; font-size: 16px; font-weight: bold; margin:0;'>Deadline warriors Group</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data",
        "🤖 Training",
        "📊 Prediction",
        "📈 Analysis"
    ])

    with tab1:
        data_tab()
    with tab2:
        train_tab()
    with tab3:
        predict_tab()
    with tab4:
        analysis_tab()
