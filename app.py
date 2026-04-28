
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# PLI DEMO - Predictive Location Intelligence by BDS
# Demo comercial con datos sintéticos.
# No utiliza APIs externas. Puede correr localmente.
# ============================================================

st.set_page_config(
    page_title="PLI Demo | Business Data Scientists",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_PRIMARY = "#0F172A"
BRAND_ACCENT = "#2563EB"
BRAND_MUTED = "#64748B"
BRAND_SUCCESS = "#16A34A"
BRAND_WARNING = "#D97706"
BRAND_RISK = "#DC2626"


# -----------------------------
# Utilidades
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def money_mxn(x):
    if pd.isna(x):
        return "N/D"
    return f"${x:,.0f} MXN"


def pct(x):
    if pd.isna(x):
        return "N/D"
    return f"{x:.1%}"


def format_number(x):
    return f"{x:,.0f}"


def decision_label(row, min_roi, max_cannibalization):
    if row["roi_12m"] >= min_roi and row["cannibalization_risk"] <= max_cannibalization and row["pli_score"] >= 72:
        return "Aprobar"
    if row["roi_12m"] >= min_roi * 0.65 and row["cannibalization_risk"] <= max_cannibalization + 0.12:
        return "Revisar"
    return "No aprobar"


def decision_color(decision):
    return {
        "Aprobar": BRAND_SUCCESS,
        "Revisar": BRAND_WARNING,
        "No aprobar": BRAND_RISK,
    }.get(decision, BRAND_MUTED)


# -----------------------------
# Generación de datos sintéticos
# -----------------------------
@st.cache_data
def generate_synthetic_network(seed=42):
    rng = np.random.default_rng(seed)

    zones = [
        ("Polanco", 19.4320, -99.1940, 1.35),
        ("Roma-Condesa", 19.4145, -99.1670, 1.25),
        ("Del Valle", 19.3845, -99.1630, 1.18),
        ("Coyoacán", 19.3467, -99.1617, 1.05),
        ("Santa Fe", 19.3590, -99.2760, 1.10),
        ("Narvarte", 19.3940, -99.1540, 1.12),
        ("Lindavista", 19.4930, -99.1320, 0.96),
        ("Iztapalapa", 19.3550, -99.0630, 0.82),
        ("Tlalpan", 19.2920, -99.1660, 0.93),
        ("Azcapotzalco", 19.4860, -99.1850, 0.91),
        ("Satélite", 19.5100, -99.2350, 1.00),
        ("Cuajimalpa", 19.3570, -99.2990, 0.98),
    ]

    records = []
    for i in range(72):
        zone_name, zlat, zlon, affluence = zones[rng.integers(0, len(zones))]
        lat = zlat + rng.normal(0, 0.012)
        lon = zlon + rng.normal(0, 0.012)

        foot_traffic = max(800, rng.normal(3800 * affluence, 850))
        households = max(900, rng.normal(7200 * affluence, 1400))
        income_index = np.clip(rng.normal(72 * affluence, 10), 25, 100)
        competitors_1km = int(np.clip(rng.normal(6 / affluence, 2.2), 0, 14))
        transit_score = np.clip(rng.normal(68 * affluence, 13), 10, 100)
        rent_m2 = max(180, rng.normal(520 * affluence, 120))
        visibility = np.clip(rng.normal(70 * affluence, 14), 15, 100)
        parking_score = np.clip(rng.normal(52 * affluence, 18), 5, 100)
        ecommerce_density = np.clip(rng.normal(55 * affluence, 16), 5, 100)
        store_size_m2 = max(65, rng.normal(155, 45))
        format_type = rng.choice(["Conveniencia", "Farmacia", "QSR"], p=[0.55, 0.25, 0.20])

        monthly_sales = (
            250000
            + foot_traffic * 82
            + households * 24
            + income_index * 8500
            + transit_score * 4200
            + visibility * 5100
            + parking_score * 1400
            + ecommerce_density * 1900
            + store_size_m2 * 1100
            - competitors_1km * 68000
            - rent_m2 * 180
            + rng.normal(0, 140000)
        )

        if format_type == "Farmacia":
            monthly_sales *= 0.88 + income_index / 350
        elif format_type == "QSR":
            monthly_sales *= 0.78 + foot_traffic / 16000

        monthly_sales = max(monthly_sales, 250000)

        records.append(
            {
                "store_id": f"T-{i+1:03d}",
                "type": "Tienda actual",
                "zone": zone_name,
                "format": format_type,
                "lat": lat,
                "lon": lon,
                "foot_traffic": round(foot_traffic),
                "households": round(households),
                "income_index": round(income_index, 1),
                "competitors_1km": competitors_1km,
                "transit_score": round(transit_score, 1),
                "rent_m2": round(rent_m2, 1),
                "visibility": round(visibility, 1),
                "parking_score": round(parking_score, 1),
                "ecommerce_density": round(ecommerce_density, 1),
                "store_size_m2": round(store_size_m2, 1),
                "monthly_sales": round(monthly_sales),
            }
        )

    current_stores = pd.DataFrame(records)

    candidates = []
    for i in range(40):
        zone_name, zlat, zlon, affluence = zones[rng.integers(0, len(zones))]
        lat = zlat + rng.normal(0, 0.017)
        lon = zlon + rng.normal(0, 0.017)

        foot_traffic = max(700, rng.normal(3950 * affluence, 1050))
        households = max(700, rng.normal(7500 * affluence, 1900))
        income_index = np.clip(rng.normal(70 * affluence, 12), 20, 100)
        competitors_1km = int(np.clip(rng.normal(6.5 / affluence, 2.8), 0, 16))
        transit_score = np.clip(rng.normal(66 * affluence, 15), 10, 100)
        rent_m2 = max(160, rng.normal(535 * affluence, 145))
        visibility = np.clip(rng.normal(68 * affluence, 16), 10, 100)
        parking_score = np.clip(rng.normal(50 * affluence, 20), 5, 100)
        ecommerce_density = np.clip(rng.normal(56 * affluence, 17), 5, 100)
        store_size_m2 = max(55, rng.normal(160, 50))
        format_type = rng.choice(["Conveniencia", "Farmacia", "QSR"], p=[0.55, 0.25, 0.20])

        candidates.append(
            {
                "candidate_id": f"C-{i+1:03d}",
                "type": "Candidato",
                "zone": zone_name,
                "format": format_type,
                "lat": lat,
                "lon": lon,
                "foot_traffic": round(foot_traffic),
                "households": round(households),
                "income_index": round(income_index, 1),
                "competitors_1km": competitors_1km,
                "transit_score": round(transit_score, 1),
                "rent_m2": round(rent_m2, 1),
                "visibility": round(visibility, 1),
                "parking_score": round(parking_score, 1),
                "ecommerce_density": round(ecommerce_density, 1),
                "store_size_m2": round(store_size_m2, 1),
            }
        )

    candidate_sites = pd.DataFrame(candidates)
    return current_stores, candidate_sites


@st.cache_data
def train_model(current_stores):
    features = [
        "foot_traffic",
        "households",
        "income_index",
        "competitors_1km",
        "transit_score",
        "rent_m2",
        "visibility",
        "parking_score",
        "ecommerce_density",
        "store_size_m2",
    ]

    x = current_stores[features]
    y = current_stores["monthly_sales"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=7
    )

    model = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.045,
        max_depth=3,
        random_state=7,
    )
    model.fit(x_train, y_train)

    pred_test = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, pred_test),
        "mape": mean_absolute_percentage_error(y_test, pred_test),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    importance = pd.DataFrame(
        {
            "driver": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, features, metrics, importance


def enrich_candidates(current_stores, candidate_sites, model, features, assumptions):
    candidates = candidate_sites.copy()
    candidates["predicted_monthly_sales"] = model.predict(candidates[features]).round(0)

    nearest_distances = []
    nearest_store_ids = []
    nearest_sales = []

    for _, c in candidates.iterrows():
        distances = current_stores.apply(
            lambda s: haversine_km(c["lat"], c["lon"], s["lat"], s["lon"]), axis=1
        )
        nearest_idx = distances.idxmin()
        nearest_distances.append(distances.loc[nearest_idx])
        nearest_store_ids.append(current_stores.loc[nearest_idx, "store_id"])
        nearest_sales.append(current_stores.loc[nearest_idx, "monthly_sales"])

    candidates["nearest_store_km"] = np.round(nearest_distances, 2)
    candidates["nearest_store_id"] = nearest_store_ids
    candidates["nearest_store_sales"] = nearest_sales

    # Riesgo de canibalización: alto si hay tienda actual cercana, muchas tiendas competidoras y baja densidad.
    candidates["cannibalization_risk"] = np.clip(
        0.52 * np.exp(-candidates["nearest_store_km"] / 1.8)
        + 0.025 * candidates["competitors_1km"]
        - 0.0018 * candidates["households"] / 10
        + 0.03 * (candidates["format"] == "Conveniencia").astype(int),
        0,
        0.85,
    )

    candidates["net_monthly_sales"] = (
        candidates["predicted_monthly_sales"] * (1 - candidates["cannibalization_risk"])
    ).round(0)

    candidates["monthly_gross_profit"] = (
        candidates["net_monthly_sales"] * assumptions["gross_margin"]
    ).round(0)

    candidates["monthly_fixed_cost"] = (
        assumptions["base_fixed_cost"]
        + candidates["rent_m2"] * candidates["store_size_m2"]
        + assumptions["labor_cost"]
    ).round(0)

    candidates["monthly_ebitda"] = (
        candidates["monthly_gross_profit"] - candidates["monthly_fixed_cost"]
    ).round(0)

    candidates["capex"] = (
        assumptions["capex_base"] + candidates["store_size_m2"] * assumptions["capex_per_m2"]
    ).round(0)

    candidates["roi_12m"] = (candidates["monthly_ebitda"] * 12) / candidates["capex"]
    candidates["payback_months"] = np.where(
        candidates["monthly_ebitda"] > 0,
        candidates["capex"] / candidates["monthly_ebitda"],
        np.nan,
    )

    # Score ejecutivo PLI.
    potential_score = (
        0.24 * (candidates["net_monthly_sales"] / candidates["net_monthly_sales"].max())
        + 0.16 * (candidates["foot_traffic"] / candidates["foot_traffic"].max())
        + 0.14 * (candidates["households"] / candidates["households"].max())
        + 0.10 * (candidates["income_index"] / 100)
        + 0.10 * (candidates["visibility"] / 100)
        + 0.08 * (candidates["transit_score"] / 100)
        + 0.08 * (candidates["parking_score"] / 100)
        + 0.10 * np.clip(candidates["roi_12m"], 0, 2.5) / 2.5
    )
    risk_penalty = (
        0.18 * candidates["cannibalization_risk"]
        + 0.035 * candidates["competitors_1km"]
        + 0.00015 * candidates["rent_m2"]
    )
    candidates["pli_score"] = np.clip((potential_score - risk_penalty + 0.20) * 100, 0, 100).round(1)

    # Capacidad estimada de mercado por zona.
    zone_capacity = (
        candidates.groupby("zone")
        .agg(
            candidate_count=("candidate_id", "count"),
            avg_households=("households", "mean"),
            avg_income=("income_index", "mean"),
            avg_competitors=("competitors_1km", "mean"),
            avg_cannibalization=("cannibalization_risk", "mean"),
            total_potential_sales=("net_monthly_sales", "sum"),
        )
        .reset_index()
    )
    zone_capacity["market_hold_capacity"] = np.clip(
        (
            zone_capacity["avg_households"] / 5200
            + zone_capacity["avg_income"] / 38
            - zone_capacity["avg_competitors"] / 5.8
            - zone_capacity["avg_cannibalization"] * 1.8
        ).round(0),
        0,
        9,
    ).astype(int)

    return candidates, zone_capacity


def apply_filters(candidates, selected_formats, zones, min_score):
    filtered = candidates.copy()
    if selected_formats:
        filtered = filtered[filtered["format"].isin(selected_formats)]
    if zones:
        filtered = filtered[filtered["zone"].isin(zones)]
    filtered = filtered[filtered["pli_score"] >= min_score]
    return filtered


# -----------------------------
# Carga de datos
# -----------------------------
current_stores, candidate_sites = generate_synthetic_network()
model, FEATURES, metrics, importance = train_model(current_stores)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://dummyimage.com/280x72/0f172a/ffffff.png&text=BDS+%7C+PLI",
    use_container_width=True,
)
st.sidebar.title("Parámetros de decisión")
st.sidebar.caption("Ajusta supuestos para simular decisiones de apertura.")

selected_formats = st.sidebar.multiselect(
    "Formatos a evaluar",
    sorted(candidate_sites["format"].unique()),
    default=sorted(candidate_sites["format"].unique()),
)

zones = st.sidebar.multiselect(
    "Zonas",
    sorted(candidate_sites["zone"].unique()),
    default=[],
)

min_score = st.sidebar.slider("Score PLI mínimo", 0, 100, 45, 1)

gross_margin = st.sidebar.slider("Margen bruto estimado", 0.20, 0.65, 0.38, 0.01)
base_fixed_cost = st.sidebar.number_input(
    "Costo fijo base mensual",
    min_value=0,
    value=120_000,
    step=10_000,
)
labor_cost = st.sidebar.number_input(
    "Costo laboral mensual",
    min_value=0,
    value=95_000,
    step=5_000,
)
capex_base = st.sidebar.number_input(
    "CAPEX base por tienda",
    min_value=0,
    value=1_150_000,
    step=50_000,
)
capex_per_m2 = st.sidebar.number_input(
    "CAPEX por m²",
    min_value=0,
    value=8_500,
    step=500,
)

min_roi = st.sidebar.slider("ROI 12 meses mínimo", 0.0, 2.0, 0.55, 0.05)
max_cannibalization = st.sidebar.slider("Canibalización máxima", 0.0, 0.85, 0.32, 0.01)
portfolio_size = st.sidebar.slider("Candidatos a aprobar en portafolio", 1, 12, 5, 1)

assumptions = {
    "gross_margin": gross_margin,
    "base_fixed_cost": base_fixed_cost,
    "labor_cost": labor_cost,
    "capex_base": capex_base,
    "capex_per_m2": capex_per_m2,
}

candidates, zone_capacity = enrich_candidates(current_stores, candidate_sites, model, FEATURES, assumptions)
candidates["decision"] = candidates.apply(lambda r: decision_label(r, min_roi, max_cannibalization), axis=1)

filtered_candidates = apply_filters(candidates, selected_formats, zones, min_score)
portfolio = (
    filtered_candidates[filtered_candidates["decision"] == "Aprobar"]
    .sort_values(["pli_score", "roi_12m"], ascending=False)
    .head(portfolio_size)
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div style="padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(90deg, #0F172A, #1E293B); color: white;">
        <h1 style="margin-bottom: 0.2rem;">Predictive Location Intelligence</h1>
        <p style="font-size: 1.05rem; color: #CBD5E1; margin-top: 0;">
        Demo funcional para decidir dónde abrir, dónde corregir y dónde no invertir.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# -----------------------------
# KPIs
# -----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Candidatos evaluados", format_number(len(filtered_candidates)))

with kpi2:
    st.metric(
        "Aprobables",
        format_number((filtered_candidates["decision"] == "Aprobar").sum()),
    )

with kpi3:
    st.metric(
        "Venta mensual potencial neta",
        money_mxn(portfolio["net_monthly_sales"].sum()) if len(portfolio) else "$0 MXN",
    )

with kpi4:
    total_capex = portfolio["capex"].sum() if len(portfolio) else 0
    total_ebitda_12m = portfolio["monthly_ebitda"].sum() * 12 if len(portfolio) else 0
    portfolio_roi = total_ebitda_12m / total_capex if total_capex else 0
    st.metric("ROI 12m portafolio", pct(portfolio_roi))

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs(
    [
        "1. Mapa ejecutivo",
        "2. Ranking PLI",
        "3. Business case",
        "4. Canibalización",
        "5. Market hold capacity",
        "6. Modelo y backtesting",
        "7. Memo ejecutivo",
    ]
)

# -----------------------------
# Tab 1: Mapa
# -----------------------------
with tabs[0]:
    st.subheader("Mapa de red actual y ubicaciones candidatas")

    map_current = current_stores.copy()
    map_current["layer"] = "Red actual"
    map_current["size_metric"] = map_current["monthly_sales"] / 10000
    map_current["color_group"] = "Tienda actual"
    map_current["hover_text"] = (
        "Tienda: " + map_current["store_id"]
        + "<br>Zona: " + map_current["zone"]
        + "<br>Formato: " + map_current["format"]
        + "<br>Ventas mensuales: " + map_current["monthly_sales"].apply(money_mxn)
    )

    map_candidates = filtered_candidates.copy()
    map_candidates["layer"] = "Candidatos"
    map_candidates["size_metric"] = map_candidates["pli_score"]
    map_candidates["color_group"] = map_candidates["decision"]
    map_candidates["hover_text"] = (
        "Candidato: " + map_candidates["candidate_id"]
        + "<br>Zona: " + map_candidates["zone"]
        + "<br>Formato: " + map_candidates["format"]
        + "<br>Score PLI: " + map_candidates["pli_score"].astype(str)
        + "<br>Venta neta estimada: " + map_candidates["net_monthly_sales"].apply(money_mxn)
        + "<br>ROI 12m: " + map_candidates["roi_12m"].apply(pct)
        + "<br>Canibalización: " + map_candidates["cannibalization_risk"].apply(pct)
    )

    map_data = pd.concat(
        [
            map_current[
                ["lat", "lon", "zone", "format", "layer", "size_metric", "color_group", "hover_text"]
            ],
            map_candidates[
                ["lat", "lon", "zone", "format", "layer", "size_metric", "color_group", "hover_text"]
            ],
        ],
        ignore_index=True,
    )

    color_map = {
        "Tienda actual": "#64748B",
        "Aprobar": "#16A34A",
        "Revisar": "#D97706",
        "No aprobar": "#DC2626",
    }

    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        color="color_group",
        size="size_metric",
        hover_name="zone",
        hover_data={"hover_text": True, "lat": False, "lon": False, "size_metric": False},
        color_discrete_map=color_map,
        zoom=10,
        height=680,
        mapbox_style="open-street-map",
    )
    fig_map.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Tipo")
    st.plotly_chart(fig_map, use_container_width=True)

    st.info(
        "Lectura comercial: el mapa no solo ubica puntos; traduce cada ubicación en una recomendación económica sujeta a ROI, riesgo de canibalización y potencial de demanda."
    )

# -----------------------------
# Tab 2: Ranking
# -----------------------------
with tabs[1]:
    st.subheader("Ranking de ubicaciones candidatas")

    ranking_cols = [
        "candidate_id",
        "zone",
        "format",
        "pli_score",
        "decision",
        "predicted_monthly_sales",
        "net_monthly_sales",
        "monthly_ebitda",
        "roi_12m",
        "payback_months",
        "cannibalization_risk",
        "nearest_store_km",
        "competitors_1km",
    ]

    ranking = filtered_candidates[ranking_cols].sort_values(
        ["decision", "pli_score", "roi_12m"], ascending=[True, False, False]
    )

    styled = ranking.style.format(
        {
            "predicted_monthly_sales": "${:,.0f}",
            "net_monthly_sales": "${:,.0f}",
            "monthly_ebitda": "${:,.0f}",
            "roi_12m": "{:.1%}",
            "payback_months": "{:.1f}",
            "cannibalization_risk": "{:.1%}",
            "nearest_store_km": "{:.2f}",
        }
    ).background_gradient(subset=["pli_score"], cmap="Greens")

    st.dataframe(styled, use_container_width=True, height=460)

    st.divider()

    selected_candidate = st.selectbox(
        "Selecciona un candidato para análisis individual",
        filtered_candidates.sort_values("pli_score", ascending=False)["candidate_id"].tolist()
        if len(filtered_candidates)
        else [],
    )

    if selected_candidate:
        row = filtered_candidates[filtered_candidates["candidate_id"] == selected_candidate].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score PLI", f"{row['pli_score']:.1f}")
        c2.metric("Decisión", row["decision"])
        c3.metric("Venta neta mensual", money_mxn(row["net_monthly_sales"]))
        c4.metric("Payback", f"{row['payback_months']:.1f} meses" if not pd.isna(row["payback_months"]) else "No recupera")

        drivers = pd.DataFrame(
            {
                "Driver": [
                    "Tráfico",
                    "Hogares",
                    "Ingreso",
                    "Visibilidad",
                    "Transporte",
                    "Estacionamiento",
                    "Competencia",
                    "Renta",
                ],
                "Valor": [
                    row["foot_traffic"],
                    row["households"],
                    row["income_index"],
                    row["visibility"],
                    row["transit_score"],
                    row["parking_score"],
                    row["competitors_1km"],
                    row["rent_m2"],
                ],
            }
        )

        fig_drivers = px.bar(
            drivers,
            x="Valor",
            y="Driver",
            orientation="h",
            title=f"Lectura de drivers para {selected_candidate}",
        )
        fig_drivers.update_layout(height=420, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_drivers, use_container_width=True)

# -----------------------------
# Tab 3: Business Case
# -----------------------------
with tabs[2]:
    st.subheader("Business case del portafolio seleccionado")

    if len(portfolio) == 0:
        st.warning("No hay candidatos aprobables con los parámetros actuales. Reduce el score mínimo o relaja los criterios de ROI/canibalización.")
    else:
        total_sales = portfolio["net_monthly_sales"].sum()
        total_gp = portfolio["monthly_gross_profit"].sum()
        total_fixed = portfolio["monthly_fixed_cost"].sum()
        total_ebitda = portfolio["monthly_ebitda"].sum()
        total_capex = portfolio["capex"].sum()
        roi_12m = (total_ebitda * 12) / total_capex if total_capex else 0
        payback = total_capex / total_ebitda if total_ebitda > 0 else np.nan

        a, b, c, d = st.columns(4)
        a.metric("Venta mensual neta", money_mxn(total_sales))
        b.metric("EBITDA mensual", money_mxn(total_ebitda))
        c.metric("CAPEX total", money_mxn(total_capex))
        d.metric("Payback promedio", f"{payback:.1f} meses" if not pd.isna(payback) else "No recupera")

        st.write("")

        waterfall = go.Figure(
            go.Waterfall(
                name="Business Case",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Venta neta", "Margen bruto", "Costos fijos", "EBITDA mensual"],
                y=[total_sales, total_gp - total_sales, -total_fixed, total_ebitda],
                text=[money_mxn(total_sales), money_mxn(total_gp), money_mxn(total_fixed), money_mxn(total_ebitda)],
                textposition="outside",
            )
        )
        waterfall.update_layout(
            title="Puente económico mensual del portafolio",
            height=480,
            margin=dict(l=0, r=0, t=60, b=0),
        )
        st.plotly_chart(waterfall, use_container_width=True)

        st.markdown("#### Portafolio recomendado")
        cols = [
            "candidate_id",
            "zone",
            "format",
            "pli_score",
            "net_monthly_sales",
            "monthly_ebitda",
            "capex",
            "roi_12m",
            "payback_months",
            "cannibalization_risk",
        ]
        st.dataframe(
            portfolio[cols]
            .sort_values("pli_score", ascending=False)
            .style.format(
                {
                    "net_monthly_sales": "${:,.0f}",
                    "monthly_ebitda": "${:,.0f}",
                    "capex": "${:,.0f}",
                    "roi_12m": "{:.1%}",
                    "payback_months": "{:.1f}",
                    "cannibalization_risk": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

# -----------------------------
# Tab 4: Canibalización
# -----------------------------
with tabs[3]:
    st.subheader("Riesgo de canibalización")

    if len(filtered_candidates) == 0:
        st.warning("No hay candidatos con los filtros actuales.")
    else:
        risk_df = filtered_candidates.sort_values("cannibalization_risk", ascending=False).head(18)

        fig_risk = px.bar(
            risk_df,
            x="candidate_id",
            y="cannibalization_risk",
            color="decision",
            color_discrete_map={
                "Aprobar": "#16A34A",
                "Revisar": "#D97706",
                "No aprobar": "#DC2626",
            },
            hover_data=["zone", "nearest_store_id", "nearest_store_km", "net_monthly_sales", "roi_12m"],
            title="Candidatos con mayor riesgo de canibalización",
        )
        fig_risk.update_layout(height=480, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("#### Lectura operativa")
        st.write(
            "Una ubicación puede tener alta venta esperada y aun así destruir valor si transfiere ventas de una tienda actual. "
            "La decisión correcta no es maximizar ventas brutas, sino ventas incrementales netas de canibalización."
        )

        st.dataframe(
            risk_df[
                [
                    "candidate_id",
                    "zone",
                    "format",
                    "nearest_store_id",
                    "nearest_store_km",
                    "nearest_store_sales",
                    "cannibalization_risk",
                    "predicted_monthly_sales",
                    "net_monthly_sales",
                    "decision",
                ]
            ].style.format(
                {
                    "nearest_store_km": "{:.2f}",
                    "nearest_store_sales": "${:,.0f}",
                    "cannibalization_risk": "{:.1%}",
                    "predicted_monthly_sales": "${:,.0f}",
                    "net_monthly_sales": "${:,.0f}",
                }
            ),
            use_container_width=True,
        )

# -----------------------------
# Tab 5: Market Hold Capacity
# -----------------------------
with tabs[4]:
    st.subheader("Market hold capacity por micromercado")

    zone_view = zone_capacity.sort_values("market_hold_capacity", ascending=False).copy()
    zone_view["total_potential_sales"] = zone_view["total_potential_sales"].round(0)

    fig_capacity = px.scatter(
        zone_view,
        x="avg_cannibalization",
        y="total_potential_sales",
        size="market_hold_capacity",
        color="market_hold_capacity",
        hover_name="zone",
        hover_data=["avg_households", "avg_income", "avg_competitors", "candidate_count"],
        title="Potencial económico vs. saturación esperada",
    )
    fig_capacity.update_layout(
        height=520,
        xaxis_tickformat=".0%",
        yaxis_tickprefix="$",
        margin=dict(l=0, r=0, t=60, b=0),
    )
    st.plotly_chart(fig_capacity, use_container_width=True)

    st.dataframe(
        zone_view[
            [
                "zone",
                "market_hold_capacity",
                "candidate_count",
                "avg_households",
                "avg_income",
                "avg_competitors",
                "avg_cannibalization",
                "total_potential_sales",
            ]
        ].style.format(
            {
                "avg_households": "{:,.0f}",
                "avg_income": "{:.1f}",
                "avg_competitors": "{:.1f}",
                "avg_cannibalization": "{:.1%}",
                "total_potential_sales": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.info(
        "La capacidad de mercado es una estimación ejecutiva: combina demanda local, ingreso, competencia y riesgo de canibalización para anticipar cuántas unidades adicionales caben sin erosionar valor."
    )

# -----------------------------
# Tab 6: Modelo
# -----------------------------
with tabs[5]:
    st.subheader("Modelo predictivo y backtesting")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² test", f"{metrics['r2']:.2f}")
    m2.metric("MAPE test", f"{metrics['mape']:.1%}")
    m3.metric("Tiendas históricas", format_number(len(current_stores)))

    st.write(
        "El demo entrena un modelo con tiendas actuales sintéticas para pronosticar ventas mensuales esperadas. "
        "En un proyecto real se reemplaza por POSAR del cliente: ventas, transacciones, ticket, inventario, tráfico, competencia, entorno sociodemográfico y variables operativas."
    )

    fig_imp = px.bar(
        importance.sort_values("importance", ascending=True),
        x="importance",
        y="driver",
        orientation="h",
        title="Importancia relativa de drivers en el modelo",
    )
    fig_imp.update_layout(height=500, margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### POSAR mínimo requerido para versión real")
    posar = pd.DataFrame(
        {
            "Bloque": [
                "Ventas y transacciones",
                "Ubicación y trade area",
                "Demografía y economía local",
                "Competencia",
                "Operación de tienda",
                "Inmueble y CAPEX",
                "Historial de decisiones",
            ],
            "Ejemplos": [
                "Venta, ticket, unidades, margen, devoluciones, horarios",
                "Lat/lon, isócronas, accesibilidad, visibilidad, estacionamiento",
                "Hogares, ingreso, población flotante, empleo, NSE",
                "Competidores, sustitutos, intensidad promocional, distancia",
                "Fill rate, quiebres, plantilla, merma, productividad",
                "Renta, m², adecuación, equipamiento, restricciones",
                "Aperturas, cierres, reubicaciones, forecast original y resultado",
            ],
        }
    )
    st.dataframe(posar, use_container_width=True, hide_index=True)

# -----------------------------
# Tab 7: Memo
# -----------------------------
with tabs[6]:
    st.subheader("Memo ejecutivo de decisión")

    if len(portfolio) == 0:
        st.warning("No hay portafolio aprobable con los parámetros actuales.")
    else:
        top_zones = ", ".join(portfolio["zone"].value_counts().head(3).index.tolist())
        approved_ids = ", ".join(portfolio["candidate_id"].tolist())

        total_sales = portfolio["net_monthly_sales"].sum()
        total_ebitda = portfolio["monthly_ebitda"].sum()
        total_capex = portfolio["capex"].sum()
        roi_12m = (total_ebitda * 12) / total_capex if total_capex else 0
        avg_cann = portfolio["cannibalization_risk"].mean()
        avg_score = portfolio["pli_score"].mean()

        memo = f"""
### Recomendación

Con los supuestos actuales, el portafolio recomendado incluye **{len(portfolio)} ubicaciones**: **{approved_ids}**.

La inversión estimada es de **{money_mxn(total_capex)}** y genera una venta mensual neta estimada de **{money_mxn(total_sales)}**. El EBITDA mensual estimado asciende a **{money_mxn(total_ebitda)}**, con un ROI a 12 meses de **{pct(roi_12m)}**.

### Racional de decisión

El portafolio se concentra en **{top_zones}**, con un score PLI promedio de **{avg_score:.1f}** y una canibalización promedio de **{pct(avg_cann)}**. La recomendación no se basa en venta bruta, sino en venta incremental neta después de riesgo de canibalización, costo fijo, renta, CAPEX y potencial de micromercado.

### Guardrails

La aprobación debe condicionarse a validación inmobiliaria, restricciones operativas, negociación de renta, verificación de competencia en campo y revisión de capacidad logística. Cualquier ubicación con canibalización mayor a **{pct(max_cannibalization)}** debe regresar al comité salvo que exista una razón estratégica explícita.

### Próxima decisión

Ejecutar visita de campo y due diligence comercial para los candidatos aprobados. Después de abrir, medir resultado real contra pronóstico para recalibrar el modelo y elevar el hit-rate de futuras decisiones.
        """

        st.markdown(memo)

        st.download_button(
            label="Descargar memo en Markdown",
            data=memo.encode("utf-8"),
            file_name=f"memo_pli_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

st.caption(
    "Demo con datos sintéticos. PLI® by Business Data Scientists: modelos, gobierno de decisión y arquitectura analítica para redes físicas."
)
