import json, os, tempfile, uuid
from pathlib import Path

import altair as alt, pandas as pd, streamlit as st
from sklearn.preprocessing import LabelEncoder

from app.ui_helpers import STREAMLIT_THEME
from backend.main import _build_pipeline_result

MAX_FILE_SIZE_MB = 10


def _read_uploaded_file(uploaded_file):
    raw = uploaded_file.read(); uploaded_file.seek(0)
    if not raw: raise ValueError("Uploaded file is empty. Please select a valid dataset file.")
    buf = pd.io.common.BytesIO(raw); name = uploaded_file.name.lower()
    if name.endswith(".csv"): return pd.read_csv(buf)
    if name.endswith((".xlsx", ".xls")): return pd.read_excel(buf)
    if name.endswith(".json"): return pd.read_json(buf)
    raise ValueError("Unsupported file type. Please upload CSV, Excel, or JSON.")


def _make_temp_copy(uploaded_file):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".csv") as tmp:
        tmp.write(uploaded_file.read()); return tmp.name


def _detect_default_target(columns):
    norm = {str(c).strip().lower(): c for c in columns}
    for key in ["survived", "target", "label", "class", "y", "loan_status", "income", "outcome", "status"]:
        if key in norm: return norm[key]
    for cand in ["survived", "target", "label", "class", "y"]:
        for c in columns:
            if cand in str(c).lower(): return c
    return columns[-1] if columns else None


def _run_pipeline(file_path, target_column, loader_placeholder):
    state = {"value": 0, "message": "Starting pipeline"}
    def cb(progress, message):
        state["value"] = progress; state["message"] = message
        with loader_placeholder.container():
            st.progress(max(0, min(progress, 100)))
            st.write(f"Status: {message}")
    return _build_pipeline_result(job_id=str(uuid.uuid4()), file_path=file_path, target_column=target_column, progress_callback=cb), state


st.set_page_config(page_title="AutoML Studio", page_icon="🤖", layout="wide")
st.markdown(STREAMLIT_THEME, unsafe_allow_html=True)
st.markdown("""<div style="display:flex;align-items:center;justify-content:space-between;margin:0 0 0.3rem 0;gap:12px"><div><h1 style="margin:0;font-size:3rem;font-weight:800;line-height:1.08">AutoML Studio</h1><div style="color:#94a3b8;font-size:1.1rem;margin-top:0.2rem">Train smarter. Predict faster.</div></div><div class="status-pill"><span class="status-dot"></span>Live</div></div>""", unsafe_allow_html=True)
main_status_placeholder = st.empty()
show_upload_form = "result" not in st.session_state and not st.session_state.get("pipeline_running", False)
if show_upload_form:
    left_col, right_col = st.columns([1, 1.5], gap="large", vertical_alignment="center")
    with left_col:
        st.markdown("### Upload Your Dataset")
        st.markdown("Train smarter by uploading a **CSV, Excel, or JSON** file. AutoML handles cleaning, training, and evaluation automatically.")
    with right_col:
        uploaded_file = st.file_uploader(
            "Drop a file here or click to browse — **Max 10 MB · CSV / XLSX / JSON**",
            type=["csv", "xlsx", "xls", "json"],
        )
else:
    uploaded_file = None
if uploaded_file is not None and show_upload_form:
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."); st.stop()
    df = _read_uploaded_file(uploaded_file); columns = list(df.columns)
    default_target = _detect_default_target(columns)
    target_index = columns.index(default_target) if default_target in columns else max(0, len(columns)-1)
    target_column = st.selectbox("Target column", columns, index=target_index)
    if st.button("Run AutoML Pipeline", type="primary"):
        st.session_state["pipeline_running"] = True; temp_path = _make_temp_copy(uploaded_file)
        st.session_state["uploaded_df"] = df; st.session_state["uploaded_file"] = uploaded_file; st.session_state["last_temp_file_path"] = temp_path; st.session_state["selected_target_column"] = target_column
        with main_status_placeholder.container():
            st.write("Training models and tuning the best candidates..."); st.progress(0)
        try:
            result, progress_holder = _run_pipeline(temp_path, target_column, main_status_placeholder)
            st.session_state["result"] = result; st.session_state["status"] = progress_holder; main_status_placeholder.empty()
        except Exception as exc: st.error(str(exc))
        finally:
            st.session_state["pipeline_running"] = False
            if os.path.exists(temp_path): os.remove(temp_path)


def _format_metric_value(value):
    if value is None: return "—"
    if isinstance(value, float): return f"{value:.4f}"
    return str(value)


def _short_reason(value):
    if value is None: return "—"
    text = " ".join(str(value).replace("because", "").split())
    return text[:87] + "..." if len(text) > 90 else text


def _humanize_feature_name(name):
    text = " ".join(str(name).replace("_", " ").split())
    return text.title() if text else "Feature"


def _numeric_summary(series):
    if series.empty: return None
    values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    if values.empty or not values.notna().all(): return None
    return float(values.min()), float(values.max()), float(values.median())


def _format_feature_help(feature, uploaded_df=None):
    if uploaded_df is None or feature not in uploaded_df.columns: return "Enter a value for this feature."
    key = str(feature).strip().lower()
    if key in {"holiday_flag", "is_holiday", "is_weekend"}: return "Binary flag: 0 = No, 1 = Yes."
    if key in {"store", "store_id"}: return "Numeric store ID. Enter a valid store number from the dataset."
    series = uploaded_df[feature].dropna(); numeric = _numeric_summary(series)
    if numeric is not None:
        minimum, maximum, median = numeric
        return f"Value should be {minimum}." if minimum == maximum else f"Typical range: {minimum} to {maximum}. Suggested value: {median:.2f}."
    choices = [str(item) for item in series.astype(str).drop_duplicates().tolist()]
    return f"Choose one of: {', '.join(choices)}." if len(choices) <= 5 else "Enter a known category value for this feature."


def _get_feature_value_options(feature, uploaded_df=None):
    if uploaded_df is None or feature not in uploaded_df.columns: return {"type": "text", "choices": []}
    series = uploaded_df[feature].dropna(); numeric = _numeric_summary(series)
    if numeric is not None:
        values = pd.to_numeric(series.astype(str).str.strip(), errors="coerce").dropna()
        return {"type": "numeric", "values": values.astype(float).tolist()}
    choices = [str(item) for item in series.astype(str).drop_duplicates().tolist()]
    if not choices: return {"type": "text", "choices": []}
    encoder = LabelEncoder(); encoder.fit(choices)
    mapping = {label: int(code) for label, code in zip(choices, encoder.transform(choices), strict=False)}
    return {"type": "categorical", "choices": choices, "mapping": mapping}


def _coerce_prediction_value(value, feature_options):
    selected_value = value.get("value") if isinstance(value, dict) else value
    mapping = value.get("mapping", {}) if isinstance(value, dict) else feature_options.get("mapping", {})
    if feature_options.get("type") == "numeric":
        try: return float(selected_value)
        except: return 0.0
    if feature_options.get("type") == "categorical":
        if selected_value is None: return 0.0
        try: return float(selected_value)
        except (TypeError, ValueError):
            if isinstance(selected_value, str):
                normalized = selected_value.strip()
                if normalized.lower() in {"nan", "none", "null", ""}: return 0.0
                if normalized in mapping: return float(mapping[normalized])
                return float(mapping.get(str(normalized), 0))
            return 0.0
    if selected_value is None: return 0.0
    try: return float(selected_value)
    except: return 0.0


def _summarize_uniform_mapping(mapping):
    if not isinstance(mapping, dict) or not mapping: return None
    values = [str(v).strip() for v in mapping.values() if v not in (None, "", "None")]
    if not values: return None
    unique = list(dict.fromkeys(values))
    if len(unique) != 1: return None
    value = unique[0].lower()
    return {"standard scaling": "All columns have attained standard scaling.", "min-max scaling": "All columns have attained min-max scaling."}.get(value, f"All columns have attained {value}.")


def _render_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics, strict=False):
        with col:
            st.markdown(f"""<div class="glass-panel" style="min-height:120px;display:flex;flex-direction:column;justify-content:center"><div style="font-size:12px;color:#a8b3c7;margin-bottom:12px;letter-spacing:0.04em;text-transform:uppercase">{label}</div><div style="font-size:clamp(1.2rem,2vw,2rem);font-weight:700;color:#f8fafc;word-break:break-word;line-height:1.2">{_format_metric_value(value)}</div></div>""", unsafe_allow_html=True)


def _render_card_list(title, items):
    items = items or ["None"]
    st.markdown(f"<div class='glass-panel' style='padding:1rem 1.1rem;margin:0.6rem 0'><div style='font-size:0.86rem;color:#a8b3c7;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.7rem'>{title}</div><ul style='margin:0;padding-left:1.1rem;color:#e2e8f0;line-height:1.8'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul></div>", unsafe_allow_html=True)


def _render_preprocessing(preprocessing):
    if not preprocessing: st.info("No preprocessing details available."); return
    for section_name, section_value in preprocessing.items():
        if isinstance(section_value, dict):
            if not section_value:
                _render_card_list(section_name.replace('_', ' ').title(), ["None"])
                continue
            if section_name == "scaling_report":
                summary = _summarize_uniform_mapping(section_value)
                if summary:
                    _render_card_list(section_name.replace('_', ' ').title(), [summary])
                    continue
            entries = [f"{key}: {value if value not in (None, 'None', '') else 'None'}" for key, value in section_value.items()]
            _render_card_list(section_name.replace('_', ' ').title(), entries)
        elif isinstance(section_value, list):
            if not section_value:
                _render_card_list(section_name.replace('_', ' ').title(), ["None"])
            else:
                unique = list(dict.fromkeys(str(i) for i in section_value))
                statement = "All: " + unique[0] if len(unique) == 1 else ", ".join(unique)
                _render_card_list(section_name.replace('_', ' ').title(), [statement])
        else:
            _render_card_list(section_name.replace('_', ' ').title(), [str(section_value if section_value not in (None, 'None', '') else 'None')])


def _render_prediction_form(best_model, feature_names, uploaded_df):
    if best_model is None or not feature_names: st.info("Prediction model is not available yet."); return
    with st.form("prediction_form"):
        inputs = {}
        columns = st.columns(2)
        for i, feature in enumerate(feature_names):
            with columns[i % 2]:
                label = _humanize_feature_name(feature)
                help_text = _format_feature_help(feature, uploaded_df)
                options = _get_feature_value_options(feature, uploaded_df)
                st.markdown(f"<div class='prediction-card'><div class='field-label'>{label}</div>", unsafe_allow_html=True)
                if options["type"] == "numeric":
                    values = pd.Series(options["values"], dtype=float)
                    default_value = float(values.median()) if not values.empty else 0.0
                    min_value = float(values.min()) if not values.empty else None; max_value = float(values.max()) if not values.empty else None
                    if min_value is not None and max_value is not None and min_value != max_value:
                        inputs[feature] = st.number_input(label, value=default_value, min_value=min_value, max_value=max_value, step=(max_value - min_value)/100 if max_value > min_value else 1.0, help=help_text, label_visibility="collapsed")
                    else: inputs[feature] = st.number_input(label, value=default_value, step=1.0, help=help_text, label_visibility="collapsed")
                elif options["type"] == "categorical":
                    choices = options["choices"]
                    if len(choices) <= 15: inputs[feature] = {"value": st.selectbox(label, choices, index=0, help=help_text, label_visibility="collapsed"), "mapping": options["mapping"]}
                    else: inputs[feature] = {"value": st.text_input(label, placeholder="Enter category value", help=help_text, label_visibility="collapsed"), "mapping": options["mapping"]}
                else: inputs[feature] = st.text_input(label, placeholder="Enter value", help=help_text, label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
        if st.form_submit_button("Predict", type="primary"):
            row = {f: _coerce_prediction_value(v, _get_feature_value_options(f, uploaded_df)) for f, v in inputs.items()}
            prediction_df = pd.DataFrame([row], columns=feature_names).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            st.success(f"Prediction: {best_model.predict(prediction_df)[0]}")


if "result" in st.session_state:
    result = st.session_state["result"]
    st.markdown("""<div style="display:flex;align-items:center;justify-content:space-between;margin:1rem 0 1.5rem 0;gap:12px;flex-wrap:wrap"><div style="font-size:1.6rem;font-weight:700;color:#f8fafc">Overview</div><div style="padding:6px 12px;border-radius:999px;background:rgba(59,130,246,0.12);border:1px solid rgba(96,165,250,0.25);color:#bfdbfe;font-size:0.78rem;font-weight:600">Live metrics</div></div>""", unsafe_allow_html=True)
    summary = result.get("dashboard_summary", {})
    metrics = [("Problem type", summary.get("problem_type", result.get("problem_type", "-"))), ("Rows", summary.get("rows", result.get("dataset", {}).get("rows", 0))), ("Columns", summary.get("columns", result.get("dataset", {}).get("columns", 0))), ("Missing values", summary.get("missing_values", result.get("dataset", {}).get("missing_values_total", 0))), ("Best model", summary.get("best_model", result.get("best_model_name", "-"))), (summary.get("best_metric_label", result.get("primary_metric", "Metric")), summary.get("best_metric_value", "-"))]
    _render_metric_cards(metrics); st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Results", "Preprocessing", "Tuning summary", "Predict"])
    with tabs[0]:
        comparison = pd.DataFrame(result.get("comparison", []))
        if not comparison.empty:
            display_columns = ["Model"] + [c for c in comparison.columns if c != "Model"]
            metric_table = comparison[display_columns].copy(); st.dataframe(metric_table, use_container_width=True, hide_index=True)
            st.markdown("### Model comparison")
            st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
            chart_series = metric_table.set_index("Model")[result.get("primary_metric", "Accuracy")] if result.get("primary_metric", "Accuracy") in metric_table.columns else metric_table.set_index("Model")
            if isinstance(chart_series, pd.DataFrame): chart_series = chart_series.iloc[:, 0]
            chart_df = chart_series.reset_index()
            chart_df.columns = ["Model", "Value"]
            if not chart_df.empty:
                chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
                min_value, max_value = float(chart_df["Value"].min()), float(chart_df["Value"].max())
                x_axis = alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelColor="#e2e8f0", labelAngle=-45, labelFontSize=12, labelFontWeight=600, labelPadding=8))
                bars = alt.Chart(chart_df).mark_bar(opacity=0.92, stroke="rgba(15, 23, 42, 0.8)", strokeWidth=1).encode(x=x_axis, y=alt.Y("Value:Q", title=result.get("primary_metric", "Accuracy"), axis=alt.Axis(labelColor="#e2e8f0", titleColor="#94a3b8"), scale=alt.Scale(domain=[max(0.0, min_value - 0.05), max(1.0, max_value + 0.05)])), color=alt.Color("Model:N", legend=None, scale=alt.Scale(range=["#7dd3fc", "#34d399", "#fbbf24", "#fca5a5", "#c4b5fd", "#a5b4fc", "#f9a8d4", "#fdba74"])), tooltip=["Model:N", "Value:Q"]).properties(height=280)
                labels = alt.Chart(chart_df).mark_text(dy=-8, color="#ffffff", fontSize=11, fontWeight=700).encode(x=alt.X("Model:N", sort=None, title=None), y=alt.Y("Value:Q", axis=None), text=alt.Text("Value:Q", format=".4f"))
                chart = (bars + labels).properties(padding={"bottom": 100})
                st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        feature_importance = result.get("feature_importance", {})
        if feature_importance:
            st.markdown("### Feature importance")
            model_name = next(iter(feature_importance)); rows = feature_importance[model_name]
            if rows:
                feature_df = pd.DataFrame(rows); chart_data = feature_df.set_index("feature")["importance"]
                if chart_data.nunique() > 1:
                    ratio = chart_data.max() / chart_data.min() if chart_data.min() > 0 else 1
                    if ratio < 3: chart_data = (chart_data - chart_data.min()) / (chart_data.max() - chart_data.min() + 1e-9) * 100
                st.bar_chart(chart_data)
    with tabs[1]: _render_preprocessing(result.get("preprocessing", {}))
    with tabs[2]:
        tuning_summary = result.get("tuning_summary", {})
        if tuning_summary:
            tuning_df = pd.DataFrame.from_dict(tuning_summary, orient="index").reset_index().rename(columns={"index": "Model"})
            keep = [c for c in ["Model", "was_tuned", "baseline_cv_score", "tuned_cv_score"] if c in tuning_df.columns]
            display_df = tuning_df[keep].rename(columns={"was_tuned": "Tuned", "baseline_cv_score": "CV Before", "tuned_cv_score": "CV After"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            tuned_rows = tuning_df[tuning_df.get("was_tuned", pd.Series(dtype=bool)) == True] if "was_tuned" in tuning_df.columns else pd.DataFrame()
            if not tuned_rows.empty and "best_params" in tuning_df.columns:
                st.markdown("#### Best Parameters (Tuned Models)")
                for _, row in tuned_rows.iterrows():
                    params = row.get("best_params")
                    if params and params not in (None, "None", "nan", {}):
                        if isinstance(params, str):
                            try: params = json.loads(params)
                            except Exception: pass
                        with st.expander(f"📌 {row['Model']}  —  CV: {row.get('baseline_cv_score', '?')} → {row.get('tuned_cv_score', '?')}"):
                            if isinstance(params, dict):
                                for k, v in params.items():
                                    st.markdown(f"**{k}**: `{v}`")
                            else:
                                st.code(str(params))
        else: st.info("No tuning summary available.")
    with tabs[3]: _render_prediction_form(result.get("best_model_object"), result.get("selected_feature_names") or result.get("feature_names") or [], st.session_state.get("uploaded_df"))

