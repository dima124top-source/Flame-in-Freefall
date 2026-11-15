from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from indexer import get_parent_retriever, build_index, PDF_DIR

# Конфигурация страницы
st.set_page_config(
    page_title="FLAME IN FREEFALL // Прогнозирование горения и ИИ-эксперт 0g",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Стилизация терминала NASA / SI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    .si-card {
        background-color: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .si-analysis-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 20px;
        border-radius: 6px;
        font-size: 1.02rem;
        line-height: 1.65;
        color: #F1F5F9;
        text-align: justify;
    }
    /* Стилизация st.radio под вкладки */
    div.row-widget.stRadio > div {
        background-color: #131B2E;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Справочник физико-химических констант полимеров (NASA BASS-II, Saffire, ACME, SoFIE)
MATERIAL_DB: Dict[str, Dict[str, Any]] = {
    "PMMA": {
        "full_name": "Полиметилметакрилат (PMMA / Cast Acrylic)",
        "formula": "(C5O2H8)n",
        "elements": {"C", "H", "O"},
        "allowed_elements": "C (углерод), H (водород), O (кислород)",
        "flag": "has_pmma",
        "moc_0g": 17.3,
        "min_flow_cms": 0.4,
        "melt_hazard": 8.5,
        "toxicity_hazard": 4.5,
        "base_spread_rate": 1.45,
        "is_meltable": True,
        "is_char_wick": False,
        "facility": "BASS-II / Saffire / SoFIE",
        "suppression_notes": "При кипении расплава в 0g возможен выброс горящих капель. Требуется отключение вентиляции + инертизация (N2/CO2) или тонкораспыленный водяной туман (Fine Water Mist).",
    },
    "Cotton": {
        "full_name": "Хлопок / Целлюлозная ткань (Cotton / SIBTEX)",
        "formula": "(C6H10O5)n",
        "elements": {"C", "H", "O"},
        "allowed_elements": "C (углерод), H (водород), O (кислород)",
        "flag": "has_cotton",
        "moc_0g": 16.5,
        "min_flow_cms": 0.3,
        "melt_hazard": 2.0,
        "toxicity_hazard": 5.5,
        "base_spread_rate": 2.30,
        "is_meltable": False,
        "is_char_wick": True,
        "facility": "Saffire-I / Saffire-IV / BASS-II",
        "suppression_notes": "Высокий риск перехода в беспламенное тление (smoldering) после сбивания открытого пламени. Обязателен тепловизионный контроль и изоляция очага.",
    },
    "PVC": {
        "full_name": "Поливинилхлорид (PVC)",
        "formula": "(C2H3Cl)n",
        "elements": {"C", "H", "Cl"},
        "allowed_elements": "C (углерод), H (водород), Cl (хлор)",
        "flag": "has_pvc",
        "moc_0g": 23.0,
        "min_flow_cms": 1.2,
        "melt_hazard": 7.0,
        "toxicity_hazard": 9.5,
        "base_spread_rate": 0.85,
        "is_meltable": True,
        "is_char_wick": False,
        "facility": "BASS-II / ACME",
        "suppression_notes": "Критическая опасность выделения коррозионно-активного и токсичного HCl. Экипаж обязан немедленно надеть изолирующие дыхательные аппараты (PBA) и включить скрубберы.",
    },
    "Nomex": {
        "full_name": "Номекс (Nomex / мета-арамид)",
        "formula": "(C14H10N2O2)n",
        "elements": {"C", "H", "N", "O"},
        "allowed_elements": "C (углерод), H (водород), N (азот), O (кислород)",
        "flag": "has_nomex",
        "moc_0g": 26.5,
        "min_flow_cms": 2.5,
        "melt_hazard": 1.5,
        "toxicity_hazard": 6.5,
        "base_spread_rate": 0.35,
        "is_meltable": False,
        "is_char_wick": True,
        "facility": "BASS-II / Saffire",
        "suppression_notes": "Образует защитный коксовый слой (char), но в смеси с плавящимися полимерами работает как фитиль, удерживающий расплав в 0g.",
    },
    "PTFE": {
        "full_name": "Политетрафторэтилен (PTFE / Teflon)",
        "formula": "(C2F4)n",
        "elements": {"C", "F"},
        "allowed_elements": "C (углерод), F (фтор)",
        "flag": "has_ptfe",
        "moc_0g": 95.0,
        "min_flow_cms": 5.0,
        "melt_hazard": 4.0,
        "toxicity_hazard": 9.8,
        "base_spread_rate": 0.05,
        "is_meltable": True,
        "is_char_wick": False,
        "facility": "ACME / Wire Insulation Tests",
        "suppression_notes": "При внешнем нагреве от соседних горящих компонентов разлагается с выделением сверхтоксичных фторорганических соединений и HF (при наличии влаги).",
    },
}

ELEMENT_NAMES_RU = {
    "C": "C (углерод)",
    "H": "H (водород)",
    "O": "O (кислород)",
    "Cl": "Cl (хлор)",
    "N": "N (азот)",
    "F": "F (фтор)",
}

ALL_ELEMENTS = {"C", "H", "O", "Cl", "N", "F", "S", "Br"}
CSV_DIR = Path("./data/csv")


def extract_clean_text(content: Any) -> str:
    """Извлекает чистый текст из ответа LLM, отбрасывая signature и служебные блоки."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        clean_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                clean_parts.append(str(item["text"]))
            elif isinstance(item, str):
                clean_parts.append(item)
        return "".join(clean_parts)
    return str(content)


def build_mixture_profile(weights: Dict[str, float]) -> Dict[str, Any]:
    active = {k: v for k, v in weights.items() if v > 0}
    if not active:
        active = {"PMMA": 1.0}

    total_w = sum(active.values())
    norm_w = {k: v / total_w for k, v in active.items()}
    is_mixture = len(norm_w) > 1

    inv_moc = sum(w / MATERIAL_DB[k]["moc_0g"] for k, w in norm_w.items())
    effective_moc = round(1.0 / inv_moc, 1)

    min_flow = round(
        sum(w * MATERIAL_DB[k]["min_flow_cms"] for k, w in norm_w.items()) * (0.85 if is_mixture else 1.0),
        2,
    )

    has_meltable = any(MATERIAL_DB[k]["is_meltable"] for k in norm_w)
    has_wick = any(MATERIAL_DB[k]["is_char_wick"] for k in norm_w)
    wick_synergy = 1.28 if (is_mixture and has_meltable and has_wick) else 1.0

    base_spread = sum(w * MATERIAL_DB[k]["base_spread_rate"] for k, w in norm_w.items()) * wick_synergy

    melt_hazard = min(10.0, sum(w * MATERIAL_DB[k]["melt_hazard"] for k, w in norm_w.items()) * (1.15 if wick_synergy > 1.0 else 1.0))
    weighted_tox = sum(w * MATERIAL_DB[k]["toxicity_hazard"] for k, w in norm_w.items())
    peak_tox = max(MATERIAL_DB[k]["toxicity_hazard"] * (0.65 + 0.35 * w) for k, w in norm_w.items())
    toxicity_hazard = round(max(weighted_tox, peak_tox), 1)

    allowed_set = set()
    for k in norm_w:
        allowed_set |= MATERIAL_DB[k]["elements"]
    forbidden_set = sorted(list(ALL_ELEMENTS - allowed_set))

    allowed_str = ", ".join(ELEMENT_NAMES_RU[el] for el in sorted(list(allowed_set)) if el in ELEMENT_NAMES_RU)
    forbidden_str = (
        f"В составе отсутствуют: {', '.join(forbidden_set)}. Упоминание соединений с этими элементами строго запрещено."
        if forbidden_set
        else "В составе присутствуют все основные галогены и гетероатомы."
    )

    if not is_mixture:
        k = next(iter(norm_w))
        mat = MATERIAL_DB[k]
        display_name = mat["full_name"]
        formula_str = mat["formula"]
        facilities = mat["facility"]
        suppression = mat["suppression_notes"]
        flags = [mat["flag"]]
    else:
        comp_parts = [f"{k} ({w * 100:.0f}%)" for k, w in sorted(norm_w.items(), key=lambda x: -x[1])]
        display_name = "Смесь: " + " + ".join(comp_parts)
        formula_str = " + ".join(f"{w * 100:.0f}% {MATERIAL_DB[k]['formula']}" for k, w in norm_w.items())
        facilities = " / ".join(sorted(list({MATERIAL_DB[k]["facility"].split(" / ")[0] for k in norm_w})))
        suppression_list = [f"[{k}]: {MATERIAL_DB[k]['suppression_notes']}" for k in norm_w]
        if wick_synergy > 1.0:
            suppression_list.insert(
                0,
                "[СИНЕРГИЯ 0g — ФИТИЛЬНЫЙ ЭФФЕКТ]: Волокнистый каркас удерживает расплав в невесомости, препятствуя его отрыву и вызывая продолжительное двухфазное горение.",
            )
        suppression = " ".join(suppression_list)
        flags = [MATERIAL_DB[k]["flag"] for k in norm_w]

    return {
        "is_mixture": is_mixture,
        "weights": norm_w,
        "display_name": display_name,
        "formula": formula_str,
        "allowed_elements": allowed_str,
        "forbidden_notes": forbidden_str,
        "flags": flags,
        "moc_0g": effective_moc,
        "min_flow_cms": min_flow,
        "melt_hazard": round(melt_hazard, 1),
        "toxicity_hazard": toxicity_hazard,
        "base_spread_rate": round(base_spread, 3),
        "wick_synergy": wick_synergy,
        "facility": facilities,
        "suppression_notes": suppression,
    }


@st.cache_data(ttl=60)
def simulate_combustion_profile(
    weights_tuple: Tuple[Tuple[str, float], ...],
    o2_pct: float,
    flow_cms: float,
    pressure_kpa: float,
) -> pd.DataFrame:
    weights = dict(weights_tuple)
    profile = build_mixture_profile(weights)

    if not profile["is_mixture"]:
        single_key = next(iter(weights))
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        matching_csvs = list(CSV_DIR.glob(f"*{single_key.lower()}*.csv"))
        if matching_csvs:
            df = pd.read_csv(matching_csvs[0])
            if {"time_s", "spread_rate_mms", "flame_temp_k", "co_ppm"}.issubset(set(df.columns)):
                return df

    time_s = np.linspace(0, 120, 61)
    o2_factor = max(0.0, (o2_pct - (profile["moc_0g"] - 2.5)) / 10.0)

    if flow_cms < profile["min_flow_cms"]:
        flow_factor = 0.25 * (flow_cms / max(profile["min_flow_cms"], 0.1))
    else:
        flow_factor = 1.0 + 0.65 * np.log1p(flow_cms) * np.exp(-0.04 * flow_cms)

    pressure_factor = (pressure_kpa / 101.3) ** 0.45

    if o2_pct < profile["moc_0g"] and set(weights.keys()) == {"PTFE"}:
        spread_rate = np.full_like(time_s, 0.02 * pressure_factor)
        flame_temp = 300 + 420 * (1 - np.exp(-0.04 * time_s))
        co_ppm = 8 * profile["toxicity_hazard"] * (1 - np.exp(-0.03 * time_s))
    elif o2_pct < profile["moc_0g"]:
        decay = np.exp(-0.055 * time_s)
        spread_rate = profile["base_spread_rate"] * 0.42 * decay
        flame_temp = 300 + 680 * decay
        co_ppm = 18 + 50 * (1 - decay)
    else:
        transient = 1 - np.exp(-0.08 * time_s)
        secondary_wave = (1 + 0.16 * (1 - np.exp(-0.035 * np.maximum(0, time_s - 25)))) if profile["is_mixture"] else 1.0
        oscillation = 1 + 0.04 * np.sin(0.25 * time_s)

        spread_rate = (
            profile["base_spread_rate"]
            * o2_factor
            * flow_factor
            * pressure_factor
            * transient
            * secondary_wave
            * oscillation
        )
        flame_temp = 300 + (1150 + 28 * (o2_pct - 21.0)) * transient * (1.04 if profile["wick_synergy"] > 1.0 else 1.0)
        co_ppm = (24.0 * profile["toxicity_hazard"] * o2_factor * (1 + 0.3 / (flow_cms + 0.2))) * transient * secondary_wave

    return pd.DataFrame({
        "time_s": time_s,
        "spread_rate_mms": np.round(spread_rate, 3),
        "flame_temp_k": np.round(flame_temp, 1),
        "co_ppm": np.round(co_ppm, 1),
    })


def compute_profile_risk_index(
    profile: Dict[str, Any], o2_pct: float, flow_cms: float, pressure_kpa: float
) -> Tuple[float, str, str]:
    o2_margin = o2_pct - profile["moc_0g"]
    if not profile["is_mixture"] and "PTFE" in profile["weights"]:
        moc_score = 2.0 if o2_pct <= 24.0 else 3.5
    else:
        moc_score = np.clip(5.5 + o2_margin * 0.75, 1.0, 10.0)

    if flow_cms < profile["min_flow_cms"]:
        flow_score = 2.0
    elif 0.3 <= flow_cms <= 6.0:
        flow_score = 9.0
    else:
        flow_score = 7.2

    pressure_adj = 0.6 if (pressure_kpa < 75.0 and o2_pct > 25.0) else 0.0
    synergy_adj = 0.5 if profile["wick_synergy"] > 1.0 else 0.0

    raw_index = (
        0.36 * moc_score
        + 0.20 * flow_score
        + 0.22 * profile["melt_hazard"]
        + 0.22 * profile["toxicity_hazard"]
        + pressure_adj
        + synergy_adj
    )
    risk_index = float(np.clip(round(raw_index, 1), 1.0, 10.0))

    if risk_index < 4.5:
        return risk_index, "НОРМА", "#10B981"
    elif risk_index < 7.2:
        return risk_index, "ПОВЫШЕННЫЙ", "#F59E0B"
    else:
        return risk_index, "КРИТИЧЕСКИЙ", "#EF4444"


def build_risk_matrix_df(
    current_profile: Dict[str, Any], o2_pct: float, flow_cms: float, pressure_kpa: float
) -> pd.DataFrame:
    rows = []
    for key in MATERIAL_DB:
        p = build_mixture_profile({key: 1.0})
        idx, status, _ = compute_profile_risk_index(p, o2_pct, flow_cms, pressure_kpa)
        rows.append({
            "Объект / Материал": key,
            "Состав / Формула": p["formula"],
            "MOC в 0g (%)": p["moc_0g"],
            "Мин. поток (см/с)": p["min_flow_cms"],
            "Риск расплава (1-10)": p["melt_hazard"],
            "Токсичность (1-10)": p["toxicity_hazard"],
            "Индекс риска SI": idx,
            "Статус отсека": status,
        })

    if current_profile["is_mixture"]:
        idx_mix, status_mix, _ = compute_profile_risk_index(current_profile, o2_pct, flow_cms, pressure_kpa)
        rows.append({
            "Объект / Материал": f"★ {current_profile['display_name']}",
            "Состав / Формула": current_profile["formula"],
            "MOC в 0g (%)": current_profile["moc_0g"],
            "Мин. поток (см/с)": current_profile["min_flow_cms"],
            "Риск расплава (1-10)": current_profile["melt_hazard"],
            "Токсичность (1-10)": current_profile["toxicity_hazard"],
            "Индекс риска SI": idx_mix,
            "Статус отсека": status_mix,
        })

    return pd.DataFrame(rows).sort_values(by="Индекс риска SI", ascending=False).reset_index(drop=True)


def retrieve_nasa_context(
    query_text: str, profile: Dict[str, Any], isolate_material: bool = True
) -> Tuple[str, List[str]]:
    retriever = get_parent_retriever()

    if isolate_material:
        flags = profile["flags"]
        if len(flags) == 1:
            meta_filter = {flags[0]: True}
        else:
            meta_filter = {"$or": [{f: True} for f in flags]}
        retriever.search_kwargs = {"k": 4, "filter": meta_filter}
    else:
        retriever.search_kwargs = {"k": 4}

    retrieved_docs = retriever.invoke(query_text)

    sources_used = []
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        src = f"{doc.metadata.get('source_file', 'NASA_PSI_Report')} (стр. {doc.metadata.get('page', '?')})"
        if src not in sources_used:
            sources_used.append(src)
        context_blocks.append(f"[Фрагмент {i} | {src}]:\n{doc.page_content[:1200]}")

    rag_context = "\n\n".join(context_blocks) if context_blocks else (
        "Релевантные PDF-фрагменты не найдены в локальной базе. Опирайся на верифицированные физико-химические данные NASA PSI и текущую телеметрию."
    )
    return rag_context, sources_used


def create_llm(lm_url: str, model_name: str, api_key: str, temperature: float = 0.2):
    clean_model = model_name.strip()
    clean_key = api_key.strip()
    clean_url = lm_url.strip()

    if "googleapis.com" in clean_url or clean_key.startswith("AIza"):
        return ChatGoogleGenerativeAI(
            model=clean_model,
            google_api_key=clean_key,
            temperature=temperature,
            max_output_tokens=2500,
        )

    return ChatOpenAI(
        base_url=clean_url,
        api_key=clean_key,
        model=clean_model,
        temperature=temperature,
        max_tokens=2500,
        request_timeout=120,
    )


def build_expert_system_prompt(
    profile: Dict[str, Any],
    o2_pct: float,
    flow_cms: float,
    pressure_kpa: float,
    risk_index: float,
    status: str,
    telemetry_df: pd.DataFrame,
    is_report_mode: bool = False,
) -> str:
    """Формирует единый анти-галлюцинационный системный промпт для SI-отчета и диалогового ассистента."""
    mixture_rule = (
        "Анализируется МНОГОКОМПОНЕНТНАЯ СМЕСЬ: обязательно учитывай взаимное влияние компонентов "
        "(инициирование пиролиза более стойких полимеров, фитильный эффект при удержании расплава на волокнистом каркасе и суммирование токсичности)."
        if profile["is_mixture"]
        else "Анализируется однокомпонентный материал."
    )

    if is_report_mode:
        format_block = """### ФОРМАТ ВЫВОДА (ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ SI)
- Сформируй цельное экспертное заключение на основе текущих параметров симулятора и контекста NASA PSI.
- Логически покрой три аспекта (внутри связных академических абзацев, БЕЗ маркированных и нумерованных списков):
  1. Верифицированные данные NASA PSI по исследуемому составу (строго с проверкой источника).
  2. Физика процесса в текущих условиях симулятора (0g, состав атмосферы, поведение расплава/капель, синергия смеси).
  3. Протокол действий для оператора (отключение вентиляции, инертизация, выбор огнетушащего средства, СИЗОД).
- Пиши плотным техническим языком, без воды. Проверяй корректность склонения имен собственных и терминов."""
    else:
        format_block = """### ФОРМАТ ОТВЕТА
- Всегда опирайся на текущие параметры симулятора при оценке опасности.
- Структурируй ответ в виде трех четких блоков:
  1. **Анализ по данным NASA PSI / Историческая справка** (строго с проверкой источника).
  2. **Физика процесса в текущих условиях симулятора** (0g, состав атмосферы, поведение расплава/капель, синергия смеси).
  3. **Протокол действий для оператора** (конкретные шаги: отключение вентиляции, инертизация, выбор огнетушащего средства, СИЗОД).
- Пиши кратко, техническим языком, без воды. Проверяй корректность склонения имен собственных и терминов."""

    return f"""Ты — высокоточный инженер-эксперт по физике горения, материаловедению и пожарной безопасности пилотируемых космических аппаратов (МКС, Lunar Gateway, Orion). Твоя задача — анализировать параметры симулятора, отвечать на вопросы оператора и выдавать протоколы реагирования на русском языке на основе предоставленного контекста из базы данных NASA PSI и строгих законов физики.

### КРИТИЧЕСКИЕ ПРАВИЛА ДОСТОВЕРНОСТИ (ANTI-HALLUCINATION PROTOCOL)

1. **Строгое разделение источников информации:**
   - **Факты из контекста (NASA PSI):** Ссылайся на «базу данных NASA PSI», «загруженные отчеты» или конкретные эксперименты (FLEX, FLEX-2, BASS-II, Saffire, ACME, SoFIE) ТОЛЬКО в том случае, если эта информация явно присутствует в переданных тебе фрагментах текста (контексте).
   - **Общие инженерные и исторические знания:** Если вопрос касается исторических инцидентов («Аполлон-1», «Аполлон-13», станция «Мир») или базовой физики, которых нет в переданном контексте, отвечай на основе проверенных научных фактов, но **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** приписывать эти сведения базе данных NASA PSI.
   - **Отсутствие данных:** Если в контексте нет точных цифр, названий материалов или результатов эксперимента, прямо напиши: *«В предоставленных документах NASA PSI эти данные отсутствуют»*. Не додумывай детали, марки сплавов, химические составы или номера отчетов.

2. **Точность физической и химической терминологии:**
   - **Химическая строгость состава:** Активный объект — {profile['display_name']} ({profile['formula']}). Разрешенные элементы: {profile['allowed_elements']}. {profile['forbidden_notes']} КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО приписывать продуктам горения активного объекта элементы, отсутствующие в его составе.
   - **Не путай газы и механизмы их нейтрализации:** Четко разделяй угарный газ (CO — продукт неполного сгорания/пиролиза, удаляется каталитическим окислением или спецфильтрами) и углекислый газ (CO2 — продукт полного сгорания и дыхания, поглощается LiOH или цеолитами).
   - **Физика разбавителей и пламени:** Не приписывай инертным газам (N2, He) несуществующих свойств (например, N2 снижает температуру пламени как тепловой балласт за счет теплоемкости, а не за счет поглощения ИК-излучения). Не называй дефлаграцию (быстрое горение в O2) «сверхзвуковой скоростью» (сверхзвук — это детонация).
   - **Микрогравитация (0g) и смеси:** Учитывай отсутствие естественной конвекции. В 0g массоперенос окислителя определяется диффузией и принудительной вентиляцией. {mixture_rule}

### ТЕКУЩАЯ ТЕЛЕМЕТРИЯ ОТСЕКА СИМУЛЯТОРА (РЕЖИМ 0g)
- **Горючая нагрузка:** {profile['display_name']} | Формула: {profile['formula']}
- **Атмосфера:** O2 = {o2_pct:.1f}% (Эффективный MOC в 0g = {profile['moc_0g']}%), принудительная вентиляция = {flow_cms:.1f} см/с (порог устойчивости = {profile['min_flow_cms']} см/с), давление = {pressure_kpa:.1f} кПа.
- **Прогноз динамики:** Статус угрозы = {status} (Индекс риска SI = {risk_index}/10.0), пиковая скорость пламени = {telemetry_df['spread_rate_mms'].max():.2f} мм/с, пиковая температура = {telemetry_df['flame_temp_k'].max():.0f} K, прогноз CO = {telemetry_df['co_ppm'].max():.0f} ppm, коэффициент фитильной синергии = {profile['wick_synergy']:.2f}.
- **Регламент тушения материала:** {profile['suppression_notes']}

{format_block}"""


def query_si_rag_engine(
    profile: Dict[str, Any],
    o2_pct: float,
    flow_cms: float,
    pressure_kpa: float,
    risk_index: float,
    status: str,
    telemetry_df: pd.DataFrame,
    lm_url: str,
    model_name: str,
    api_key: str,
) -> Tuple[str, List[str]]:
    """Генерирует глубокое экспертное заключение SI по компоненту или смеси."""
    comp_names = " ".join(profile["weights"].keys())
    rag_context, sources_used = retrieve_nasa_context(
        f"{comp_names} composite mixture microgravity flame spread suppression NASA PSI",
        profile,
        isolate_material=True,
    )

    system_prompt = build_expert_system_prompt(
        profile=profile,
        o2_pct=o2_pct,
        flow_cms=flow_cms,
        pressure_kpa=pressure_kpa,
        risk_index=risk_index,
        status=status,
        telemetry_df=telemetry_df,
        is_report_mode=True,
    )

    user_prompt = f"""ДАННЫЕ ИЗ ОТЧЕТОВ NASA PSI (КОНТЕКСТ):
{rag_context}

Сформируй экспертное заключение SI по текущей телеметрии отсека и протокол действий для оператора."""

    llm = create_llm(lm_url, model_name, api_key, temperature=0.15)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return extract_clean_text(response.content), sources_used


def ask_si_assistant(
    user_question: str,
    chat_history: List[Dict[str, str]],
    profile: Dict[str, Any],
    o2_pct: float,
    flow_cms: float,
    pressure_kpa: float,
    risk_index: float,
    status: str,
    telemetry_df: pd.DataFrame,
    lm_url: str,
    model_name: str,
    api_key: str,
    strict_material_filter: bool,
) -> Tuple[str, List[str]]:
    """Диалоговый ИИ-помощник по физике горения компонентов и смесей в 0g."""
    comp_names = " ".join(profile["weights"].keys())
    rag_context, sources_used = retrieve_nasa_context(
        f"{user_question} {comp_names} microgravity combustion fire suppression",
        profile,
        isolate_material=strict_material_filter,
    )

    system_prompt = build_expert_system_prompt(
        profile=profile,
        o2_pct=o2_pct,
        flow_cms=flow_cms,
        pressure_kpa=pressure_kpa,
        risk_index=risk_index,
        status=status,
        telemetry_df=telemetry_df,
        is_report_mode=False,
    )

    messages = [SystemMessage(content=system_prompt)]

    history_slice = chat_history[-6:]
    if history_slice and history_slice[0]["role"] == "assistant":
        history_slice = history_slice[1:]

    for msg in history_slice:
        clean_msg = extract_clean_text(msg.get("content", ""))
        if msg["role"] == "user" and clean_msg:
            messages.append(HumanMessage(content=clean_msg))
        elif msg["role"] == "assistant" and clean_msg and not clean_msg.startswith("⚠️"):
            messages.append(AIMessage(content=clean_msg))

    augmented_user_msg = f"""Вопрос оператора: {user_question}

Релевантные выдержки из базы отчетов NASA PSI (КОНТЕКСТ):
{rag_context}"""

    messages.append(HumanMessage(content=augmented_user_msg))

    llm = create_llm(lm_url, model_name, api_key, temperature=0.2)
    response = llm.invoke(messages)
    return extract_clean_text(response.content), sources_used


# ==================== ИНТЕРФЕЙС STREAMLIT ====================

if "si_chat_history" not in st.session_state:
    st.session_state.si_chat_history = [
        {
            "role": "assistant",
            "content": (
                "Бортовой ИИ-эксперт **FLAME IN FREEFALL** активен. Система поддерживает анализ как **отдельных полимеров**, "
                "так и **многокомпонентных смесей** (кабельные сборки, композиты, ткани с полимерными покрытиями) в условиях микрогравитации ($0g$).\n\n"
                "Выберите компонент или задайте состав смеси в боковой панели слева, чтобы спрогнозировать динамику пламени, токсичность и тактику тушения."
            ),
            "sources": [],
        }
    ]

if st.session_state.pop("switch_to_chat", False):
    st.session_state["active_tab"] = "💬 ИИ-помощник: Физика горения и борьба с пожаром в 0g"


st.title("🛰️ FLAME IN FREEFALL // Прогнозирование горения и ИИ-эксперт 0g")
st.caption("Аналитическая платформа NASA Physical Sciences Informatics (BASS-II • Saffire • ACME • SoFIE) | Моделирование чистых полимеров и смесей")

# Боковая панель: Выбор компонента / смеси и параметры отсека
with st.sidebar:
    st.header("🧪 Состав горючей нагрузки")

    selection_mode = st.radio(
        "Режим выбора материала",
        options=["Один компонент", "Смесь компонентов"],
        horizontal=True,
    )

    component_weights: Dict[str, float] = {}

    if selection_mode == "Один компонент":
        selected_single = st.selectbox(
            "Выберите компонент",
            options=list(MATERIAL_DB.keys()),
            format_func=lambda k: f"{k} — {MATERIAL_DB[k]['formula']}",
        )
        component_weights = {selected_single: 100.0}
    else:
        selected_multi = st.multiselect(
            "Выберите компоненты смеси (2 и более)",
            options=list(MATERIAL_DB.keys()),
            default=["Cotton", "PMMA"],
            format_func=lambda k: f"{k} ({MATERIAL_DB[k]['formula']})",
        )
        if not selected_multi:
            st.warning("Выберите хотя бы один компонент. По умолчанию выбран PMMA.")
            selected_multi = ["PMMA"]

        st.caption("Задайте относительную массовую долю каждого компонента:")
        raw_weights = {}
        for comp in selected_multi:
            raw_weights[comp] = st.slider(
                f"Доля {comp}",
                min_value=5,
                max_value=95,
                value=int(100 / len(selected_multi)),
                step=5,
                key=f"weight_{comp}",
            )
        total_raw = sum(raw_weights.values())
        component_weights = {k: (v / total_raw) * 100.0 for k, v in raw_weights.items()}

        norm_summary = " | ".join(f"**{k}**: {w:.0f}%" for k, w in component_weights.items())
        st.info(f"Нормированный состав: {norm_summary}")

    st.divider()
    st.header("🎛️ Параметры атмосферы отсека")
    o2_conc = st.slider("Концентрация кислорода O₂ (%)", 15.0, 30.0, 21.0, 0.5)
    vent_flow = st.slider("Скорость принудительной вентиляции (см/с)", 0.0, 20.0, 2.5, 0.1)
    cabin_pressure = st.slider("Давление в отсеке (кПа)", 55.0, 105.0, 101.3, 0.5)

    st.divider()
    st.subheader("⚙️ Конфигурация LLM & RAG")
    lm_studio_url = st.text_input("Endpoint API (LM Studio / Cloud)", value="http://localhost:1234/v1")
    lm_model_name = st.text_input("Идентификатор модели", value="local-model")
    lm_api_key = st.text_input("API Key", value="lm-studio", type="password")

    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    st.caption(f"PDF-отчетов в базе `./data/pdf`: **{pdf_count}**")

    if st.button("🔄 Переиндексировать PDF (CPU)", use_container_width=True):
        with st.spinner("Парсинг PDF, авто-тегирование и построение эмбеддингов на CPU..."):
            indexed_pages = build_index(reset=True)
            if indexed_pages > 0:
                st.success(f"Готово! Проиндексировано страниц: {indexed_pages}")
            else:
                st.warning("Поместите PDF-отчеты NASA PSI в папку ./data/pdf")

# Расчет профиля выбранного компонента или смеси
active_profile = build_mixture_profile(component_weights)
risk_idx, threat_status, status_color = compute_profile_risk_index(
    active_profile, o2_conc, vent_flow, cabin_pressure
)
weights_tuple = tuple(sorted((k, round(v, 4)) for k, v in active_profile["weights"].items()))
telemetry_df = simulate_combustion_profile(weights_tuple, o2_conc, vent_flow, cabin_pressure)

# Верхний блок KPI и статуса угрозы
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Индекс пожарного риска SI",
        f"{risk_idx} / 10.0",
        delta=f"Эфф. MOC 0g: {active_profile['moc_0g']}%",
        delta_color="inverse",
    )
with col2:
    st.markdown(
        f"""
        <div class="si-card">
            <div style="font-size:0.85rem; color:#94A3B8;">Статус угрозы отсека</div>
            <div style="font-size:1.5rem; font-weight:700; color:{status_color};">{threat_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.metric(
        "Прогноз скорости пламени",
        f"{telemetry_df['spread_rate_mms'].max():.2f} мм/с",
        delta=f"Синергия: ×{active_profile['wick_synergy']:.2f}" if active_profile["is_mixture"] else f"Поток: {vent_flow:.1f} см/с",
    )
with col4:
    st.metric(
        "Прогноз температуры / CO",
        f"{telemetry_df['flame_temp_k'].max():.0f} K",
        delta=f"CO: {telemetry_df['co_ppm'].max():.0f} ppm",
        delta_color="inverse",
    )

st.divider()

selected_tab = st.radio(
    "Режим аналитики:",
    options=[
        "🔬 Прогноз горения компонента / смеси и отчет SI",
        "💬 ИИ-помощник: Физика горения и борьба с пожаром в 0g",
        "📊 Матрица ранжирования рисков (NASA PSI)",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)

if selected_tab == "🔬 Прогноз горения компонента / смеси и отчет SI":
    st.subheader(f"Прогноз динамики горения в 0g: {active_profile['display_name']}")
    st.caption(
        f"Состав: `{active_profile['formula']}` | Элементы: **{active_profile['allowed_elements']}** | "
        f"Установки NASA: **{active_profile['facility']}**"
    )

    if active_profile["is_mixture"] and active_profile["wick_synergy"] > 1.0:
        st.warning(
            "⚠️ **Обнаружен фитильный эффект (Wick Effect) в микрогравитации:** сочетание плавящегося полимера и волокнистого/коксообразующего каркаса "
            "удерживает расплав в зоне реакции при отсутствии гравитации, увеличивая скорость распространения пламени на 28%."
        )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_spread = px.line(
            telemetry_df,
            x="time_s",
            y="spread_rate_mms",
            title="Прогноз скорости распространения пламени (мм/с)",
            labels={"time_s": "Время от воспламенения (с)", "spread_rate_mms": "Скорость V_f (мм/с)"},
            template="plotly_dark",
        )
        fig_spread.update_traces(line=dict(color="#38BDF8", width=3))
        fig_spread.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_spread, use_container_width=True)

    with chart_col2:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=telemetry_df["time_s"],
                y=telemetry_df["flame_temp_k"],
                name="Температура пламени (K)",
                line=dict(color="#F97316", width=3),
            )
        )
        fig_temp.add_trace(
            go.Scatter(
                x=telemetry_df["time_s"],
                y=telemetry_df["co_ppm"],
                name="Генерация CO (ppm)",
                yaxis="y2",
                line=dict(color="#A855F7", width=2, dash="dot"),
            )
        )
        fig_temp.update_layout(
            title="Термический профиль и выход токсичного CO",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=45, b=20),
            xaxis=dict(title="Время от воспламенения (с)"),
            yaxis=dict(title="Температура (K)", side="left"),
            yaxis2=dict(title="CO (ppm)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    st.divider()
    if st.button("⚡ Сформировать экспертное заключение SI по прогнозу горения", type="primary"):
        with st.spinner("SI анализирует термодинамику компонентов, синергию смеси и базу NASA PSI..."):
            try:
                analysis_text, used_sources = query_si_rag_engine(
                    profile=active_profile,
                    o2_pct=o2_conc,
                    flow_cms=vent_flow,
                    pressure_kpa=cabin_pressure,
                    risk_index=risk_idx,
                    status=threat_status,
                    telemetry_df=telemetry_df,
                    lm_url=lm_studio_url,
                    model_name=lm_model_name,
                    api_key=lm_api_key,
                )
                st.markdown(f'<div class="si-analysis-box">{analysis_text}</div>', unsafe_allow_html=True)
                if used_sources:
                    st.caption("📚 Верифицированные источники NASA PSI: " + " • ".join(used_sources))
            except Exception as exc:
                detail = getattr(exc, "body", None) or getattr(exc, "message", str(exc))
                st.error(f"Ошибка API: {detail}")

elif selected_tab == "💬 ИИ-помощник: Физика горения и борьба с пожаром в 0g":
    st.subheader("Диалоговый консультант по горению компонентов и смесей в 0g")

    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt1:
        st.checkbox(
            f"Изолировать поиск в базе NASA PSI только по активным компонентам ({', '.join(active_profile['weights'].keys())})",
            value=False,
            key="strict_filter_val",
        )
    with col_opt2:
        if st.button("🗑️ Очистить диалог", use_container_width=True):
            st.session_state.si_chat_history = st.session_state.si_chat_history[:1]
            st.rerun()

    st.caption("⚡ Быстрые тактические запросы:")
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_prompt = None
    with q_col1:
        if st.button("🧯 Тактика тушения текущего состава", use_container_width=True):
            quick_prompt = (
                f"В отсеке горит {active_profile['display_name']} при O2 = {o2_conc}% и вентиляции {vent_flow} см/с. "
                f"Опиши пошаговый алгоритм тушения, выбор огнетушащего вещества и риски токсичных выделений в невесомости."
            )
    with q_col2:
        if st.button("🧪 Как компоненты влияют друг на друга в смеси?", use_container_width=True):
            quick_prompt = (
                f"Проанализируй физику совместного горения для конфигурации {active_profile['display_name']}: "
                f"какой компонент воспламеняется первым, возникает ли фитильный эффект или термическое разложение "
                f"негорючих добавок (например, PTFE/PVC) при нагреве от пламени?"
            )
    with q_col3:
        if st.button("🌀 Влияние слабой вентиляции (0.3–5 см/с) в 0g", use_container_width=True):
            quick_prompt = (
                "Объясни на основе экспериментов NASA BASS-II и Saffire, почему в микрогравитации слабый обдув "
                "усиливает горение полимеров и смесей, и в какой момент при тушении нужно отключать вентиляцию."
            )

    if quick_prompt:
        st.session_state["pending_prompt"] = quick_prompt
        st.rerun()

    for msg in st.session_state.si_chat_history:
        with st.chat_message(msg["role"], avatar="🛰️" if msg["role"] == "assistant" else "👨‍🚀"):
            st.markdown(extract_clean_text(msg.get("content", "")))
            if msg.get("sources"):
                st.caption("📚 Источники NASA PSI: " + " • ".join(msg["sources"]))

elif selected_tab == "📊 Матрица ранжирования рисков (NASA PSI)":
    st.subheader("Сравнительное ранжирование чистых полимеров и текущей смеси")
    df_ranking = build_risk_matrix_df(active_profile, o2_conc, vent_flow, cabin_pressure)

    fig_bar = px.bar(
        df_ranking,
        x="Индекс риска SI",
        y="Объект / Материал",
        orientation="h",
        color="Статус отсека",
        color_discrete_map={"НОРМА": "#10B981", "ПОВЫШЕННЫЙ": "#F59E0B", "КРИТИЧЕСКИЙ": "#EF4444"},
        text="Индекс риска SI",
        template="plotly_dark",
        title="Интегральный индекс пожарной опасности в микрогравитации (1–10)",
    )
    fig_bar.update_layout(height=340, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        df_ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Индекс риска SI": st.column_config.ProgressColumn("Индекс риска SI", min_value=0.0, max_value=10.0, format="%.1f"),
        },
    )

# Поле ввода ВСЕГДА на верхнем уровне в самом конце файла
user_input = st.chat_input("Задайте вопрос ИИ-эксперту по горению компонентов, смесей или тактике тушения в 0g...")
active_prompt = st.session_state.pop("pending_prompt", None) or user_input

if active_prompt:
    st.session_state.si_chat_history.append({"role": "user", "content": active_prompt})

    if st.session_state.get("active_tab") != "💬 ИИ-помощник: Физика горения и борьба с пожаром в 0g":
        st.session_state["switch_to_chat"] = True

    with st.spinner("Анализ физики процесса и поиск в отчетах NASA PSI..."):
        try:
            reply_text, sources = ask_si_assistant(
                user_question=active_prompt,
                chat_history=st.session_state.si_chat_history[:-1],
                profile=active_profile,
                o2_pct=o2_conc,
                flow_cms=vent_flow,
                pressure_kpa=cabin_pressure,
                risk_index=risk_idx,
                status=threat_status,
                telemetry_df=telemetry_df,
                lm_url=lm_studio_url,
                model_name=lm_model_name,
                api_key=lm_api_key,
                strict_material_filter=st.session_state.get("strict_filter_val", False),
            )
            st.session_state.si_chat_history.append({
                "role": "assistant",
                "content": reply_text,
                "sources": sources,
            })
        except Exception as exc:
            err_msg = f"⚠️ Ошибка связи с сервером модели ({lm_studio_url}): {exc}"
            st.session_state.si_chat_history.append({
                "role": "assistant",
                "content": err_msg,
                "sources": [],
            })
    st.rerun()