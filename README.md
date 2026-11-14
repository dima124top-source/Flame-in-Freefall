# FreefallFlame SI: Spacecraft Fire Safety RAG Assistant

AI-powered scientific assistant built for **NASA Space Apps Challenge 2026** (*Flame in Freefall* challenge).

## Overview
In microgravity, the absence of buoyant convection fundamentally alters flame propagation, soot morphology, and extinguishment limits. This tool indexes peer-reviewed NASA combustion reports (Saffire, BASS, FLEX, SAME) and historical incident reviews into a vector knowledge base to provide flight engineers with actionable fire safety insights.

## Setup & Local Run
1. Create virtual environment (Python 3.12):
   ```bash
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
Создать и активировать виртуальное окружение Python, установить зависимости: pip install -r requirements.txt.

Поместить PDF-отчеты в data/pdf/ и выполнить индексацию: python scripts/indexer.py.

Запустить веб-приложение: streamlit run scripts/app.py.

В боковой панели указать Endpoint API, имя модели и API-ключ (для Google AI Studio, OpenRouter или локального LM Studio).