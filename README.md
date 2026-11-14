# FreefallFlame SI: Spacecraft Fire Safety RAG Assistant

AI-powered scientific assistant built for **NASA Space Apps Challenge 2026** (*Flame in Freefall* challenge).

## Overview
In microgravity, the absence of buoyant convection fundamentally alters flame propagation, soot morphology, and extinguishment limits. This tool indexes peer-reviewed NASA combustion reports (Saffire, BASS, FLEX, SAME) and historical incident reviews into a vector knowledge base to provide flight engineers with actionable fire safety insights.

## Setup & Local Run
1. Create virtual environment (Python 3.12):
   ```bash
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1