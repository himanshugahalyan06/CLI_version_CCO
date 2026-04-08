# ☁️ Cloud Cost Optimizer (CCO)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Reinforcement Learning (RL) environment and CLI tool for simulating cloud server scaling. Designed for the **OpenEnv** ecosystem.

## 📖 Overview

The **Cloud Cost Optimizer** simulates the enterprise dilemma of balancing **Performance** (latency) vs **Cost** (server count). It provides a high-fidelity simulator where an AI agent must dynamically scale a fleet of cloud servers based on real-time traffic telemetry.

## 🚀 Professional CLI Features

The project now features a unified, industry-standard CLI built with `Typer` and `Rich`:

- 🖥️ **Integrated API Server**: Fast and robust FastAPI-based environment server.
- 🤖 **AI Simulation**: Built-in support for running LLM-based agents (OpenAI, NVIDIA NIM, etc.) against the environment.
- 📊 **Rich Output**: Beautiful terminal tables and progress bars for real-time monitoring.
- ⚙️ **Configurable**: Managed via Environment Variables or `.env` files.
- 🧪 **Tested**: Comprehensive unit tests for core environment physics.

---

## 🛠️ Installation

### Local Setup (Recommended)
We recommend using `uv` or `pip` in a virtual environment.

```bash
# Clone and enter the directory
python3 -m pip install -e .
```

---

## 🕹️ CLI Usage

The primary entry point is the `cco` command.

### 1. Start the Environment Server
Before running a simulation, start the environment API:
```bash
cco serve
```
*The server runs on `http://0.0.0.0:7860` by default. You can access the Interactive API docs at `/scalar`.*

### 2. Run an AI Simulation
In a new terminal, run the agent:
```bash
# Run with default settings (Task: easy)
cco simulate

# Run a specific task with a specific model
cco simulate --task hard --model meta/llama-3.1-405b-instruct
```

---

## 🎯 Task Progression

1. **Easy (`easy`)**: Flat, predictable traffic. Focuses on baseline efficiency.
2. **Medium (`medium`)**: Sinusoidal waves. Tests smooth scaling without oscillations.
3. **Hard (`hard`)**: Extreme, volatile spikes. Challenges aggressive yet stable scaling.

---

## 🏗️ Architecture (Industry Standard)

```text
src/cloud_optimizer/
├── api/        # FastAPI Application Layer
├── cli/        # Typer CLI Interface
├── core/       # RL Environment & Logic (Physics)
└── config.py   # Pydantic Settings
tests/          # Unit Tests
```

---

## ⚙️ Configuration

Configure the tool using environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `dummy` | Your LLM API Key (Injected by Validator) |
| `API_BASE_URL` | `https://integrate.api.nvidia.com/v1` | LLM Proxy URL (Injected by Validator) |
| `MODEL_NAME` | `meta/llama-3.1-8b-instruct` | LLM Model Name |
| `ENV_URL` | `http://127.0.0.1:7860` | Internal Environment Server URL |
| `API_PORT` | `7860` | Internal server port |

---

## 🧪 Development

Run tests with:
```bash
make test
```

Developed with ❤️ for the AI Engineering community.