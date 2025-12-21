# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STORM** is an automated knowledge management system that uses large language models to generate Wikipedia-style articles from scratch. It features two modes:
- **STORM**: Automated generation through retrieval and multi-perspective question asking
- **Co-STORM**: Human-AI collaborative knowledge curation system

Published as PyPI package `knowledge-storm` (v1.1.1).

## Common Commands

### Installation

```bash
# Create virtual environment
conda create -n storm python=3.11
conda activate storm

# Install dependencies
pip install -r requirements.txt

# Or use uv (recommended for faster installs)
uv pip install -r requirements.txt
```

### Development Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Format code (required before commits)
black knowledge_storm/
```

### Running Examples

**STORM Examples:**
```bash
python examples/storm_examples/run_storm_wiki_gpt.py \
    --output-dir ./output \
    --retriever bing \
    --do-research \
    --do-generate-outline \
    --do-generate-article \
    --do-polish-article

# Run with Claude
python examples/storm_examples/run_storm_wiki_claude.py \
    --output-dir ./output \
    --retriever bing

# Run with VectorRM
python examples/storm_examples/run_storm_wiki_gpt_with_VectorRM.py \
    --output-dir ./output
```

**Co-STORM Example:**
```bash
python examples/costorm_examples/run_costorm_gpt.py \
    --output-dir ./output \
    --retriever bing
```

### Streamlit Demo Interface

```bash
cd frontend/demo_light
pip install -r requirements.txt
streamlit run storm.py
```

### Testing

**Note:** This project currently lacks comprehensive test coverage. There are no dedicated test files or test directories. When adding tests:
- Use `pytest` for testing framework
- Create `tests/` directory following Python conventions
- Add tests for core modules in `knowledge_storm/storm_wiki/modules/`

## Architecture

### Core Components

```
knowledge_storm/
├── storm_wiki/              # STORM engine
│   ├── engine.py            # Main STORM orchestration (17,988 lines)
│   └── modules/             # Core modules
│       ├── article_generation.py
│       ├── article_polish.py
│       ├── knowledge_curation.py
│       ├── outline_generation.py
│       ├── persona_generator.py
│       ├── retriever.py
│       └── storm_dataclass.py
├── collaborative_storm/     # Co-STORM engine
│   ├── engine.py            # Main Co-STORM orchestration (32,330 lines)
│   └── modules/             # Co-STORM specific modules
│       ├── co_storm_agents.py
│       ├── grounded_question_answering.py
│       └── warmstart_hierarchical_chat.py
├── interface.py             # Unified interfaces (21,021 lines)
├── lm.py                    # Language model interface (42,725 lines)
├── rm.py                    # Retrieval model interface (47,638 lines)
├── encoder.py               # Text encoding utilities
├── utils.py                 # Helper functions
├── dataclass.py             # Data structures
└── logging_wrapper.py       # Logging configuration
```

### STORM Workflow (4 Stages)

1. **Knowledge Curation** - Collect information through simulated dialogue
2. **Outline Generation** - Create structured article outline
3. **Article Generation** - Generate full article from outline and references
4. **Article Polishing** - Refine and improve article quality

### Co-STORM Workflow (3 Stages)

1. **Warm Start** - Establish shared conceptual space between human and AI
2. **Collaborative Discourse** - Multi-agent dialogue with turn management
3. **Dynamic Mind Map** - Maintain evolving concept graph

### Language Model Support

**Via LiteLLM (`lm.py`):**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Azure OpenAI
- Mistral (via VLLM)
- 100+ other models

Configure models in `secrets.toml`:
```toml
OPENAI_API_KEY="your_key"
ANTHROPIC_API_KEY="your_key"
```

### Retrieval System Support

**Multiple search backends (`rm.py`):**
- You.com (YouRM)
- Bing Search
- Brave Search
- Serper
- DuckDuckGo
- Tavily Search
- SearXNG
- Azure AI Search
- VectorRM (custom vector database with Qdrant)

### Key Dependencies

- **dspy** (v2.4.9) - Programmable LLM framework
- **litellm** - Unified 100+ LLM API interface
- **langchain** - LLM application framework
- **qdrant-client** - High-performance vector database
- **streamlit** - Web UI framework
- **sentence-transformers** - Text embeddings

## Configuration Files

### `pyproject.toml`
- Modern Python project configuration
- Python >=3.11 required
- Fixed dependency versions for stability

### `requirements.txt`
- 12 core dependencies with pinned versions
- Ensure environment reproducibility

### `.pre-commit-config.yaml`
- Black formatter for code style
- Only formats `knowledge_storm/` directory
- Automatically runs on commit

### `secrets.toml` (create yourself)
Store API keys and secrets (git-ignored):
```toml
OPENAI_API_KEY="sk-..."
BING_SEARCH_API_KEY="..."
ANTHROPIC_API_KEY="..."
```

## Important Files

### Documentation
- **README.md** (372 lines) - Comprehensive project documentation, installation, API usage
- **CONTRIBUTING.md** (48 lines) - Contribution guidelines, development setup
- **frontend/demo_light/README.md** - Streamlit demo instructions

### Examples
- `examples/storm_examples/` - STORM usage examples for different models
- `examples/costorm_examples/` - Co-STORM collaborative examples

### Frontend
- `frontend/demo_light/storm.py` - Streamlit-based interactive demo
- Real-time visualization of generation process

## Development Guidelines

### Code Style
- **Format with Black** before committing (pre-commit hooks enforce this)
- Follow existing patterns in `knowledge_storm/` modules
- Keep modules focused on single responsibilities

### Adding Features
1. Follow the modular architecture in `storm_wiki/modules/` or `collaborative_storm/modules/`
2. Extend `interface.py` for new abstract base classes
3. Add example scripts in `examples/` directory
4. Update `README.md` with new capabilities

### Known Issues
- **Large files**: Some core files exceed 1000 lines (e.g., `lm.py`, `rm.py`, `engine.py`)
- **No tests**: Project lacks comprehensive test coverage
- **No CI/CD**: No GitHub Actions or similar CI configuration

### Extending Retrieval
To add new retrieval systems, extend `knowledge_storm/rm.py`:
- Implement BaseRM abstract class
- Add configuration in `secrets.toml`
- Update examples with new `--retriever` option

### Extending Language Models
To add new LLM providers, extend `knowledge_storm/lm.py`:
- Implement BaseLM abstract class
- Configure in `secrets.toml`
- Use LiteLLM for standard API compatibility

## Dataset Resources

The project includes reference to datasets:
- **FreshWiki** - Clean Wikipedia articles for evaluation
- **WildSeek** - Web-scraped knowledge base

Refer to `README.md` for dataset access and usage.

## PyPI Package

The project is published as `knowledge-storm` on PyPI:
```bash
pip install knowledge-storm
```

This installs the package for direct import and use in other projects.

## Important Notes

1. **Python Version**: Requires Python 3.10+ (3.11 recommended)
2. **API Keys**: All providers require valid API keys in `secrets.toml`
3. **Rate Limits**: Be mindful of API rate limits when running batch operations
4. **Output Directory**: Always specify `--output-dir` for example scripts
5. **Memory Usage**: Large language models require significant RAM (8GB+ recommended)