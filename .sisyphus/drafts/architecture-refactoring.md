# Draft: Architecture Refactoring

## Requirements (confirmed)
- [requirement]: 프로젝트를 구조화하고 human readable 한 아키텍쳐로 재구성 리펙토링

## Current Architecture Analysis

### Current Issues
1. **agent.py**: 1157 lines - too large, mixed responsibilities
2. **Monolithic structure**: Planning, research, synthesis, dynamic replanning all in one file
3. **Low readability**: Functions are not organized by domain/responsibility
4. **Tight coupling**: State management, graph creation, and execution logic are tightly coupled

### Current File Structure
```
mini-research-agent/
├── run.py
├── requirements.txt
├── .env
├── src/
│   ├── agent.py              # 1157 lines - TOO LARGE
│   ├── state.py              # State types (good separation)
│   ├── config.py             # Configuration (good separation)
│   ├── tools.py              # MCP tools (good separation)
│   ├── prompts.py            # Prompts (good separation)
│   └── tui/
│       ├── app.py
│       ├── controller.py
│       ├── models.py
│       ├── views.py
│       └── styles.py
└── tests/
```

## Technical Decisions
- **Architecture Pattern**: Modular by Function (기능별 모듈 분리)
- **Refactoring Scope**: agent.py만 리팩토링
- **Test Updates**: 예, 모두 업데이트
- **Test Infrastructure**: 존재하지 않음 - 설정 필요

## Target Architecture Structure (Modular by Function)

```
src/
├── agent.py                # Main entry point, orchestration (simplified)
├── core/                  # Core business logic modules
│   ├── __init__.py
│   ├── planning.py          # Intent analysis, task decomposition, plan validation
│   ├── research.py         # Research node, tool execution coordination
│   ├── synthesis.py        # Report generation and synthesis
│   ├── replanning.py       # Dynamic replanning logic
│   ├── state_manager.py    # State update and transition logic
│   └── graph_builder.py   # LangGraph graph construction
├── state.py               # State type definitions (keep existing)
├── config.py              # Configuration (keep existing)
├── tools.py               # MCP tools (keep existing)
├── prompts.py             # Prompts (keep existing)
└── tui/                  # TUI (unchanged)
    ├── app.py
    ├── controller.py
    ├── models.py
    ├── views.py
    └── styles.py
```

## Open Questions
- 테스트 인프라 설정: pytest, unittest 중 무엇을 선호하시나요?
- 파일 네이밍 컨벤션: snake_case, camelCase 중 선호하시는 것이 있나요?
