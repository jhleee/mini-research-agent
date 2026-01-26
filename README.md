# Mini Research Agent

복잡한 연구 질문을 하위 질문으로 분해하고, 웹 검색과 웹페이지 읽기를 통해 정보를 수집한 후, 종합적인 마크다운 리포트를 생성하는 지능형 연구 에이전트입니다.

## Features

- **계층적 연구 계획**: 복잡한 질문을 주제별로 분해하고 각 주제에 대한 세부 검색 쿼리 생성
- **비교 질문 처리**: "A와 B를 비교해라" 같은 질문을 독립적인 주제로 분리하여 체계적으로 처리
- **자동 웹 리서치**: 웹 검색 및 웹페이지 읽기를 통한 자동 정보 수집
- **실시간 진행 상황**: TUI(Terminal UI)를 통한 실시간 연구 진행 상황 확인
- **마크다운 리포트**: 수집된 정보를 종합하여 구조화된 마크다운 리포트 생성

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain + OpenAI |
| Terminal UI | Textual |
| HTTP Client | httpx (SSE 스트리밍 지원) |
| LLM | Z.AI API (glm-4.7 model) |

## Installation

```bash
# 저장소 클론
git clone https://github.com/jhleee/mini-research-agent.git
cd mini-research-agent

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 설정
```

### 환경 변수

`.env` 파일에 다음 변수들을 설정합니다:

```env
OPENAI_API_KEY=<your-z-ai-api-key>
OPENAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
OPENAI_MODEL=glm-4.7
```

## Usage

```bash
# TUI 애플리케이션 실행
python run.py
```

TUI에서 연구 질문을 입력하면 에이전트가 자동으로 연구를 수행합니다.

### API 사용

```python
import asyncio
from src.agent import run_research_with_tools

async def main():
    query = "Python과 Rust의 메모리 관리 방식을 비교해라"

    def callback(event):
        print(f"Event: {event}")

    report = await run_research_with_tools(query, callback)
    print(report)

asyncio.run(main())
```

## Architecture

### 프로젝트 구조

```
mini-research-agent/
├── run.py                    # 메인 진입점
├── requirements.txt          # 의존성
├── .env                      # 환경 변수 (API 키)
│
├── src/
│   ├── agent.py              # LangGraph 워크플로우 (핵심 로직)
│   ├── state.py              # 상태 타입 정의
│   ├── config.py             # 설정 상수
│   ├── tools.py              # 웹 검색/읽기 도구
│   ├── prompts.py            # LLM 시스템 프롬프트
│   │
│   └── tui/                  # Terminal UI
│       ├── app.py            # Textual 앱
│       ├── controller.py     # MVC 컨트롤러
│       ├── models.py         # 데이터 모델
│       ├── views.py          # UI 위젯
│       └── styles.py         # CSS 스타일
│
└── tests/
    ├── test_phase1.py        # Phase 1 테스트 (의도 분석)
    ├── test_phase2.py        # Phase 2 테스트 (태스크 분해)
    ├── test_phase3.py        # Phase 3 테스트 (계획 검증)
    └── test_end_to_end.py    # E2E 테스트
```

## LangGraph Workflow

연구 에이전트의 핵심인 LangGraph 워크플로우는 다음과 같은 노드들로 구성됩니다.

### 노드 구성도

```
                        ┌─────────────────┐
                        │   START         │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   plan_node     │
                        │                 │
                        │ - Intent 분석    │
                        │ - 태스크 분해    │
                        │ - 계획 검증     │
                        │ - 계획 개선     │
                        └────────┬────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │                                     │
              │  ┌─────────────────┐                │
              │  │  research_node  │◄───────────┐   │
              │  │                 │            │   │
              │  │ LLM + Tools     │            │   │
              │  │ (도구 호출 결정) │            │   │
              │  └────────┬────────┘            │   │
              │           │                     │   │
              │           ▼                     │   │
              │  ┌─────────────────┐            │   │
              │  │should_continue()│            │   │
              │  │   (라우팅)       │            │   │
              │  └────────┬────────┘            │   │
              │           │                     │   │
              │     ┌─────┼─────┬───────┐       │   │
              │     │     │     │       │       │   │
              │     ▼     │     ▼       ▼       │   │
              │ "tools"   │ "research" "end"    │   │
              │     │     │     │               │   │
              │     │     │     └───────┐       │   │
              │     ▼     │             │       │   │
              │  ┌────────┴──┐          │       │   │
              │  │tools_node │          │       │   │
              │  │           │          │       │   │
              │  │-web_search│          │       │   │
              │  │-read_page │          │       │   │
              │  └─────┬─────┘          │       │   │
              │        │                │       │   │
              │        ▼                │       │   │
              │  ┌────────────┐         │       │   │
              │  │process_    │         │       │   │
              │  │tool_results│─────────┼───────┘   │
              │  │            │         │           │
              │  │- 결과 추출  │         │           │
              │  │- 다음 태스크│         │           │
              │  └────────────┘         │           │
              │                         │           │
              │  Research Loop          │           │
              └─────────────────────────┼───────────┘
                                        │
                          "synthesize"  │
                                        ▼
                        ┌─────────────────┐
                        │ synthesize_node │
                        │                 │
                        │ 최종 리포트 생성 │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │      END        │
                        └─────────────────┘
```

### 그래프 엣지 정의

```python
# src/agent.py:create_research_graph()

# 엣지 구성
graph.set_entry_point("plan")
graph.add_edge("plan", "research")

# research 노드 이후 조건부 라우팅
graph.add_conditional_edges(
    "research",
    should_continue,
    {
        "tools": "tools",
        "research": "research",
        "synthesize": "synthesize",
        "end": END,
    }
)

# tools → process_results → research (항상 research로 복귀)
graph.add_edge("tools", "process_results")
graph.add_edge("process_results", "research")

# synthesize → END
graph.add_edge("synthesize", END)
```

### 노드 상세 설명

| 노드 | 함수명 | 역할 |
|------|--------|------|
| **plan_node** | `plan_node(state)` | 4단계 계획 수립 파이프라인 실행 |
| **research_node** | `research_node(state)` | 도구 바인딩된 LLM으로 검색 전략 결정 |
| **tools_node** | `ToolNode(TOOLS)` | 실제 도구 실행 (웹 검색, 웹페이지 읽기) |
| **process_results** | `process_tool_results(state)` | 도구 결과에서 정보 추출 및 태스크 진행 |
| **synthesize_node** | `synthesize_node(state)` | 수집된 정보를 종합하여 최종 리포트 생성 |

### 라우팅 로직 (should_continue)

```python
def should_continue(state):
    # 마지막 메시지에 tool_calls가 있으면 → "tools"
    # status가 "synthesizing"이거나 iteration >= MAX_ITERATIONS → "synthesize"
    # 그 외 → "research" (다음 반복)
    # status가 "done"이면 → "end"
```

### 계획 수립 4단계 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                        plan_node 내부                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Intent Analysis (의도 분석)                           │
│  ├─ 질문에서 독립적인 주제 식별                                  │
│  ├─ 비교 질문 처리 (예: "A와 B 비교" → 2개 주제)                 │
│  └─ 결과: main_topics 리스트                                    │
│                         │                                       │
│                         ▼                                       │
│  Phase 2: Task Decomposition (태스크 분해)                      │
│  ├─ 각 주제별 2-4개 검색 쿼리 생성                              │
│  ├─ 주제에 특화된 구체적인 쿼리                                 │
│  └─ 결과: sub_tasks 리스트                                      │
│                         │                                       │
│                         ▼                                       │
│  Phase 3: Plan Validation (계획 검증)                           │
│  ├─ 계획의 완전성 검증                                          │
│  ├─ 원본 질문 커버리지 확인                                     │
│  └─ 결과: is_valid, confidence, issues                         │
│                         │                                       │
│                         ▼                                       │
│  Phase 4: Refinement (계획 개선) - 선택적                       │
│  ├─ 검증 피드백 기반 계획 개선                                  │
│  └─ 최대 1회 개선                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## State Management

### ResearchState 구조

```python
class ResearchState(TypedDict):
    query: str                    # 원본 사용자 질문
    messages: Annotated[list, add_messages]  # LLM 메시지 히스토리 (자동 병합)
    research_plan: list[str]      # 플랫 쿼리 리스트 (하위 호환)
    search_queries: list[str]     # 실행된 검색 쿼리 목록
    read_urls: list[str]          # 읽은 URL 목록
    findings: list[str]           # 수집된 연구 결과
    iteration: int                # 루프 카운터
    report: str                   # 최종 마크다운 리포트
    status: str                   # "planning" → "researching" → "synthesizing" → "done"
    hierarchical_plan: Optional[HierarchicalPlan]  # 계층적 계획 구조
    current_main_task_id: Optional[str]  # 현재 실행 중인 메인 태스크 ID
    current_sub_task_id: Optional[str]   # 현재 실행 중인 서브 태스크 ID
    planning_phase: str           # "intent_analysis" → "decomposition" → "validation" → "complete"
```

### 계층적 계획 구조

```python
HierarchicalPlan:
  ├─ original_query: str          # 원본 질문
  ├─ intent_count: int            # 식별된 의도 수
  ├─ main_tasks: list[MainTask]   # 메인 태스크 리스트
  │   ├─ id: str                  # "1", "2", ...
  │   ├─ topic: str               # 주제명
  │   ├─ description: str         # 주제 설명
  │   ├─ status: str              # TaskStatus (pending/in_progress/completed)
  │   ├─ sub_tasks: list[SubTask] # 서브 태스크 리스트
  │   │   ├─ id: str              # "1.1", "1.2", ...
  │   │   ├─ query: str           # 검색 쿼리
  │   │   ├─ status: str          # TaskStatus
  │   │   └─ findings: list[str]  # 수집된 정보
  │   └─ summary: str             # 메인 태스크에 대한 결과 요약
  ├─ is_validated: bool
  └─ refinement_count: int
```

## Tools (MCP Integration)

Z.AI MCP 엔드포인트를 통해 두 가지 도구를 제공합니다:

| 도구 | 함수 | 설명 |
|------|------|------|
| **web_search** | `web_search(query: str)` | 웹 검색 수행 |
| **read_webpage** | `read_webpage(url: str)` | 웹페이지 콘텐츠 읽기 (8000자 제한) |

### MCP 통신

- **프로토콜**: JSON-RPC 2.0 over HTTP
- **스트리밍**: SSE (Server-Sent Events) 지원
- **인증**: Bearer 토큰

## TUI Architecture

Textual 프레임워크 기반 MVC 아키텍처:

```
┌─────────────────────────────────────────────┐
│               ResearchApp                    │
│                  (app.py)                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐        ┌───────────────┐  │
│  │  ChatPanel  │        │   TaskPanel   │  │
│  │             │        │               │  │
│  │  메시지 표시 │        │  태스크 진행   │  │
│  │             │        │  상황 표시     │  │
│  └─────────────┘        └───────────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │            InputBar                  │   │
│  │         질문 입력 영역                │   │
│  └─────────────────────────────────────┘   │
│                                             │
├─────────────────────────────────────────────┤
│            ResearchController               │
│             (controller.py)                 │
│                                             │
│  - 상태 관리                                │
│  - 이벤트 핸들링                            │
│  - 에이전트 ↔ UI 연결                       │
└─────────────────────────────────────────────┘
```

## Configuration

`src/config.py`에서 주요 설정을 관리합니다:

```python
# Agent 제한
MAX_ITERATIONS = 10      # 최대 연구 반복 횟수
MAX_SEARCH_RESULTS = 5   # 검색당 최대 결과 수

# Planning 설정
MAX_MAIN_TASKS = 4       # 최대 메인 태스크 수
MAX_SUB_TASKS = 4        # 메인 태스크당 최대 서브 태스크 수
MAX_REFINEMENTS = 1      # 최대 계획 개선 횟수
VALIDATION_THRESHOLD = 0.7  # 검증 임계값
```

## Testing

```bash
# Phase 1 테스트 (의도 분석)
python test_phase1.py

# Phase 2 테스트 (태스크 분해)
python test_phase2.py

# Phase 3 테스트 (계획 검증)
python test_phase3.py

# E2E 테스트 (전체 파이프라인)
python test_end_to_end.py
```

## Data Flow

```
사용자 질문 입력
       │
       ▼
┌──────────────────┐
│    plan_node     │
│                  │
│ 1. 의도 분석     │
│ 2. 태스크 분해   │
│ 3. 계획 검증     │
│ 4. 계획 개선     │
└────────┬─────────┘
         │
         ▼
계층적 계획 생성
(main_tasks, sub_tasks)
         │
         ▼
┌──────────────────┐
│  research_node   │◄──────┐
│                  │       │
│ 다음 서브태스크   │       │
│ 선택 및 도구 호출 │       │
└────────┬─────────┘       │
         │                 │
         ▼                 │
┌──────────────────┐       │
│   tools_node     │       │
│                  │       │
│ - web_search()   │       │
│ - read_webpage() │       │
└────────┬─────────┘       │
         │                 │
         ▼                 │
┌──────────────────┐       │
│ process_results  │       │
│                  │       │
│ - 결과 추출      │───────┘
│ - 태스크 완료    │  (모든 태스크 완료까지 반복)
│ - 다음 태스크    │
└────────┬─────────┘
         │
         ▼ (모든 태스크 완료)
┌──────────────────┐
│ synthesize_node  │
│                  │
│ 최종 마크다운    │
│ 리포트 생성      │
└────────┬─────────┘
         │
         ▼
   최종 리포트 출력
```

## License

MIT License

## Contributing

이슈 및 PR 환영합니다.
