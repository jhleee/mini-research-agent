# Ralph Loop 기반 개선 계획서

> 이 문서는 Ralph Loop 패턴에서 영감받은 개선사항을 정의합니다.
> 다른 개발자(인간 또는 AI)가 이 문서만 보고도 구현할 수 있도록 작성되었습니다.

## 배경

**Ralph Loop**는 자율 AI 에이전트 루프 기법으로, 핵심 원칙:
1. **Fresh Context** - 매 이터레이션마다 새로운 컨텍스트
2. **File-based Memory** - 진행상황을 파일에 저장
3. **Backpressure** - 외부 검증으로 자기수정 강제
4. **Clear Exit Criteria** - 명확한 종료 조건

현재 우리 시스템은 Dynamic Replanning을 지원하지만, Ralph Loop의 핵심 개념들이 부족합니다.

---

## 개선 1: Fresh Context per Iteration

### 목표
매 research iteration마다 LLM에 fresh context를 제공하여 "스마트 존" 유지

### 현재 문제
```python
# src/agent.py - research_node()
messages.extend(state.get("messages", [])[-10:])  # 이전 메시지 누적
```
메시지가 누적되어 컨텍스트가 오염될 수 있음

### 구현 계획

#### 1.1 State에 findings_summary 필드 추가

**파일**: `src/state.py`

```python
class ResearchState(TypedDict):
    # ... 기존 필드들 ...

    # Fresh Context 지원
    findings_summary: str  # 이전 findings의 요약본 (매 iteration 갱신)
    iteration_context: str  # 현재 iteration에 필요한 컨텍스트만
```

#### 1.2 Findings 요약 함수 추가

**파일**: `src/agent.py`

```python
def summarize_findings(findings: list[str], max_length: int = 1000) -> str:
    """이전 findings를 간결하게 요약하여 fresh context 제공.

    Args:
        findings: 지금까지 수집된 findings 리스트
        max_length: 요약 최대 길이

    Returns:
        간결한 요약 문자열
    """
    if not findings:
        return ""

    # 각 finding에서 핵심 정보만 추출 (첫 100자)
    summaries = []
    for f in findings[-5:]:  # 최근 5개만
        short = f[:100] + "..." if len(f) > 100 else f
        summaries.append(f"- {short}")

    return "Previous findings:\n" + "\n".join(summaries)
```

#### 1.3 research_node 수정

**파일**: `src/agent.py`

```python
def research_node(state: ResearchState) -> dict:
    """Execute research with FRESH context each iteration."""
    llm = create_llm().bind_tools(TOOLS)

    # Fresh context: 이전 메시지 대신 요약만 사용
    findings_summary = summarize_findings(state.get("findings", []))

    # 현재 task 정보
    current_query, current_main_topic = find_current_task(...)

    # Fresh messages - 이전 대화 없이 새로 시작
    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"""
Research query: {state['query']}
Current focus: {current_query}

{findings_summary}

Use web_search to find information about: {current_query}
""")
    ]

    # 이전 messages를 extend하지 않음!
    response = llm.invoke(messages)

    return {
        "messages": [response],  # 새 메시지만
        "findings_summary": findings_summary,
        # ...
    }
```

### 테스트 방법
```python
# test_fresh_context.py
def test_fresh_context():
    """Verify that each iteration starts with fresh context."""
    # 10번 iteration 후에도 messages 길이가 제한적인지 확인
    # findings_summary가 적절히 생성되는지 확인
```

---

## 개선 2: File-based Memory (Progress Persistence)

### 목표
연구 진행상황을 파일에 저장하여:
1. 중단 후 재개 가능
2. 디버깅 용이
3. Git history로 진행 추적

### 구현 계획

#### 2.1 Progress 파일 구조 정의

**파일**: `src/progress.py` (새 파일)

```python
"""Progress persistence for research sessions."""
import json
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

PROGRESS_DIR = Path(".research_progress")


class ProgressManager:
    """Manages research progress persistence to files."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.progress_file = PROGRESS_DIR / f"{session_id}.json"
        PROGRESS_DIR.mkdir(exist_ok=True)

    def save(self, state: dict) -> None:
        """Save current research state to file.

        Args:
            state: Current ResearchState dict
        """
        progress = {
            "session_id": self.session_id,
            "updated_at": datetime.now().isoformat(),
            "query": state.get("query", ""),
            "status": state.get("status", ""),
            "iteration": state.get("iteration", 0),
            "replan_count": state.get("replan_count", 0),
            "completed_queries": state.get("search_queries", []),
            "findings": state.get("findings", []),
            "hierarchical_plan": state.get("hierarchical_plan"),
            "current_task": {
                "main_id": state.get("current_main_task_id"),
                "sub_id": state.get("current_sub_task_id"),
            },
        }

        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def load(self) -> Optional[dict]:
        """Load previous progress if exists.

        Returns:
            Progress dict or None if not found
        """
        if not self.progress_file.exists():
            return None

        with open(self.progress_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear(self) -> None:
        """Clear progress file after completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()

    @staticmethod
    def generate_session_id(query: str) -> str:
        """Generate unique session ID from query."""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"{timestamp}_{query_hash}"
```

#### 2.2 Progress 저장 hook 추가

**파일**: `src/agent.py`

`process_tool_results` 함수 수정:

```python
def process_tool_results(state: ResearchState) -> dict:
    """Process tool results and save progress."""
    # ... 기존 로직 ...

    # Progress 저장 (session_id가 있으면)
    session_id = state.get("session_id")
    if session_id:
        from .progress import ProgressManager
        pm = ProgressManager(session_id)
        pm.save(state)

    return {
        # ... 기존 반환값 ...
    }
```

#### 2.3 Resume 기능 추가

**파일**: `src/agent.py`

```python
async def run_research_with_tools(
    query: str,
    callback=None,
    config={"recursion_limit": 200},
    resume_session: Optional[str] = None,  # 새 파라미터
) -> str:
    """Run research with optional resume capability.

    Args:
        query: Research question
        callback: Event callback function
        config: LangGraph config
        resume_session: Session ID to resume (optional)
    """
    from .progress import ProgressManager

    # Resume 또는 새 세션
    if resume_session:
        pm = ProgressManager(resume_session)
        previous = pm.load()
        if previous:
            # 이전 상태에서 시작
            initial_state = {
                "query": previous["query"],
                "findings": previous["findings"],
                "search_queries": previous["completed_queries"],
                "iteration": previous["iteration"],
                "hierarchical_plan": previous["hierarchical_plan"],
                "current_main_task_id": previous["current_task"]["main_id"],
                "current_sub_task_id": previous["current_task"]["sub_id"],
                "status": "researching",  # 재개
                # ... 기타 필드 ...
            }
            session_id = resume_session
        else:
            raise ValueError(f"Session {resume_session} not found")
    else:
        session_id = ProgressManager.generate_session_id(query)
        initial_state = {
            "query": query,
            "session_id": session_id,
            # ... 기존 초기화 ...
        }

    # ... 나머지 실행 로직 ...
```

### 파일 구조 예시
```
.research_progress/
├── 20260127_143022_a1b2c3d4.json
└── 20260127_150045_e5f6g7h8.json
```

### 테스트 방법
```python
# test_progress.py
async def test_resume_session():
    """Test research can be resumed from saved progress."""
    # 1. 연구 시작 후 중간에 중단
    # 2. session_id로 재개
    # 3. 이전 findings가 유지되는지 확인
```

---

## 개선 3: Backpressure (Quality Validation)

### 목표
검색 결과 품질을 자동 검증하여 불충분하면 재시도

### 구현 계획

#### 3.1 품질 검증 프롬프트 추가

**파일**: `src/prompts.py`

```python
QUALITY_VALIDATOR_PROMPT = """You are a research quality validator.

Original Query: {original_query}
Current Sub-task: {current_task}
Findings:
{findings}

Evaluate if the findings adequately answer the sub-task:

1. RELEVANCE: Do findings directly address the query?
2. COMPLETENESS: Is key information present?
3. SPECIFICITY: Are there concrete facts/data (not just vague statements)?

Output ONLY valid JSON:
{{
    "is_sufficient": true/false,
    "score": 0.0-1.0,
    "missing": ["list of missing information"],
    "feedback": "Brief explanation for the AI to improve"
}}

Be strict. If findings are vague or don't directly answer the query, mark as insufficient."""
```

#### 3.2 Validation 함수 추가

**파일**: `src/agent.py`

```python
from .prompts import QUALITY_VALIDATOR_PROMPT

def validate_findings_quality(
    original_query: str,
    current_task: str,
    findings: list[str],
    llm: ChatOpenAI,
    threshold: float = 0.6
) -> tuple[bool, str]:
    """Validate if findings adequately answer the task.

    Args:
        original_query: User's original research query
        current_task: Current sub-task being researched
        findings: Collected findings for this task
        llm: LLM instance
        threshold: Minimum score to pass (0.0-1.0)

    Returns:
        (is_valid, feedback) tuple
    """
    if not findings:
        return False, "No findings collected yet"

    prompt = QUALITY_VALIDATOR_PROMPT.format(
        original_query=original_query,
        current_task=current_task,
        findings="\n".join(findings[-3:])  # 최근 3개
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Validate these findings.")
    ]

    try:
        response = llm.invoke(messages)
        result = parse_json_response(response.content)

        if not isinstance(result, dict):
            return True, ""  # 파싱 실패시 통과

        score = float(result.get("score", 0.7))
        is_sufficient = result.get("is_sufficient", True)
        feedback = result.get("feedback", "")

        if score >= threshold and is_sufficient:
            return True, ""
        else:
            return False, feedback

    except Exception:
        return True, ""  # 에러시 통과
```

#### 3.3 State에 retry 필드 추가

**파일**: `src/state.py`

```python
class ResearchState(TypedDict):
    # ... 기존 필드들 ...

    # Backpressure 지원
    task_retry_count: dict[str, int]  # {task_id: retry_count}
    quality_feedback: str  # 품질 검증 피드백 (재시도시 사용)
```

#### 3.4 process_tool_results에 품질 검증 추가

**파일**: `src/agent.py`

```python
MAX_TASK_RETRIES = 2  # config.py에 추가

def process_tool_results(state: ResearchState) -> dict:
    """Process results with quality backpressure."""
    # ... 기존 findings 수집 로직 ...

    # 품질 검증 (Backpressure)
    current_task_id = state.get("current_sub_task_id")
    retry_counts = state.get("task_retry_count", {})
    current_retries = retry_counts.get(current_task_id, 0)

    quality_feedback = ""
    should_retry = False

    if new_findings and current_retries < MAX_TASK_RETRIES:
        llm = create_llm()
        current_query, _ = find_current_task(...)

        is_valid, feedback = validate_findings_quality(
            state["query"],
            current_query or "",
            new_findings,
            llm
        )

        if not is_valid:
            should_retry = True
            quality_feedback = feedback
            retry_counts[current_task_id] = current_retries + 1

    if should_retry:
        # 같은 task 재시도 - task 진행하지 않음
        return {
            "findings": state.get("findings", []) + new_findings,
            "task_retry_count": retry_counts,
            "quality_feedback": quality_feedback,
            # current_task_id 유지 (진행 안함)
            "current_main_task_id": state.get("current_main_task_id"),
            "current_sub_task_id": state.get("current_sub_task_id"),
        }
    else:
        # 다음 task로 진행
        next_main_id, next_sub_id = get_next_pending_task(updated_plan)
        return {
            "findings": state.get("findings", []) + new_findings,
            "task_retry_count": retry_counts,
            "quality_feedback": "",
            "current_main_task_id": next_main_id,
            "current_sub_task_id": next_sub_id,
            # ...
        }
```

#### 3.5 research_node에서 feedback 활용

**파일**: `src/agent.py`

```python
def research_node(state: ResearchState) -> dict:
    """Research with quality feedback."""
    # ...

    # 품질 피드백이 있으면 프롬프트에 추가
    quality_feedback = state.get("quality_feedback", "")
    feedback_section = ""
    if quality_feedback:
        feedback_section = f"""
IMPORTANT: Previous search was insufficient.
Feedback: {quality_feedback}
Please search more thoroughly and find specific information.
"""

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"""
{feedback_section}
Research query: {state['query']}
Current focus: {current_query}
...
""")
    ]
```

### 테스트 방법
```python
# test_backpressure.py
def test_quality_validation():
    """Test that low-quality findings trigger retry."""
    # 1. 모호한 findings 제공
    # 2. validation이 실패하는지 확인
    # 3. retry가 트리거되는지 확인
```

---

## 개선 4: Clear Exit Criteria

### 목표
명확한 종료 조건을 정의하여 무한 루프 방지 및 완성도 보장

### 구현 계획

#### 4.1 Exit Criteria 정의

**파일**: `src/config.py`

```python
# Exit Criteria Configuration
EXIT_CRITERIA = {
    "max_iterations": 15,           # 최대 iteration 수
    "max_replans": 3,               # 최대 replan 횟수
    "min_findings_per_task": 1,     # task당 최소 findings 수
    "max_task_retries": 2,          # task당 최대 재시도
    "completion_threshold": 0.8,    # 완료 판정 임계값
}
```

#### 4.2 Exit Checker 함수

**파일**: `src/agent.py`

```python
from .config import EXIT_CRITERIA

def check_exit_criteria(state: ResearchState) -> tuple[bool, str]:
    """Check if research should exit.

    Returns:
        (should_exit, reason) tuple
    """
    # 1. Iteration 제한
    if state.get("iteration", 0) >= EXIT_CRITERIA["max_iterations"]:
        return True, "max_iterations_reached"

    # 2. 모든 task 완료
    plan = state.get("hierarchical_plan")
    if plan:
        all_complete = all(
            mt["status"] == TaskStatus.COMPLETED.value
            for mt in plan["main_tasks"]
        )
        if all_complete:
            return True, "all_tasks_completed"

    # 3. 충분한 findings 수집
    findings = state.get("findings", [])
    plan = state.get("hierarchical_plan")
    if plan:
        total_tasks = sum(len(mt["sub_tasks"]) for mt in plan["main_tasks"])
        min_required = total_tasks * EXIT_CRITERIA["min_findings_per_task"]
        if len(findings) >= min_required:
            # 완료율 체크
            completed_tasks = sum(
                1 for mt in plan["main_tasks"]
                for st in mt["sub_tasks"]
                if st["status"] == TaskStatus.COMPLETED.value
            )
            completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            if completion_rate >= EXIT_CRITERIA["completion_threshold"]:
                return True, "completion_threshold_met"

    # 4. Replan 제한
    if state.get("replan_count", 0) >= EXIT_CRITERIA["max_replans"]:
        # replan 제한에 도달했지만 계속 진행은 가능
        pass

    return False, ""


def should_continue(state: ResearchState) -> Literal["research", "tools", "synthesize", "end"]:
    """Determine next step with exit criteria check."""

    # Exit criteria 체크
    should_exit, reason = check_exit_criteria(state)
    if should_exit:
        # 이유를 로깅
        print(f"Exit criteria met: {reason}")
        return "synthesize"

    # ... 기존 로직 ...
```

#### 4.3 Completion Report 생성

**파일**: `src/agent.py`

```python
def generate_completion_report(state: ResearchState) -> dict:
    """Generate a completion status report.

    Returns:
        Report dict with completion statistics
    """
    plan = state.get("hierarchical_plan")

    if not plan:
        return {"status": "no_plan"}

    total_tasks = sum(len(mt["sub_tasks"]) for mt in plan["main_tasks"])
    completed_tasks = sum(
        1 for mt in plan["main_tasks"]
        for st in mt["sub_tasks"]
        if st["status"] == TaskStatus.COMPLETED.value
    )

    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
        "total_findings": len(state.get("findings", [])),
        "iterations_used": state.get("iteration", 0),
        "replans_used": state.get("replan_count", 0),
        "exit_reason": state.get("exit_reason", "unknown"),
    }
```

### 테스트 방법
```python
# test_exit_criteria.py
def test_max_iterations_exit():
    """Test that max iterations triggers exit."""
    state = {"iteration": 15, ...}
    should_exit, reason = check_exit_criteria(state)
    assert should_exit == True
    assert reason == "max_iterations_reached"
```

---

## 구현 우선순위

| 순위 | 개선 | 난이도 | 영향도 | 예상 시간 |
|------|------|--------|--------|-----------|
| 1 | Exit Criteria | 낮음 | 높음 | 1-2시간 |
| 2 | Fresh Context | 중간 | 높음 | 2-3시간 |
| 3 | File-based Memory | 중간 | 중간 | 3-4시간 |
| 4 | Backpressure | 높음 | 높음 | 4-5시간 |

---

## 체크리스트

### 개선 1: Fresh Context
- [ ] `summarize_findings()` 함수 구현
- [ ] `research_node()` 수정 - fresh messages
- [ ] `findings_summary` state 필드 추가
- [ ] 유닛 테스트 작성

### 개선 2: File-based Memory
- [ ] `src/progress.py` 파일 생성
- [ ] `ProgressManager` 클래스 구현
- [ ] `process_tool_results()`에 저장 로직 추가
- [ ] `run_research_with_tools()`에 resume 파라미터 추가
- [ ] `.gitignore`에 `.research_progress/` 추가
- [ ] 유닛 테스트 작성

### 개선 3: Backpressure
- [ ] `QUALITY_VALIDATOR_PROMPT` 추가
- [ ] `validate_findings_quality()` 함수 구현
- [ ] `task_retry_count`, `quality_feedback` state 필드 추가
- [ ] `process_tool_results()`에 validation 로직 추가
- [ ] `research_node()`에 feedback 활용 로직 추가
- [ ] `MAX_TASK_RETRIES` config 추가
- [ ] 유닛 테스트 작성

### 개선 4: Exit Criteria
- [ ] `EXIT_CRITERIA` config 추가
- [ ] `check_exit_criteria()` 함수 구현
- [ ] `should_continue()`에 exit check 통합
- [ ] `generate_completion_report()` 함수 구현
- [ ] 유닛 테스트 작성

---

## 참고 자료

- [Ralph Loop GitHub](https://github.com/snarktank/ralph)
- [Vercel Ralph Loop Agent](https://github.com/vercel-labs/ralph-loop-agent)
- [Ralph Playbook](https://claytonfarr.github.io/ralph-playbook/)
