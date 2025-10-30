---
title: LangGraph Sample Project
authors:
  - BZ
date: 2025-10-02
categories: 
  - LLM
---

<!-- more -->
# LangGraph Sample Project

## Objective

- [x] **Independent deployable services**
  - Each agent can scale horizontally (e.g., analysis_service replicas)
  - You can version and deploy agents independently

- [x] **Schema isolation**
	- Each service defines its own `Pydantic` input/output
	- Supervisor does schema translation

- [x] **Resilience**
	- Supervisor can retry subgraph calls, add timeout handling

- [x] **Observability**
	- You can trace inter-agent calls via `httpx` middleware or `OpenTelemetry`

- [x] **Extensible**
	- Just add new agents (summarizer_service, retriever_service, etc.)
	- Supervisor graph can grow dynamically without coupling

### Project Structure

🎯 What's Included:

- **3 Independent Services:**
    - Research Service (Port **8081**) - Handles research queries with validation, planning, gathering, and summarization
    - Analysis Service (Port **8082**) - Extracts insights, generates recommendations, and creates analysis reports
    - Supervisor Service (Port **8080**) - Orchestrates the entire workflow via REST API calls
- **Key Features:**

    - ✅ Independent LangGraph workflows in each service
    - ✅ Shared error handler with per-node retry tracking
    - ✅ REST API communication between services
    - ✅ Parallel node support - no state conflicts
    - ✅ Docker deployment ready with docker-compose
    - ✅ Health checks for monitoring
    - ✅ Comprehensive error handling with automatic retries
    - ✅ Test scripts for validation




```sh title="Microservice-style LangGraph architecture"
agentic_system/
├── shared/
│   └── error_handler.py          # Shared error handling logic
│
├── research_service/
│   ├── main.py                    # FastAPI app
│   ├── schema.py                  # Pydantic models
│   ├── graph.py                   # LangGraph workflow
│   ├── requirements.txt
│   └── Dockerfile
│
├── analysis_service/
│   ├── main.py                    # FastAPI app
│   ├── schema.py                  # Pydantic models
│   ├── graph.py                   # LangGraph workflow
│   ├── requirements.txt
│   └── Dockerfile
│
├── supervisor_service/
│   ├── main.py                    # FastAPI app
│   ├── schema.py                  # Pydantic models
│   ├── graph.py                   # LangGraph workflow
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml
├── run_services.sh
└── test_system.py
```

## Services

??? "Research Service"

    ```sh
      ├── research_service/
      ├── main.py                    # FastAPI app
      ├── schema.py                  # Pydantic models
      ├── graph.py                   # LangGraph workflow
    ```

    ??? "codes: research_service/"
        ```python linenums="1" title="schema.py"
          from typing import List, Optionalfrom pydantic import BaseModel, Field
          from typing import Optional


          class ResearchState(BaseModel):
              query: str = ""
              research_plan: str = ""
              search_results: list[str] = Field(default_factory=list)
              summary: str = ""

              # Error handling
              error_messages: list[str] = Field(default_factory=list)
              retry_count: int = 0
              max_retries: int = 3
              failed_nodes: dict[str, int] = Field(default_factory=dict)

              class Config:
                  arbitrary_types_allowed = True


          class ResearchRequest(BaseModel):
              query: str
              max_retries: int = 3


          class ResearchResponse(BaseModel):
              query: str
              research_plan: str
              search_results: list[str]
              summary: str
              error_messages: list[str]
              failed_nodes: dict[str, int]
              success: bool
        ```

        ```python linenums="1" title="graph.py"
          import logging
          from typing import Optional

          from langgraph.graph import StateGraph, END
          from langchain_ollama import ChatOllama
          from langchain_core.messages import HumanMessage, SystemMessage

          from agentic_app.research_service.schema import ResearchState
          from agentic_app.shared.error_handler import handle_node_errors, create_universal_router

          logger = logging.getLogger(__name__)


          class ResearchNodes:
              def __init__(self, llm: Optional[ChatOllama] = None):
                  self.llm = llm or ChatOllama(model="gpt-oss", temperature=0)

              @handle_node_errors("validate_query", "Failed to validate query")
              def validate_query(self, state: ResearchState) -> dict:
                  logger.info(f"Validating query: {state.query}")

                  if not state.query or len(state.query.strip()) < 5:
                      raise ValueError("Query must be at least 5 characters long")

                  return {}

              @handle_node_errors("create_plan", "Failed to create research plan")
              async def create_plan(self, state: ResearchState) -> dict:
                  logger.info("Creating research plan")

                  messages = [
                      SystemMessage(content="Create a brief 3-step research plan."),
                      HumanMessage(content=f"Create a research plan for: {state.query}")
                  ]

                  response = await self.llm.ainvoke(messages)

                  return {
                      "research_plan": response.content
                  }

              @handle_node_errors("gather_info", "Failed to gather information")
              async def gather_info(self, state: ResearchState) -> dict:
                  logger.info("Gathering information")

                  # Simulate research gathering
                  search_results = [
                      f"Finding 1 about {state.query}",
                      f"Finding 2 about {state.query}",
                      f"Finding 3 about {state.query}",
                  ]

                  return {
                      "search_results": search_results
                  }

              @handle_node_errors("summarize", "Failed to summarize")
              async def summarize(self, state: ResearchState) -> dict:
                  logger.info("Summarizing findings")

                  findings = "\n".join(f"- {r}" for r in state.search_results)

                  messages = [
                      SystemMessage(content="Summarize these research findings concisely."),
                      HumanMessage(content=f"Plan: {state.research_plan}\n\nFindings:\n{findings}")
                  ]

                  response = await self.llm.ainvoke(messages)

                  return {
                      "summary": response.content
                  }


          def create_research_graph():
              nodes = ResearchNodes()
              workflow = StateGraph(ResearchState)

              workflow.add_node("validate_query", nodes.validate_query)
              workflow.add_node("create_plan", nodes.create_plan)
              workflow.add_node("gather_info", nodes.gather_info)
              workflow.add_node("summarize", nodes.summarize)

              workflow.set_entry_point("validate_query")

              workflow.add_conditional_edges(
                  "validate_query",
                  create_universal_router(next_node="create_plan", node_name="validate_query")
              )
              workflow.add_conditional_edges(
                  "create_plan",
                  create_universal_router(next_node="gather_info", node_name="create_plan")
              )
              workflow.add_conditional_edges(
                  "gather_info",
                  create_universal_router(next_node="summarize", node_name="gather_info")
              )
              workflow.add_conditional_edges(
                  "summarize",
                  create_universal_router(next_node=END, node_name="summarize")
              )

              return workflow.compile()


        ```


        ```python linenums="1" title="main.py"
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            import logging

            from agentic_app.analysis_service.graph import create_analysis_graph
            from agentic_app.analysis_service.schema import AnalysisState, AnalysisRequest, AnalysisResponse

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            app = FastAPI(title="Analysis Service", version="1.0.0")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            analysis_graph = create_analysis_graph()


            @app.post("/analyze", response_model=AnalysisResponse)
            async def analyze_research(request: AnalysisRequest):
                """Analyze research summary and generate insights"""
                try:
                    logger.info("Received analysis request")

                    initial_state = AnalysisState(
                        research_summary=request.research_summary,
                        max_retries=request.max_retries
                    )

                    final_state = await analysis_graph.ainvoke(initial_state)

                    return AnalysisResponse(
                        insights=final_state.get("insights", []),
                        recommendations=final_state.get("recommendations", []),
                        final_analysis=final_state.get("final_analysis", ""),
                        error_messages=final_state.get("error_messages", []),
                        failed_nodes=final_state.get("failed_nodes", {}),
                        success=len(final_state.get("error_messages", [])) == 0
                    )

                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))


            @app.get("/health")
            async def health():
                return {"status": "healthy", "service": "analysis"}

            if __name__ == '__main__':
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8082)
        ```

??? "Analysis Service"

    ```sh
     ├── analysis_service/
     ├── main.py                    # FastAPI app
     ├── schema.py                  # Pydantic models
     ├── graph.py    
    ```

    ??? "codes: analysis_service/"

        ```python linenums="1" title="schema.py"
          from pydantic import BaseModel, Field, ConfigDict

          class AnalysisState(BaseModel):
              model_config = ConfigDict(arbitrary_types_allowed=True)
              research_summary: str = ""
              insights: list[str] = Field(default_factory=list)
              recommendations: list[str] = Field(default_factory=list)
              final_analysis: str = ""

              # Error handling
              error_messages: list[str] = Field(default_factory=list)
              retry_count: int = 0
              max_retries: int = 3
              failed_nodes: dict[str, int] = Field(default_factory=dict)


          class AnalysisRequest(BaseModel):
              research_summary: str
              max_retries: int = 3


          class AnalysisResponse(BaseModel):
              insights: list[str]
              recommendations: list[str]
              final_analysis: str
              error_messages: list[str]
              failed_nodes: dict[str, int]
              success: bool
        ```

        ```python linenums="1" title="graph.py"

            import logging
            from typing import Optional

            from langgraph.graph import StateGraph, END
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            from agentic_app.analysis_service.schema import AnalysisState
            from agentic_app.shared.error_handler import handle_node_errors, create_universal_router

            logger = logging.getLogger(__name__)


            class AnalysisNodes:
                def __init__(self, llm: Optional[ChatOllama] = None):
                    self.llm = llm or ChatOllama(model="gpt-oss", temperature=0)

                @handle_node_errors("extract_insights", "Failed to extract insights")
                async def extract_insights(self, state: AnalysisState) -> dict:
                    logger.info("Extracting insights")

                    messages = [
                        SystemMessage(content="Extract 3 key insights from this research."),
                        HumanMessage(content=state.research_summary)
                    ]

                    response = await self.llm.ainvoke(messages)

                    # Parse insights (simplified)
                    insights = [line.strip() for line in response.content.split('\n') if line.strip()][:3]

                    return {
                        "insights": insights
                    }

                @handle_node_errors("generate_recommendations", "Failed to generate recommendations")
                async def generate_recommendations(self, state: AnalysisState) -> dict:
                    logger.info("Generating recommendations")

                    insights_text = "\n".join(f"- {i}" for i in state.insights)

                    messages = [
                        SystemMessage(content="Generate 3 actionable recommendations based on these insights."),
                        HumanMessage(content=insights_text)
                    ]

                    response = await self.llm.ainvoke(messages)

                    recommendations = [line.strip() for line in response.content.split('\n') if line.strip()][:3]

                    return {
                        "recommendations": recommendations
                    }

                @handle_node_errors("create_analysis", "Failed to create final analysis")
                async def create_analysis(self, state: AnalysisState) -> dict:
                    logger.info("Creating final analysis")

                    messages = [
                        SystemMessage(content="Create a concise final analysis report."),
                        HumanMessage(
                            content=f"Summary: {state.research_summary}\n\nInsights: {state.insights}\n\nRecommendations: {state.recommendations}")
                    ]

                    response = await self.llm.ainvoke(messages)

                    return {
                        "final_analysis": response.content
                    }


            def create_analysis_graph():
                nodes = AnalysisNodes()
                workflow = StateGraph(AnalysisState)

                workflow.add_node("extract_insights", nodes.extract_insights)
                workflow.add_node("generate_recommendations", nodes.generate_recommendations)
                workflow.add_node("create_analysis", nodes.create_analysis)

                workflow.set_entry_point("extract_insights")

                workflow.add_conditional_edges(
                    "extract_insights",
                    create_universal_router(next_node="generate_recommendations", node_name="extract_insights")
                )
                workflow.add_conditional_edges(
                    "generate_recommendations",
                    create_universal_router(next_node="create_analysis", node_name="generate_recommendations")
                )
                workflow.add_conditional_edges(
                    "create_analysis",
                    create_universal_router(next_node=END, node_name="create_analysis")
                )

                return workflow.compile()
        ```

        ```python linenums="1" title="main.py"

            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            import logging

            from agentic_app.analysis_service.graph import create_analysis_graph
            from agentic_app.analysis_service.schema import AnalysisState, AnalysisRequest, AnalysisResponse

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            app = FastAPI(title="Analysis Service", version="1.0.0")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            analysis_graph = create_analysis_graph()


            @app.post("/analyze", response_model=AnalysisResponse)
            async def analyze_research(request: AnalysisRequest):
                """Analyze research summary and generate insights"""
                try:
                    logger.info("Received analysis request")

                    initial_state = AnalysisState(
                        research_summary=request.research_summary,
                        max_retries=request.max_retries
                    )

                    final_state = await analysis_graph.ainvoke(initial_state)

                    return AnalysisResponse(
                        insights=final_state.get("insights", []),
                        recommendations=final_state.get("recommendations", []),
                        final_analysis=final_state.get("final_analysis", ""),
                        error_messages=final_state.get("error_messages", []),
                        failed_nodes=final_state.get("failed_nodes", {}),
                        success=len(final_state.get("error_messages", [])) == 0
                    )

                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))


            @app.get("/health")
            async def health():
                return {"status": "healthy", "service": "analysis"}

            if __name__ == '__main__':
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8082)
        ```







??? "Supervisor Service"

    ```sh
     ├── supervisor_service/
     ├── main.py                    # FastAPI app
     ├── schema.py                  # Pydantic models
     ├── graph.py                   # LangGraph workflow
    ```

    ??? "codes: analysis_service/"

        ```python linenums="1" title="schema.py"

          from pydantic import BaseModel, Field
          from typing import Optional


          class SupervisorState(BaseModel):
              original_query: str = ""
              research_result: dict = Field(default_factory=dict)
              analysis_result: dict = Field(default_factory=dict)
              final_report: str = ""

              # Error handling
              error_messages: list[str] = Field(default_factory=list)
              retry_count: int = 0
              max_retries: int = 3
              failed_nodes: dict[str, int] = Field(default_factory=dict)

              class Config:
                  arbitrary_types_allowed = True


          class SupervisorRequest(BaseModel):
              query: str
              max_retries: int = 3
              research_service_url: str = "http://localhost:8081"
              analysis_service_url: str = "http://localhost:8082"


          class SupervisorResponse(BaseModel):
              query: str
              research_summary: str
              analysis_report: str
              final_report: str
              error_messages: list[str]
              failed_nodes: dict[str, int]
              success: bool

        ```

        ```python linenums="1" title="graph.py"

            import logging
            from typing import Optional

            import httpx
            from langgraph.graph import StateGraph, END
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage

            from agentic_app.shared.error_handler import handle_node_errors, create_universal_router
            from agentic_app.supervisor_service.schema import SupervisorState

            logger = logging.getLogger(__name__)


            class SupervisorNodes:
                def __init__(self, research_url: str, analysis_url: str, llm: Optional[ChatOllama] = None):
                    self.research_url = research_url
                    self.analysis_url = analysis_url
                    self.llm = llm or ChatOllama(model="gpt-oss", temperature=0)

                @handle_node_errors("call_research", "Failed to call research service")
                async def call_research(self, state: SupervisorState) -> dict:
                    logger.info(f"Calling research service at {self.research_url}")

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        response = await client.post(
                            f"{self.research_url}/research",
                            json={"query": state.original_query, "max_retries": state.max_retries}
                        )
                        response.raise_for_status()
                        result = response.json()

                    if not result.get("success"):
                        raise Exception(f"Research service failed: {result.get('error_messages')}")

                    return {
                        "research_result": result
                    }

                @handle_node_errors("call_analysis", "Failed to call analysis service")
                async def call_analysis(self, state: SupervisorState) -> dict:
                    logger.info(f"Calling analysis service at {self.analysis_url}")

                    research_summary = state.research_result.get("summary", "")

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        response = await client.post(
                            f"{self.analysis_url}/analyze",
                            json={"research_summary": research_summary, "max_retries": state.max_retries}
                        )
                        response.raise_for_status()
                        result = response.json()

                    if not result.get("success"):
                        raise Exception(f"Analysis service failed: {result.get('error_messages')}")

                    return {
                        "analysis_result": result
                    }

                @handle_node_errors("generate_final_report", "Failed to generate final report")
                async def generate_final_report(self, state: SupervisorState) -> dict:
                    logger.info("Generating final report")

                    messages = [
                        SystemMessage(content="Create a comprehensive final report combining research and analysis."),
                        HumanMessage(
                            content=f"Query: {state.original_query}\n\nResearch: {state.research_result.get('summary')}\n\nAnalysis: {state.analysis_result.get('final_analysis')}")
                    ]

                    response = await self.llm.ainvoke(messages)

                    return {
                        "final_report": response.content
                    }


            def create_supervisor_graph(research_url: str, analysis_url: str):
                nodes = SupervisorNodes(research_url, analysis_url)
                workflow = StateGraph(SupervisorState)

                workflow.add_node("call_research", nodes.call_research)
                workflow.add_node("call_analysis", nodes.call_analysis)
                workflow.add_node("generate_final_report", nodes.generate_final_report)

                workflow.set_entry_point("call_research")

                workflow.add_conditional_edges(
                    "call_research",
                    create_universal_router(next_node="call_analysis", node_name="call_research")
                )
                workflow.add_conditional_edges(
                    "call_analysis",
                    create_universal_router(next_node="generate_final_report", node_name="call_analysis")
                )
                workflow.add_conditional_edges(
                    "generate_final_report",
                    create_universal_router(next_node=END, node_name="generate_final_report")
                )

                return workflow.compile()
        ```

        ```python linenums="1" title="main.py"

            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            import logging

            from agentic_app.research_service.graph import create_research_graph
            from agentic_app.research_service.schema import ResearchResponse, ResearchRequest, ResearchState

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            app = FastAPI(title="Research Service", version="1.0.0")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Initialize graph
            research_graph = create_research_graph()


            @app.post("/research", response_model=ResearchResponse)
            async def conduct_research(request: ResearchRequest):
                """Conduct research on a given query"""
                try:
                    logger.info(f"Received research request: {request.query}")

                    initial_state = ResearchState(
                        query=request.query,
                        max_retries=request.max_retries
                    )

                    final_state = await research_graph.ainvoke(initial_state)

                    return ResearchResponse(
                        query=final_state.get("query", ""),
                        research_plan=final_state.get("research_plan", ""),
                        search_results=final_state.get("search_results", []),
                        summary=final_state.get("summary", ""),
                        error_messages=final_state.get("error_messages", []),
                        failed_nodes=final_state.get("failed_nodes", {}),
                        success=len(final_state.get("error_messages", [])) == 0
                    )

                except Exception as e:
                    logger.error(f"Research failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))


            @app.get("/health")
            async def health():
                return {"status": "healthy", "service": "research"}


            if __name__ == '__main__':
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=8081)
        ```

??? "Others"

    ```python linenums="1" title="error_handler.py"
    import asyncio
    import logging
    from functools import wraps
    from typing import Protocol, Optional, TypeVar, Any, Callable

    logger = logging.getLogger(__name__)


    class ErrorState(Protocol):
        error_messages: list[str]
        retry_count: int
        max_retries: int
        failed_nodes: dict[str, int]


    StateType = TypeVar('StateType', bound=ErrorState)


    class ErrorHandler:
        @staticmethod
        def handle_error(state: StateType, error: Exception, node_name: str, custom_message: Optional[str] = None) -> dict:
            error_msg = custom_message or f"Error in {node_name}: {str(error)}"
            logger.error(f"Node '{node_name}' failed: {str(error)}")

            failed_nodes = dict(
                getattr(state, 'failed_nodes', {}) if hasattr(state, 'failed_nodes') else state.get('failed_nodes', {}))
            node_retry_count = failed_nodes.get(node_name, 0) + 1
            failed_nodes[node_name] = node_retry_count

            retry_count = getattr(state, 'retry_count', 0) if hasattr(state, 'retry_count') else state.get('retry_count', 0)

            return {
                "error_messages": [f"{node_name}: {error_msg}"],
                "retry_count": retry_count + 1,
                "failed_nodes": failed_nodes
            }

        @staticmethod
        def should_retry(state: StateType, node_name: Optional[str] = None) -> bool:
            if node_name:
                failed_nodes = getattr(state, 'failed_nodes', {}) if hasattr(state, 'failed_nodes') else state.get(
                    'failed_nodes', {})
                node_retry_count = failed_nodes.get(node_name, 0)
                max_retries = getattr(state, 'max_retries', 3) if hasattr(state, 'max_retries') else state.get(
                    'max_retries', 3)
                error_messages = getattr(state, 'error_messages', []) if hasattr(state, 'error_messages') else state.get(
                    'error_messages', [])
                has_node_error = any(node_name in msg for msg in error_messages)
                return node_retry_count < max_retries and has_node_error
            else:
                retry_count = getattr(state, 'retry_count', 0) if hasattr(state, 'retry_count') else state.get(
                    'retry_count', 0)
                max_retries = getattr(state, 'max_retries', 3) if hasattr(state, 'max_retries') else state.get(
                    'max_retries', 3)
                error_messages = getattr(state, 'error_messages', []) if hasattr(state, 'error_messages') else state.get(
                    'error_messages', [])
                return retry_count < max_retries and len(error_messages) > 0

        @staticmethod
        def clear_errors(state: StateType) -> dict[str, Any]:
            return {
                "error_messages": [],
                "retry_count": 0,
            }


    def handle_node_errors(node_name: str, custom_message: Optional[str] = None):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(self, state: StateType) -> dict[str, Any]:
                try:
                    result = await func(self, state)
                    if result is None:
                        result = {}
                    result.update(ErrorHandler.clear_errors(state))
                    return result
                except Exception as e:
                    logger.exception(f"Error in async node '{node_name}'")
                    return ErrorHandler.handle_error(state, e, node_name, custom_message)

            @wraps(func)
            def sync_wrapper(self, state: StateType) -> dict[str, Any]:
                try:
                    result = func(self, state)
                    if result is None:
                        result = {}
                    result.update(ErrorHandler.clear_errors(state))
                    return result
                except Exception as e:
                    logger.exception(f"Error in sync node '{node_name}'")
                    return ErrorHandler.handle_error(state, e, node_name, custom_message)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


    def create_universal_router(next_node: str, end_node: str = "END", node_name: Optional[str] = None):
        def router(state) -> str:
            if isinstance(state, dict):
                error_messages = state.get('error_messages', [])
                max_retries = state.get('max_retries', 3)
                failed_nodes = state.get('failed_nodes', {})
            else:
                error_messages = state.error_messages
                max_retries = state.max_retries
                failed_nodes = state.failed_nodes

            if len(error_messages) > 0 and node_name:
                has_node_error = any(node_name in msg for msg in error_messages)

                if has_node_error:
                    node_retry_count = failed_nodes.get(node_name, 0)
                    if node_retry_count < max_retries:
                        logger.info(f"Retrying {node_name}, attempt {node_retry_count}/{max_retries}")
                        return node_name
                    else:
                        logger.error(f"Max retries reached for {node_name}, ending execution")
                        return end_node

            return next_node

        return router

    ```


    ```sh linenums="1" title="run_services.sh"
    # Terminal 1 - Research Service
    cd research_service
    uvicorn main:app --reload --port 8001

    # Terminal 2 - Analysis Service
    cd analysis_service
    uvicorn main:app --reload --port 8002

    # Terminal 3 - Supervisor Service
    cd supervisor_service
    uvicorn main:app --reload --port 8000
    ```










## ~~Codes Explanation~~


??? "Full Code(old)"

    ??? graph
        ```mermaid
          graph TD
            Start([Start]) --> ValidateQuery[Validate Query]
            
            ValidateQuery -->|Success| CreatePlan[Create Research Plan]
            ValidateQuery -->|Error & Retries| ValidateQuery
            ValidateQuery -->|Error & Max Retries| End([End])
            
            CreatePlan -->|Success| GatherInfo[Gather Information]
            CreatePlan -->|Error & Retries| CreatePlan
            CreatePlan -->|Error & Max Retries| End
            
            GatherInfo -->|Success| Synthesize[Synthesize Findings]
            GatherInfo -->|Error & Retries| GatherInfo
            GatherInfo -->|Error & Max Retries| End
            
            Synthesize -->|Success| GenerateReport[Generate Report]
            Synthesize -->|Error & Retries| Synthesize
            Synthesize -->|Error & Max Retries| End
            
            GenerateReport -->|Success| End
            GenerateReport -->|Error & Retries| GenerateReport
            GenerateReport -->|Error & Max Retries| End
            
            style ValidateQuery fill:#e1f5ff
            style CreatePlan fill:#e1f5ff
            style GatherInfo fill:#e1f5ff
            style Synthesize fill:#e1f5ff
            style GenerateReport fill:#e1f5ff
            style Start fill:#d4edda
            style End fill:#f8d7da
        ```

    ```python linenums="1"
    import asyncio
    import logging
    from functools import wraps
    from typing import Protocol, Optional, TypeVar, Any, Callable, Annotated
    from langgraph.graph import StateGraph, END
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph.state import CompiledStateGraph
    from pydantic import BaseModel, Field

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    # ============================================================================
    # ERROR HANDLER (from your code)
    # ============================================================================

    class ErrorState(Protocol):
        error_messages: list[str]
        retry_count: int
        max_retries: int
        last_failed_node: Optional[str]
        current_node: Optional[str]


    StateType = TypeVar('StateType', bound=ErrorState)


    class ErrorHandler:
        @staticmethod
        def handle_error(state: StateType, error: Exception, node_name: str, custom_message: Optional[str] = None) -> dict:
            error_msg = custom_message or f"Error in {node_name}: {str(error)}"
            logger.error(f"Node '{node_name}' failed: {str(error)}")

            return {
                "error_messages": [error_msg],
                "retry_count": state.retry_count + 1,
                "last_failed_node": node_name,
                "current_node": node_name
            }

        @staticmethod
        def should_retry(state: StateType) -> bool:
            return state.retry_count < state.max_retries and len(state.error_messages) > 0

        @staticmethod
        def clear_errors(state: StateType) -> dict[str, Any]:
            return {
                "error_messages": [],
                "retry_count": 0,
                "last_failed_node": None,
            }

        @staticmethod
        def get_error_summary(state: StateType) -> dict[str, Any]:
            return {
                "has_errors": len(state.error_messages) > 0,
                "error_count": len(state.error_messages),
                "retry_count": state.retry_count,
                "last_failed_node": state.last_failed_node,
                "can_retry": ErrorHandler.should_retry(state)
            }


    def handle_node_errors(node_name: str, custom_message: Optional[str] = None):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(self, state: StateType) -> dict[str, Any]:
                try:
                    result = await func(self, state)
                    if result is None:
                        result = {}
                    result.update(ErrorHandler.clear_errors(state))
                    return result
                except Exception as e:
                    logger.exception(f"Error in async node '{node_name}'")
                    return ErrorHandler.handle_error(state, e, node_name, custom_message)

            @wraps(func)
            def sync_wrapper(self, state: StateType) -> dict[str, Any]:
                try:
                    result = func(self, state)
                    if result is None:
                        result = {}
                    result.update(ErrorHandler.clear_errors(state))
                    return result
                except Exception as e:
                    logger.exception(f"Error in sync node '{node_name}'")
                    return ErrorHandler.handle_error(state, e, node_name, custom_message)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


    # ============================================================================
    # STATE DEFINITION
    # ============================================================================

    class ResearchState(BaseModel):
        """State for the research assistant workflow"""
        query: str = ""
        research_plan: str = ""
        search_results: list[str] = Field(default_factory=list)
        summary: str = ""
        final_report: str = ""

        # Error handling fields
        error_messages: list[str] = Field(default_factory=list)
        retry_count: int = 0
        max_retries: int = 3
        last_failed_node: Optional[str] = None
        current_node: Optional[str] = None

        # Control flow
        should_continue: bool = True

        class Config:
            arbitrary_types_allowed = True


    # ============================================================================
    # RESEARCH NODES
    # ============================================================================

    class ResearchNodes:
        """Collection of nodes for the research workflow"""

        def __init__(self, llm: Optional[ChatOllama] = None):
            self.llm = llm or ChatOllama(model="gpt-oss", temperature=0)

        @handle_node_errors("validate_query", "Failed to validate the research query")
        def validate_query(self, state: ResearchState) -> dict[str, Any]:
            """Validate that the query is appropriate for research"""
            logger.info(f"Validating query: {state.query}")

            if not state.query or len(state.query.strip()) < 5:
                raise ValueError("Query must be at least 5 characters long")

            # Simulate potential validation issues
            if "error" in state.query.lower():
                raise ValueError("Query contains forbidden terms")

            return {
                "current_node": "validate_query",
                "should_continue": True
            }

        @handle_node_errors("create_research_plan", "Failed to create research plan")
        async def create_research_plan(self, state: ResearchState) -> dict[str, Any]:
            """Create a research plan based on the query"""
            logger.info(f"Creating research plan for: {state.query}")

            messages = [
                SystemMessage(content="You are a research planning assistant. Create a brief 3-step research plan."),
                HumanMessage(content=f"Create a research plan for: {state.query}")
            ]

            response = await self.llm.ainvoke(messages)

            if not response.content:
                raise ValueError("LLM returned empty research plan")

            return {
                "research_plan": response.content,
                "current_node": "create_research_plan",
                "should_continue": True
            }

        @handle_node_errors("gather_information", "Failed to gather information")
        async def gather_information(self, state: ResearchState) -> dict[str, Any]:
            """Simulate gathering information from various sources"""
            logger.info("Gathering information...")

            # Simulate API calls that might fail
            await asyncio.sleep(0.5)

            # Simulate random failures for demonstration
            import random
            if random.random() < 0.2:  # 20% chance of failure
                raise ConnectionError("Failed to connect to research database")

            # Simulate search results
            search_results = [
                f"Research finding 1 about {state.query}",
                f"Research finding 2 about {state.query}",
                f"Research finding 3 about {state.query}",
            ]

            return {
                "search_results": search_results,
                "current_node": "gather_information",
                "should_continue": True
            }

        @handle_node_errors("synthesize_findings", "Failed to synthesize findings")
        async def synthesize_findings(self, state: ResearchState) -> dict[str, Any]:
            """Synthesize the gathered information into a summary"""
            logger.info("Synthesizing findings...")

            if not state.search_results:
                raise ValueError("No search results available to synthesize")

            findings_text = "\n".join(f"- {result}" for result in state.search_results)

            messages = [
                SystemMessage(content="You are a research synthesis assistant. Summarize the findings concisely."),
                HumanMessage(
                    content=f"Research Plan:\n{state.research_plan}\n\nFindings:\n{findings_text}\n\nProvide a brief summary.")
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "summary": response.content,
                "current_node": "synthesize_findings",
                "should_continue": True
            }

        @handle_node_errors("generate_report", "Failed to generate final report")
        async def generate_report(self, state: ResearchState) -> dict[str, Any]:
            """Generate the final research report"""
            logger.info("Generating final report...")

            messages = [
                SystemMessage(content="You are a report writing assistant. Create a concise final report."),
                HumanMessage(content=f"Query: {state.query}\n\nSummary: {state.summary}\n\nCreate a final report.")
            ]

            response = await self.llm.ainvoke(messages)

            return {
                "final_report": response.content,
                "current_node": "generate_report",
                "should_continue": False
            }


    # ============================================================================
    # ROUTING LOGIC
    # ============================================================================


    def create_universal_router(next_node: str, end_node: str = END):
        """Create a universal router that handles errors and retries"""

        def router(state) -> str:
            # Handle both dict and Pydantic model
            if isinstance(state, dict):
                error_messages = state.get('error_messages', [])
                retry_count = state.get('retry_count', 0)
                max_retries = state.get('max_retries', 3)
                last_failed_node = state.get('last_failed_node', 'validate_query')
            else:
                error_messages = state.error_messages
                retry_count = state.retry_count
                max_retries = state.max_retries
                last_failed_node = state.last_failed_node or 'validate_query'

            if len(error_messages) > 0:
                if retry_count < max_retries:
                    logger.info(f"Retrying {last_failed_node}, attempt {retry_count}/{max_retries}")
                    return last_failed_node
                else:
                    logger.error(f"Max retries reached for {last_failed_node}, ending execution")
                    return end_node
            else:
                return next_node

        return router


    def should_retry_node(state) -> str:
        """Route back to the failed node for retry"""
        if isinstance(state, dict):
            last_failed = state.get('last_failed_node', 'validate_query')
        else:
            last_failed = state.last_failed_node or 'validate_query'

        logger.info(f"Routing to retry node: {last_failed}")
        return last_failed





    # ============================================================================
    # GRAPH CONSTRUCTION
    # ============================================================================

    def create_research_graph(llm: Optional[ChatOllama] = None) -> CompiledStateGraph:
        """Create the research assistant graph with error handling"""

        nodes = ResearchNodes(llm)

        # Create the graph
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("validate_query", nodes.validate_query)
        workflow.add_node("create_research_plan", nodes.create_research_plan)
        workflow.add_node("gather_information", nodes.gather_information)
        workflow.add_node("synthesize_findings", nodes.synthesize_findings)
        workflow.add_node("generate_report", nodes.generate_report)

        # Set entry point
        workflow.set_entry_point("validate_query")

        # Add conditional edges using universal router
        # Router will automatically retry the failed node or move to next node
        workflow.add_conditional_edges(
            "validate_query",
            create_universal_router(next_node="create_research_plan")
        )
        workflow.add_conditional_edges(
            "create_research_plan",
            create_universal_router(next_node="gather_information")
        )
        workflow.add_conditional_edges(
            "gather_information",
            create_universal_router(next_node="synthesize_findings")
        )
        workflow.add_conditional_edges(
            "synthesize_findings",
            create_universal_router(next_node="generate_report")
        )
        workflow.add_conditional_edges(
            "generate_report",
            create_universal_router(next_node=END)
        )

        return workflow.compile()
    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    async def main():
        """Run the research assistant"""

        print("=" * 80)
        print("RESEARCH ASSISTANT WITH ERROR HANDLING")
        print("=" * 80)

        # Create the graph
        graph = create_research_graph()

        # Test queries
        queries = [
            "What are the latest developments in quantum computing?",
            "err",  # This will fail validation (too short)
            "Impact of artificial intelligence on healthcare",
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 80}")
            print(f"QUERY {i}: {query}")
            print(f"{'=' * 80}\n")

            initial_state = ResearchState(query=query)

            try:

                final_state = await graph.ainvoke(initial_state)

                # Display results
                print("\n" + "=" * 80)
                print("RESULTS")
                print("=" * 80)

                # final_state is a dict, not a ResearchState object
                error_messages = final_state.get("error_messages", [])

                if error_messages:
                    print(f"\n❌ FAILED with errors:")
                    for error in error_messages:
                        print(f"  - {error}")
                    print(f"\nRetry count: {final_state.get('retry_count', 0)}/{final_state.get('max_retries', 3)}")
                else:
                    print(f"\n✅ SUCCESS!")
                    research_plan = final_state.get('research_plan', 'N/A')
                    summary = final_state.get('summary', 'N/A')
                    final_report = final_state.get('final_report', 'N/A')

                    print(f"\nResearch Plan:\n{research_plan[:200] if research_plan != 'N/A' else research_plan}...")
                    print(f"\nSummary:\n{summary[:200] if summary != 'N/A' else summary}...")
                    print(f"\nFinal Report:\n{final_report[:300] if final_report != 'N/A' else final_report}...")

                # Show error summary (convert dict to object-like for ErrorHandler)
                print(f"\nError Summary:")
                print(f"  - Has errors: {len(error_messages) > 0}")
                print(f"  - Error count: {len(error_messages)}")
                print(f"  - Retry count: {final_state.get('retry_count', 0)}")
                print(f"  - Last failed node: {final_state.get('last_failed_node', 'None')}")



            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")

            await asyncio.sleep(1)


    if __name__ == "__main__":
        asyncio.run(main())
    ```
