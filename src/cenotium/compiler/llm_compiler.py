"""Main LLMCompiler implementation for task planning and execution."""

import asyncio
import itertools
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from .executor import ExecutorPool
from .output_parser import LLMCompilerPlanParser
from .task_fetching import schedule_tasks

load_dotenv()


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class FinalPlan(BaseModel):
    plan: str = Field(description="The final trip plan")


class JoinOutputs(BaseModel):
    thought: str = Field(description="Reasoning behind the selected action")
    should_replan: bool = Field(
        description="Whether to replan or provide final response"
    )
    feedback: Optional[str] = Field(None, description="Feedback for replanning")
    plan: Optional[str] = Field(None, description="Final plan if not replanning")


def _create_trip_planner_tool() -> StructuredTool:
    def plan_trip(info: Dict) -> str:
        trip_info = {
            "destination": info.get("destination", ""),
            "budget_per_person": float(info.get("budget_per_person", 0)),
            "group_size": int(info.get("group_size", 0)),
            "duration_nights": int(info.get("duration_nights", 0)),
            "duration_days": int(info.get("duration_days", 0)),
            "activities": info.get("activities", []),
        }
        total_budget = trip_info["budget_per_person"] * trip_info["group_size"]
        return f"""
Trip Plan for {trip_info["destination"]}:

Group Details:
- Group Size: {trip_info["group_size"]} people
- Duration: {trip_info["duration_nights"]} nights, {trip_info["duration_days"]} days
- Budget: ${trip_info["budget_per_person"]:,.2f} per person (Total: ${total_budget:,.2f})

Budget Breakdown (per person):
- Flights & Transportation: ${trip_info["budget_per_person"] * 0.35:,.2f}
- Accommodation: ${trip_info["budget_per_person"] * 0.30:,.2f}
- Activities & Entertainment: ${trip_info["budget_per_person"] * 0.20:,.2f}
- Food & Dining: ${trip_info["budget_per_person"] * 0.15:,.2f}
"""

    return StructuredTool.from_function(
        name="trip_planner",
        func=plan_trip,
        description="Plans a trip given destination, budget_per_person, group_size, etc.",
    )


class LLMCompiler:
    """LLM Compiler for task planning and parallel execution."""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        model: str = "gpt-4-turbo-preview",
    ):
        self.tools = tools or [_create_trip_planner_tool()]
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.executor_pool = ExecutorPool()
        self._setup_components()

    def _setup_components(self):
        self.planner_prompt = hub.pull("wfh/llm-compiler")
        self.joiner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a planning assistant that analyzes execution results and decides whether to:
1. Replan with feedback if the plan needs improvement
2. Provide a final response if the plan is complete

Analyze the plan and provide:
1. Your thought process
2. Decision to replan (true/false)
3. If replanning, feedback
4. If not, final plan""",
                ),
                ("user", "{input}"),
            ]
        )
        self.parser = LLMCompilerPlanParser(tools=self.tools)
        tool_descriptions = "\n".join(
            f"{i + 1}. {tool.description}\n" for i, tool in enumerate(self.tools)
        )
        self.planner_prompt = self.planner_prompt.partial(
            replan="",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )

    def _parse_joiner_output(self, decision: JoinOutputs) -> Dict:
        response = [AIMessage(content=f"Thought: {decision.thought}")]
        if decision.should_replan:
            return {
                "messages": response
                + [
                    SystemMessage(
                        content=f"Context from last attempt: {decision.feedback}"
                    )
                ]
            }
        return {
            "messages": response
            + [AIMessage(content=decision.plan or "No plan provided")]
        }

    def _select_recent_messages(self, state: Dict) -> Dict:
        messages = state["messages"]
        selected = []
        for msg in reversed(messages):
            selected.append(msg)
            if isinstance(msg, HumanMessage):
                break
        return {"input": str(selected[-1].content)}

    def plan_and_schedule(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        tasks_iter = self.parser.stream(messages)
        try:
            first_task = next(tasks_iter)
            tasks_iter = itertools.chain([first_task], tasks_iter)
        except StopIteration:
            tasks_iter = iter([])
        return {
            "messages": schedule_tasks(
                {"messages": messages, "tasks": tasks_iter},
                config=RunnableConfig(recursion_limit=100),
            )
        }

    def create_graph(self) -> StateGraph:
        joiner = (
            RunnableLambda(self._select_recent_messages)
            | self.joiner_prompt
            | self.llm.with_structured_output(JoinOutputs, method="function_calling")
            | RunnableLambda(self._parse_joiner_output)
        )
        graph = StateGraph(State)
        graph.add_node("plan_and_schedule", RunnableLambda(self.plan_and_schedule))
        graph.add_node("join", joiner)
        graph.add_edge("plan_and_schedule", "join")

        def should_continue(state):
            messages = state["messages"]
            if isinstance(messages[-1], AIMessage):
                return END
            return "plan_and_schedule"

        graph.add_conditional_edges("join", should_continue)
        graph.add_edge(START, "plan_and_schedule")
        return graph.compile()

    async def astream(
        self, query: str, config: Optional[Dict] = None, timeout: int = 120
    ):
        config = config or {"recursion_limit": 100}
        chain = self.create_graph()

        async def async_gen_wrapper():
            for item in chain.stream(
                {"messages": [HumanMessage(content=query)]}, config
            ):
                yield item

        agen = async_gen_wrapper()
        while True:
            try:
                step = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                yield step
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                yield {"error": "Timeout reached"}
                break

    def run(self, query: str) -> str:
        chain = self.create_graph()
        result = chain.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 100},
        )
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return ""
