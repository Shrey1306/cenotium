"""Shared ReAct agent workflow for plan-and-execute pattern."""

import asyncio
import logging
from typing import List

from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .callback_handler import LoggingCallbackHandler
from .models import Act, Plan, PlanExecute, Response

load_dotenv(override=True)

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps.
The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps."""

REPLANNER_TEMPLATE = """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer.
Do not add any superfluous steps.
The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed, respond with that.
Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done.
Do not return previously done steps as part of the plan."""


class ReactWorkflow:
    """ReAct workflow for plan-and-execute agents."""

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = "gpt-4o",
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.tools = tools
        self.model = model
        self.system_prompt = system_prompt
        self.replanning_attempts = 0

        callback_handler = LoggingCallbackHandler(logger)
        callback_manager = CallbackManager(handlers=[callback_handler])

        llm = ChatOpenAI(model=model)
        self.agent_executor = create_react_agent(
            llm, tools, state_modifier=system_prompt
        ).with_config(callback_manager=callback_manager, verbose=True)

        self.planner = self._create_planner()
        self.replanner = self._create_replanner()
        self.workflow = self._build_workflow()

    def _create_planner(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                ("placeholder", "{messages}"),
            ]
        )
        return planner_prompt | ChatOpenAI(
            model=self.model, temperature=0
        ).with_structured_output(Plan)

    def _create_replanner(self):
        replanner_prompt = ChatPromptTemplate.from_template(REPLANNER_TEMPLATE)
        return replanner_prompt | ChatOpenAI(
            model=self.model, temperature=0
        ).with_structured_output(Act)

    async def _execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"For the following plan:\n{plan_str}\n\nYou are tasked with executing step 1: {task}."
        agent_response = await self.agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    async def _plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    async def _replan_step(self, state: PlanExecute):
        output = await self.replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        return {"plan": output.action.steps}

    def _should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        self.replanning_attempts += 1
        return "agent"

    def _build_workflow(self):
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", self._plan_step)
        workflow.add_node("agent", self._execute_step)
        workflow.add_node("replan", self._replan_step)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges("replan", self._should_end, ["agent", END])

        return workflow.compile()

    async def arun(self, query: str, context: str = "") -> dict:
        """Run the workflow asynchronously."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        self.replanning_attempts = 1
        prompt = f"User Query: {query}\nUser Context: {context}" if context else query

        config = {"recursion_limit": 50}
        inputs = {"input": prompt}

        try:
            responses = []
            async for event in self.workflow.astream(inputs, config=config):
                for k, v in event.items():
                    if k != "__end__":
                        responses.append(v)

            return {
                "status": "success",
                "response": responses[-1].get("response") if responses else None,
                "attempts": self.replanning_attempts,
            }
        except GraphRecursionError:
            logger.error("Graph recursion error")
            return {
                "status": "error",
                "response": None,
                "attempts": 0,
            }

    def run(self, query: str, context: str = "") -> dict:
        """Run the workflow synchronously."""
        return asyncio.run(self.arun(query, context))


def create_react_workflow(
    tools: List[BaseTool],
    model: str = "gpt-4o",
    system_prompt: str = "You are a helpful assistant.",
) -> ReactWorkflow:
    """Factory function to create a ReAct workflow."""
    return ReactWorkflow(tools, model, system_prompt)


def run_agent(query: str, context: str, tools: List[BaseTool]) -> dict:
    """Run an agent with the given tools."""
    workflow = create_react_workflow(tools)
    return workflow.run(query, context)
