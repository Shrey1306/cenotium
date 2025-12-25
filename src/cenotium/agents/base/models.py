"""Shared Pydantic models for ReAct agents."""

import operator
from typing import Annotated, List, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    """State for plan-and-execute workflow."""

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future."""

    steps: List[str] = Field(description="Steps to follow, should be in sorted order")


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. Use Response to reply to user, Plan to continue."
    )
