"""Twilio calling tool for phone automation."""

import os

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from twilio.rest import Client


class TwilioCallParams(BaseModel):
    """Parameters for Twilio call."""

    to_number: str = Field(..., description="Phone number in E.164 format")
    message: str = Field(..., description="Message to speak on the call")


def make_twilio_call(to_number: str, message: str) -> str:
    """Place a call using Twilio."""
    account_sid = os.getenv("ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_KEY")

    if not account_sid or not auth_token:
        return "Error: ACCOUNT_SID or TWILIO_KEY not set"

    client = Client(account_sid, auth_token)
    from_number = os.getenv("TWILIO_FROM_NUMBER", "+18778515935")

    call = client.calls.create(
        twiml=f"<Response><Say>{message}</Say></Response>",
        to=to_number,
        from_=from_number,
    )
    return f"Call placed to {to_number} with SID: {call.sid}"


twilio_tool = StructuredTool.from_function(
    func=make_twilio_call,
    name="make_twilio_call",
    description="Place a call using Twilio. Provide phone number in E.164 format and message.",
    args_schema=TwilioCallParams,
)
