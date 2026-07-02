"""Shared state for the code review pipeline."""

from typing import TypedDict, Optional
import operator
from typing import Annotated
from langchain_core.messages import BaseMessage


class WorkflowState(TypedDict):
    """
    State passed between all agents.

    Fields populated progressively as agents run:
      - code_snippet     : set by user at start
      - analyzer_output  : set by static_analyzer_agent
      - reviewer_output  : set by code_reviewer_agent
      - patch_output     : set by patch_generator_agent
      - final_report     : set by orchestrator at END
      - next_agent       : routing decision set by orchestrator
      - inject_failure   : flag to trigger biased analyzer (F6)
    """
    messages:        Annotated[list[BaseMessage], operator.add]
    code_snippet:    str
    analyzer_output: str
    reviewer_output: str
    patch_output:    str
    final_report:    str
    next_agent:      str
    inject_failure:  bool
