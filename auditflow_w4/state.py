"""State for the autonomous coding agent feedback loop."""

import operator
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage


class CodingState(TypedDict):
    """
    State passed between all agents across iterations.

    Fields:
      task             : the programming task description
      current_code     : code produced by Coder in current iteration
      test_result      : output from Test Runner
      critic_diagnosis : diagnosis produced by Critic
      iteration        : current loop count
      max_iterations   : hard stop limit
      status           : RUNNING | PASSED | FAILED_MAX_ITER
      inject_failure   : if True, Test Runner returns misleading error
      messages         : full conversation history
    """
    messages:        Annotated[list[BaseMessage], operator.add]
    task:            str
    current_code:    str
    test_result:     str
    critic_diagnosis:str
    iteration:       int
    max_iterations:  int
    status:          str
    inject_failure:  bool
