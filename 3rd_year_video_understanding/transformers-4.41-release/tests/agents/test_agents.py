# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import unittest
import uuid

import pytest

from transformers.agents.agent_types import AgentText
from transformers.agents.agents import AgentMaxIterationsError, CodeAgent, ReactCodeAgent, ReactJsonAgent, Toolbox
from transformers.agents.default_tools import PythonInterpreterTool
from transformers.testing_utils import require_torch


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


def fake_react_json_llm(messages, stop_sequences=None) -> str:
    prompt = str(messages)

    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Action:
{
    "action": "python_interpreter",
    "action_input": {"code": "2*3.6452"}
}
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""


def fake_react_code_llm(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
print(result)
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
"""


def fake_code_llm_oneshot(messages, stop_sequences=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
"""


class AgentTests(unittest.TestCase):
    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_code_llm_oneshot)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert output == "7.2904"

    def test_fake_react_json_agent(self):
        agent = ReactJsonAgent(tools=[PythonInterpreterTool()], llm_engine=fake_react_json_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert output == "7.2904"
        assert agent.logs[0]["task"] == "What is 2 multiplied by 3.6452?"
        assert agent.logs[1]["observation"] == "7.2904"
        assert agent.logs[1]["rationale"].strip() == "Thought: I should multiply 2 by 3.6452. special_marker"
        assert (
            agent.logs[2]["llm_output"]
            == """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""
        )

    def test_fake_react_code_agent(self):
        agent = ReactCodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_react_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "7.2904"
        assert agent.logs[0]["task"] == "What is 2 multiplied by 3.6452?"
        assert float(agent.logs[1]["observation"].strip()) - 12.511648 < 1e-6
        assert agent.logs[2]["tool_call"] == {
            "tool_arguments": "final_answer(7.2904)",
            "tool_name": "code interpreter",
        }

    def test_setup_agent_with_empty_toolbox(self):
        ReactJsonAgent(llm_engine=fake_react_json_llm, tools=[])

    def test_react_fails_max_iterations(self):
        agent = ReactCodeAgent(
            tools=[PythonInterpreterTool()],
            llm_engine=fake_code_llm_oneshot,  # use this callable because it never ends
            max_iterations=5,
        )
        agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.logs) == 7
        assert type(agent.logs[-1]["error"]) == AgentMaxIterationsError

    @require_torch
    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = ReactCodeAgent(tools=toolset_1, llm_engine=fake_react_code_llm)
        assert len(agent.toolbox.tools) == 1  # contains only final_answer tool

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        agent = ReactCodeAgent(tools=toolset_2, llm_engine=fake_react_code_llm)
        assert len(agent.toolbox.tools) == 2  # added final_answer tool

        toolset_3 = Toolbox(toolset_2)
        agent = ReactCodeAgent(tools=toolset_3, llm_engine=fake_react_code_llm)
        assert len(agent.toolbox.tools) == 2  # added final_answer tool

        # check that add_base_tools will not interfere with existing tools
        with pytest.raises(KeyError) as e:
            agent = ReactJsonAgent(tools=toolset_3, llm_engine=fake_react_json_llm, add_base_tools=True)
        assert "python_interpreter already exists in the toolbox" in str(e)

        # check that python_interpreter base tool does not get added to code agents
        agent = ReactCodeAgent(tools=[], llm_engine=fake_react_code_llm, add_base_tools=True)
        assert len(agent.toolbox.tools) == 6  # added final_answer tool + 5 base tools (excluding interpreter)
