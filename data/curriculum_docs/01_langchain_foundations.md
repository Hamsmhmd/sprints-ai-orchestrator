# Module 1: LangChain Foundations

## Prompt templates and chains
A ChatPromptTemplate turns variables into a list of messages. Chains are built with the LCEL pipe operator: prompt | model | parser. Each piece is a Runnable, so every step supports invoke, batch and stream. Linear chains are ideal for fixed pipelines, but they cannot branch or loop, which is why dynamic routing is done with LangGraph instead.

## Structured output with Pydantic
A PydanticOutputParser reads the format instructions of a Pydantic model and validates the model response against that schema. In the Sprints assessment workflow the Assessor feedback is a Pydantic model with three fields: score (0 to 100), feedback_points (a list of short strings) and pass_status. If the model returns invalid JSON the parser raises an OutputParserException, which the application can catch and retry.

## Message types
LangChain uses typed messages: HumanMessage for the learner, AIMessage for the model, SystemMessage for instructions and ToolMessage for the result of a tool call. An AIMessage can contain tool_calls; each ToolMessage references the tool_call_id it answers. Keeping this protocol consistent makes agent traces readable in verbose logs.

## Tools
A tool is a Python function decorated with @tool. The docstring becomes the tool description that the model reads when deciding whether to call it, so docstrings must be precise. Tools should return structured data and handle their own exceptions.
