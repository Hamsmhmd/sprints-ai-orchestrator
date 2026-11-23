# Module 3: LangGraph State Machines and Memory

## StateGraph and TypedDict state
LangGraph models an application as a graph of nodes that read and update a shared state. The state is declared as a TypedDict. Each node returns a partial update, and reducers decide how updates merge: the add_messages reducer appends new messages to the conversation history instead of overwriting it.

## Conditional edges and routing
A conditional edge calls a routing function that inspects the state and returns the name of the next node. This is how an orchestrator sends a learner query to the Advisor Agent or to the Assessor Agent based on intent, and how loops are created: a node can route back to an earlier node until a condition is met.

## Checkpointers and threads
A checkpointer saves the graph state after every step. MemorySaver keeps checkpoints in process memory and is meant for development and demos; production systems use a database-backed checkpointer such as the SQLite or Postgres savers. Each conversation has a thread_id passed in the config under configurable. Different thread_ids are fully isolated, so one learner can never see another learner's history, and the same thread_id restores the earlier context in a later session, which prevents context amnesia.

## Memory strategies
Short-term memory is the message history stored in the thread. To control token cost, only the most recent messages are placed in the prompt while the full history stays in the checkpoint. Long-term facts such as the learner identifier are stored as explicit state fields.
