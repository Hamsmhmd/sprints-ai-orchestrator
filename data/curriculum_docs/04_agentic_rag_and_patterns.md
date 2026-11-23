# Module 4: Agentic RAG, Evaluator-Optimizer and Human-in-the-Loop

## Agentic RAG with self-reflection
In agentic RAG the agent decides whether retrieval was good enough. After retrieving, a reflection step asks the model whether the context is sufficient to answer the question. If it is not, the agent reformulates the query with more specific keywords and searches again. The loop must have a maximum number of attempts. If the context is still insufficient after the last attempt the agent must say that it is unable to verify the answer instead of guessing, because an invented answer is worse than no answer.

## Evaluator-Optimizer pattern
One node generates a result, a second node acts as an LLM judge and evaluates it against an explicit rubric. If the judge rejects the result, the generator refines it using the judge feedback. The loop stops when the judge approves or when the maximum number of refinement iterations is reached; the Sprints platform allows at most 3 refinements to control token consumption and latency.

## Human-in-the-loop with interrupt
The interrupt primitive pauses graph execution at a node and surfaces a payload to a human. Execution resumes when the application invokes the graph again with Command(resume=value) on the same thread_id. A checkpointer is required because the paused state must be stored. Because the node restarts from its beginning on resume, code before the interrupt must have no side effects. Use interrupts for high-stakes actions such as approving a job placement recommendation for a tier-1 hiring partner.

## Orchestrator-worker design
An orchestrator classifies the request and delegates to specialised worker agents. Workers communicate through structured messages, which keeps every step traceable.
