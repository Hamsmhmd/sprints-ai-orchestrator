# Sprints Advanced Agentic AI: FAQ and Troubleshooting

## How many refinement iterations does the Assessor allow?
The evaluator-optimizer loop allows at most 3 refinement iterations. After the third refinement the judge result is final: if the judge still rejects the assessment the submission is flagged for a human grader. This limit exists to control token consumption and latency.

## What is the pass mark for Python assignments?
The pass mark is 60 out of 100. A runtime error or a timeout can never pass. Security violations score zero.

## Why was my code blocked by the sandbox?
The sandbox only allows a safe list of standard library modules such as math, random, statistics, itertools, collections, re, json and datetime. Importing os, sys, subprocess or socket, opening files, calling eval or exec, and reading private dunder attributes are blocked, because learner code must never breach platform security.

## What happens when a tool fails?
If the learner database, the vector store or the code sandbox raises an exception, the agent returns a structured message that starts with Unable to verify. It never improvises an answer, because invented qualifications or a wrong employer match would damage the trust of hiring partners.

## How are job recommendations produced?
Recommendations come only from database records. A learner must have the readiness status Ready for Placement. The employer must be one of the learner's preferred employers and the opening must be in the learner's bootcamp track. The salary band of the opening is compared with the learner's EGP target salary.

## Which recommendations need human approval?
Any recommendation for a tier-1 partner, for example Fawry, pauses the graph with an interrupt. A human reviewer approves or rejects it and the graph resumes on the same thread_id. Recommendations for other partners are released without approval.

## Will the assistant remember my previous questions?
Yes, within the same thread_id. The checkpointer stores the message history and the learner identifier, so a follow-up such as Which partner opening fits me? works without repeating your learner id. A different learner uses a different thread_id and can never see this history.

## Why did my curriculum question return Unable to verify?
The agent searched up to three times, reformulating the query each time, and the retrieved curriculum text still did not contain the facts needed. Rephrase the question with module vocabulary such as checkpointer, conditional edge, interrupt, chunking or embeddings, or ask a human advisor.

## How fast is retrieval?
ChromaDB retrieval must return within 1000 milliseconds. The retriever records the latency of every search and the benchmark command reports the median and the 95th percentile.

## What is the difference between a chain and a graph?
A chain is a fixed linear sequence of steps. A graph has conditional edges and cycles, so it can route by intent, loop until a condition is met, pause for a human and resume later. Sprints uses LangGraph for all orchestration.
