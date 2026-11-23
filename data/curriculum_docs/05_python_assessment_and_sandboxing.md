# Module 5: Python Assignment Assessment and Safe Execution

## Grading rubric for Python assignments
Sprints grades Python assignments out of 100. A submission passes with a score of 60 or more. The rubric has four criteria: the code runs without runtime errors, the output matches the expected output, the code is readable with meaningful names, and functions include docstrings. A runtime error caps the score below the pass mark. A submission that violates the security policy receives zero.

## Executing learner code safely
Never execute learner code inside the platform process. Run it in a separate interpreter process with a wall-clock timeout, restricted builtins, an allow-list of importable modules and no network or file access. Capture stdout and the error message to give feedback. Infinite loops are stopped by the timeout and reported as timeouts.

## Feedback quality
Good feedback is specific and actionable: it quotes the error, names the line or concept involved and suggests the next step. Feedback must be consistent with the score and the pass status. The judge node checks this consistency before the learner sees the result.

## Placement readiness
A learner is ready for placement when the readiness status in the learner database is Ready for Placement. Recommendations are made only from live database records of completed modules, EGP target salary, preferred employers and open positions. Agents must never invent qualifications or partner openings.
