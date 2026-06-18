\# ADR-0001: Monorepo with separation of training and inference planes



\## Status

Accepted



\## Context

The project has a Python plane (training) and a TypeScript plane

(browser inference) that only share one artifact: the ONNX model.



\## Decision

Use a single repository (monorepo) with independent `training/` and `web/`

modules. The model is the only interface between them.



\## Alternatives considered

\- Two separate repositories: rejected because it fragments the history

&#x20; and the review process.



\## Consequences

\- A single commit history and a single entry point for any reviewer.

\- CI split by paths so not everything runs on every change.

