\# ADR-0003: Base model selection



\## Status

Accepted



\## Context

The project needs an audio classifier that runs in the browser in real time.

The model must be small enough to keep inference under 50ms and export

cleanly to ONNX.



\## Decision

Use a custom CNN trained from scratch on UrbanSound8K log-mel spectrograms.



\## Alternatives considered

\- PANNs (pretrained on AudioSet): rejected as primary path due to added

&#x20; complexity of managing external weights and larger model size. Kept as

&#x20; an optional ambition route if accuracy stays below 80%.



\## Consequences

\- Fully self-contained: no external weights to manage.

\- Clean ONNX export with no unsupported operators.

\- Target: >=80% accuracy on 10-fold CV.

