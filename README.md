# AuditFlow

AuditFlow is a lightweight middleware for auditing agentic workflows. It captures execution events during runtime, reconstructs the workflow as a causal Directed Acyclic Graph (DAG), and enables post-hoc analysis of agent behavior and failures.

The goal of AuditFlow is to improve the observability and auditability of AI agent systems by recording structured execution traces without modifying the underlying workflow logic.

## Features

- Lightweight middleware for agentic workflows
- Automatic runtime event capture
- Causal DAG reconstruction
- Structured audit logs
- Cross-run execution comparison
- Tamper-evident execution records using cryptographic hashes
- Replay and verification of recorded executions


## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/AuditFlow.git
cd AuditFlow
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running AuditFlow

Run any of the example workflows:

```bash
python workflows/<workflow_name>.py
```

Audit logs will be generated automatically during execution.

## Example Workflows

The repository includes implementations of the workflows used in our evaluation:

- Single-agent workflow
- Multi-agent workflow
- Tool-augmented workflow
- Feedback-driven workflow

## Output

AuditFlow records structured execution events and reconstructs an execution DAG that can be used for:

- execution tracing
- failure attribution
- workflow replay
- execution comparison across runs
- integrity verification



## License

This project is released under the MIT License.
