# Orchestration layer (Phase 3)

**These are optional integrations. The shipped pipeline does not use n8n or Lyzr.** `python code/main.py --run` never reads anything in this directory, and `output.csv` is the same whether or not these configs exist.

This directory renders LEITWERK's seven stages as a visible agentic DAG. It exists for the demo and for the "Orchestrate" framing.

## The rule that governs this directory

**Nothing here is on the path to `output.csv`.**

```bash
python code/main.py --run     # produces the full 110-row submission
```

That command works with this directory deleted, with n8n uninstalled, and with no network. No module under `code/` imports `serve.py`. If a webhook flakes at hour nine of the hackathon, the submission is unaffected. This is a deliberate architectural constraint, not an accident of how it turned out.

## What is here

| File | What it is |
|---|---|
| `n8n_leitwerk.json` | importable n8n workflow — the seven stages as a DAG with a branch on the safety constraint |
| `lyzr_agents.json` | agent definitions for the two stages that are genuinely agentic (S1 perception, S5 adjudication) |

The backend both of them call is `code/serve.py`, a stdlib-only HTTP server that exposes each stage function as an endpoint.

## Running it

```bash
# 1. start the stage server
python code/serve.py --port 8420

# 2. verify
curl http://127.0.0.1:8420/health

# 3. drive one message through all seven stages
curl -X POST http://127.0.0.1:8420/route -d '{"message_id":"sample_msg_053"}'
```

Then import `n8n_leitwerk.json` into n8n (Workflows → Import from File) and POST `{"message_id": "..."}` to its webhook.

## Why S5 is the node worth demoing

The adjudicator returns a two-sided argument rather than a score, which is what makes the DAG worth looking at. On the planted prompt-injection row:

```json
{
  "message_id": "sample_msg_053",
  "adjudicated_risk": 0.9961,
  "safety_forced": true,
  "prosecution": [
    { "claim": "attempts to instruct the router ('ignore all previous')", "weight": 3.4 },
    { "claim": "asks for a credential ('otp')",                          "weight": 2.6 },
    { "claim": "manufactures deadline pressure ('keep payments active')", "weight": 1.7 }
  ],
  "defence": [
    { "claim": "12 prior messages from this sender", "weight": -1.0 },
    { "claim": "user has opened this counterparty's messages before", "weight": -0.6 }
  ],
  "note": "risk 1.00 >= tau 0.72: safety constraint forces mute"
}
```

Note what the defence found: this sender is *not* a stranger — the user has twelve prior messages from them and has opened some. A relationship-only system would clear this message. The prosecution outweighs it because attempting to steer the router is itself scored as evidence of scam.

## Honest scope

- The adjudication is **deterministic**, not an LLM exchange. It runs with no API key, costs nothing, and reproduces exactly. `LEITWERK.md` §10 describes promoting it to a two-agent LLM exchange; the deterministic core is what shipped, and the ablation table shows the LLM version was not needed to get the traps right.
- `lyzr_agents.json` is a **configuration artifact**, not a deployed agent. Deploying it needs a Lyzr account and credentials that are not part of this submission.
- The n8n workflow was authored against the n8n 1.x node schema and is structurally valid for import. It has not been executed against a live n8n instance as part of this submission; `code/serve.py`, which it calls, has been smoke-tested end to end and its output is shown above.
