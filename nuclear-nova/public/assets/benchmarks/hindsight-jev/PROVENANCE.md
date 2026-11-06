# Decision boundary lab

Source: https://github.com/binzhango/hindsight-memory-agents/tree/38b5f58d7e9af83cd313358f0adb934b78cf217f

Run: `.venv/bin/python labs/decision_lab.py --output benchmarks/decision-lab/results.json`

Nine scripted cases exercise the real MemoryAgent wrapper. Memory, worker, and reviewer are test doubles. No network calls, live model quality, or latency measurements. The zero elapsed times in decision objects are placeholders, not measured service times.

The JSON includes synthetic records and complete prompts only. It has no credentials or user data.
