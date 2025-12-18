# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains

## Running τ-bench

To manually run the benchmark, use the following command. Replace `url`, `optional_api_key`, and `model_name` with your actual values. Adjust the concurrency level based on your available resources.

```
source venv/bin/activate && \
OPENAI_API_BASE="url" \
OPENAI_API_KEY="optional_api_key" \
tau2 run \
  --domain retail \
  --agent-llm openai/model_name \
  --user-llm openai/model_name \
  --max-concurrency 3
```
