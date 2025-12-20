# τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains

To manually run the benchmark, use the following command. Replace `url`, `optional_api_key`, and `model_name` with your actual values. Adjust the concurrency level based on your available resources.

```
source venv/bin/activate && \
OPENAI_API_BASE="url" \
OPENAI_API_KEY="optional_api_key" \
python3 run.py \
  --env airline \
  --model openai/model_name \
  --model-provider openai \
  --user-model openai/model_name \
  --user-model-provider openai \
  --agent-strategy tool-calling \
  --max-concurrency 3 #
```
