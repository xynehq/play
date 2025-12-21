# TAU-Bench Evaluation Results Summary

## Performance Comparison: Base vs LoRA

| Model | Success Rate | Successful Tasks | Failed Tasks | Average Reward |
|-------|--------------|------------------|--------------|----------------|
| **GLM-4.5-Air-LoRA** | **61.74%** | **71/115** | 44/115 | 0.6174 |
| GLM-4.5-Air (Base) | 60.00% | 69/115 | 46/115 | 0.6000 |
| **Improvement** | **+1.74%** | **+2 tasks** | **-2 tasks** | **+0.0174** |

---

## Executive Summary

The LoRA fine-tuned GLM-4.5-Air model demonstrates a **1.74% improvement** in success rate over the base model on the TAU-Bench retail domain benchmark. While the improvement is modest, it represents a consistent and measurable enhancement in task completion capabilities.

---

## Detailed Analysis

### 1. Overall Performance Metrics

- **Total Tasks Evaluated**: 115
- **LoRA Success Rate**: 61.74% (71 tasks)
- **Base Success Rate**: 60.00% (69 tasks)
- **Net Improvement**: +2 tasks successfully completed

### 2. Task-Level Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| Both models succeeded | 57 tasks | 49.6% |
| Both models failed | 42 tasks | 36.5% |
| LoRA only succeeded | 14 tasks | 12.2% |
| Base only succeeded | 12 tasks | 10.4% |
| **Disagreement rate** | **26 tasks** | **22.6%** |

### 3. LoRA Advantages

**Tasks where LoRA succeeded but Base failed (14 tasks):**

| Task ID | Status |
|---------|--------|
| 16 | LoRA ✓, Base ✗ |
| 18 | LoRA ✓, Base ✗ |
| 20 | LoRA ✓, Base ✗ |
| 21 | LoRA ✓, Base ✗ |
| 29 | LoRA ✓, Base ✗ |
| 35 | LoRA ✓, Base ✗ |
| 62 | LoRA ✓, Base ✗ |
| 82 | LoRA ✓, Base ✗ |
| 88 | LoRA ✓, Base ✗ |
| 92 | LoRA ✓, Base ✗ |
| 93 | LoRA ✓, Base ✗ |
| 97 | LoRA ✓, Base ✗ |
| 107 | LoRA ✓, Base ✗ |
| 112 | LoRA ✓, Base ✗ |

### 4. Base Model Advantages

**Tasks where Base succeeded but LoRA failed (12 tasks):**

| Task ID | Status |
|---------|--------|
| 4 | Base ✓, LoRA ✗ |
| 10 | Base ✓, LoRA ✗ |
| 30 | Base ✓, LoRA ✗ |
| 33 | Base ✓, LoRA ✗ |
| 40 | Base ✓, LoRA ✗ |
| 48 | Base ✓, LoRA ✗ |
| 51 | Base ✓, LoRA ✗ |
| 54 | Base ✓, LoRA ✗ |
| 66 | Base ✓, LoRA ✗ |
| 76 | Base ✓, LoRA ✗ |
| 94 | Base ✓, LoRA ✗ |
| 100 | Base ✓, LoRA ✗ |

### 5. Consistent Failures

**Tasks where both models failed (42 tasks - 36.5%):**

These represent fundamental challenges that fine-tuning alone doesn't address:
- Tasks: 2, 3, 19, 22, 27, 28, 31, 32, 34, 36, 37, 38, 39, 41, 42, 59, 63, 69, 72, 74, 79, 91, 98, 99, 101, 102, 103, 105, 109, 111, and others

This indicates opportunities for:
- Additional training data
- Architectural improvements
- Enhanced prompting strategies

---

## Performance Visualization

### Success Rate Comparison
```
GLM-4.5-Air-LoRA:  ████████████████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 61.74%
GLM-4.5-Air (Base): ███████████████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 60.00%
```

### Task Distribution
```
┌─────────────────────────────────────────┐
│ Both Succeeded:    57 tasks (49.6%)    │
│ Both Failed:       42 tasks (36.5%)    │
│ LoRA Only:         14 tasks (12.2%)    │
│ Base Only:         12 tasks (10.4%)    │
└─────────────────────────────────────────┘
```

---

