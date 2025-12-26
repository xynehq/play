import json
import asyncio
import os
import yaml
from openai import AsyncOpenAI

# ==============================
# 🔧 CONFIGURATION via YAML File
# ==============================

def load_config_from_yaml():
    """Load configuration from model_config.yaml file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "model_config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error: Invalid YAML in config file: {e}")
        return None
    
    # Extract configuration from YAML
    model = config_data.get('model_name', '')
    base_url = config_data.get('api_base', '')
    api_key = config_data.get('api_key', None)
    
    # Get model parameters with defaults
    model_params = config_data.get('model_parameters', {})
    temperature = model_params.get('temperature', 0.6)
    top_p = model_params.get('top_p', 0.95)
    max_tokens = model_params.get('max_tokens', 10000)
    top_k = model_params.get('top_k', 25)  # Note: top_k is not used in OpenAI API but loaded for completeness
    
    # Get benchmark settings
    benchmark_settings = config_data.get('benchmark', {})
    max_concurrent_requests = benchmark_settings.get('max_concurrent_requests', 1)
    
    # Hardcoded input file path as requested
    input_file = "/home/om_user/repos/vm-benchmarks/hyperswitch-benchmark/aggregate_data.json"
    
    # Auto-generate output file path based on model name
    output_dir = os.path.join(script_dir, "..", "model_output")
    safe_model_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model)
    output_file = os.path.join(output_dir, f"{safe_model_name}_output.json")
    
    # Auto-generate prompts directory
    prompts_dir = os.path.join(script_dir, "..", "model_prompts")
    
    # Default number of outputs per task
    num_outputs = 3
    
    return {
        'model': model,
        'base_url': base_url,
        'input': input_file,
        'output': output_file,
        'prompts_dir': prompts_dir,
        'api_key': api_key,
        'num_outputs': num_outputs,
        'temperature': temperature,
        'top_p': top_p,
        'max_tokens': max_tokens,
        'top_k': top_k,
        'max_concurrent_requests': max_concurrent_requests
    }


def load_existing_results(output_file, model_name):
    """Load existing results if the output file exists, create it if it doesn't."""
    import os
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created directory: {output_dir}")
    
    # Try to load existing results
    try:
        with open(output_file, "r") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, dict) and "model_name" in existing_data and "results" in existing_data:
                return existing_data["results"]
            elif isinstance(existing_data, list):
                # Convert old format to new format
                return existing_data
            return []
    except FileNotFoundError:
        # Create empty output file if it doesn't exist
        try:
            output_structure = {
                "model_name": model_name,
                "results": []
            }
            with open(output_file, "w") as f:
                json.dump(output_structure, f, indent=2)
            print(f"📄 Created new output file: {output_file}")
            return []
        except Exception as e:
            print(f"⚠️ Error creating output file: {e}")
            return []
    except json.JSONDecodeError:
        print(f"⚠️ Corrupted JSON file, starting fresh: {output_file}")
        return []


async def save_result_immediately(result_item, results_list, output_file, model_name, lock):
    """Save individual result immediately to file."""
    async with lock:
        results_list.append(result_item)
        
        # Ensure output directory exists
        import os
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Create output structure with model name
        output_structure = {
            "model_name": model_name,
            "results": results_list
        }
        
        # Write updated results to file immediately
        try:
            with open(output_file, "w") as f:
                json.dump(output_structure, f, indent=2)
            print(f"  💾 Saved result for ID: {result_item['id']} | Total results: {len(results_list)}")
        except Exception as e:
            print(f"  ⚠️ Error saving result: {e}")


def construct_prompt(item):
    """Construct a comprehensive prompt from item fields based on task type.
    
    Now includes: task_type, error_handling_pattern
    Uses dynamic language tag instead of hardcoded 'rust'
    """
    
    task_type = item.get('task_type', '').lower()
    prompt_parts = []
    
    # Determine what fields to include based on task type
    is_debugging = 'debug' in task_type
    is_generation = 'generation' in task_type or 'gen' in task_type
    
    # Add task type
    if item.get('task_type'):
        prompt_parts.extend([
            f"**Task Type:** {item['task_type']}",
            ""
        ])
    
    # Add context code (common for all tasks) with dynamic language tag
    if item.get('context_code'):
        language = item.get('language', '')
        prompt_parts.extend([
            "**Context Code:**",
            f"```{language}",
            item['context_code'],
            "```",
            ""
        ])
    
    # Add buggy code (typically for debugging tasks)
    if item.get('buggy_code'):
        language = item.get('language', '')
        prompt_parts.extend([
            "**Buggy Code:**",
            f"```{language}",
            item['buggy_code'],
            "```",
            ""
        ])
    
    # Add error_handling_pattern if present
    if item.get('error_handling_pattern'):
        prompt_parts.extend([
            f"**Error Handling Pattern:** {item['error_handling_pattern']}",
            ""
        ])
    
    # Add main prompt (required for all tasks)
    if item.get('prompt'):
        prompt_parts.extend([
            f"**Task:**",
            item['prompt'],
            ""
        ])
    
    # Add constraints only for debugging or generation tasks
    if (is_debugging or is_generation) and item.get('constraints'):
        prompt_parts.extend([
            f"**Constraints:**",
            item['constraints'],
            ""
        ])
    
    return "\n".join(prompt_parts)


def determine_system_message(task_type):
    """Determine appropriate system instruction based on task type.
    
    Note: This will be prepended to the user prompt, not sent as a separate system message.
    """
    
    task_type_lower = task_type.lower() if task_type else ""
    
    if "debug" in task_type_lower:
        return "You are an expert code analyst and debugger. You provide detailed technical analysis, identify bugs accurately, and suggest robust solutions with proper error handling patterns."
    elif "generation" in task_type_lower or "gen" in task_type_lower:
        return "You are an expert software developer specialized in code generation. You write clean, efficient, and well-documented code following best practices."
    elif "understanding" in task_type_lower or "explanation" in task_type_lower:
        return "You are an expert code analyst specialized in code comprehension and explanation. You provide clear, detailed explanations of code functionality and design patterns."
    else:
        return "You are an expert AI model specialized in code-related tasks including generation, analysis, and debugging."


def save_prompt_to_file(item_id, full_prompt, prompts_dir):
    """Save the full prompt to a separate file in model_prompts directory."""
    try:
        os.makedirs(prompts_dir, exist_ok=True)
        prompt_file = os.path.join(prompts_dir, f"{item_id}_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(full_prompt)
    except Exception as e:
        print(f"    ⚠️ Warning: Could not save prompt to file: {e}")


async def generate_response(client, config, system_instruction, user_prompt):
    """Generate response from the model (async).
    
    Args:
        system_instruction: System-level instruction to prepend to user prompt
        user_prompt: The actual user prompt/task
    """
    # Combine system instruction with user prompt
    combined_prompt = f"{system_instruction}\n\n{user_prompt}"
    
    # Estimate input token count (rough approximation: 1 token ≈ 4 chars)
    estimated_input_tokens = len(combined_prompt) // 4
    estimated_total_tokens = estimated_input_tokens + config['max_tokens']
    
    # Log warning if estimated total exceeds common context limits
    if estimated_total_tokens > 30000:
        print(f"  ⚠️  Warning: Estimated total tokens ({estimated_total_tokens}) may exceed context window. "
              f"Input: ~{estimated_input_tokens}, Output: {config['max_tokens']}")
    
    try:
        response = await client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "user", "content": combined_prompt}
            ],
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
        )

        # Extract response
        generated_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        
        # Check if response was truncated
        if finish_reason == "length":
            print(f"  ⚠️  Warning: Response truncated due to max_tokens limit ({config['max_tokens']})")
        
        if generated_text:
            generated_text = generated_text.strip()
        else:
            generated_text = ""

        return generated_text

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return f"Error during generation: {str(e)}"


async def generate_multiple_responses(client, config, system_instruction, user_prompt, num_responses):
    """Generate multiple responses from the model for the same prompt.
    
    Args:
        client: AsyncOpenAI client
        config: Configuration dictionary
        system_instruction: System-level instruction
        user_prompt: The actual user prompt/task
        num_responses: Number of responses to generate
    
    Returns:
        dict: Dictionary with keys 'output1', 'output2', 'output3', etc.
    """
    tasks = []
    for i in range(num_responses):
        # print(f"    🔄 Generating response {i+1}/{num_responses}...")
        tasks.append(generate_response(client, config, system_instruction, user_prompt))
    
    results = await asyncio.gather(*tasks)
    
    responses = {}
    for i, response in enumerate(results):
        responses[f"output{i+1}"] = response
        
    return responses


async def main():
    # Load configuration from YAML file
    config = load_config_from_yaml()
    
    if config is None:
        print("❌ Failed to load configuration. Exiting.")
        return
    
    print("🚀 Universal Code Generation Inference - Single Model (Multiple Outputs)")
    print("="*60)
    print("This script will run inference on a single model")
    print("Each input will generate multiple outputs for comparison")
    print("Configuration loaded from model_config.yaml\n")
    
    print("\n" + "="*60)
    print("📋 Configuration Summary")
    print("="*60)
    print(f"📝 Model: {config['model']}")
    print(f"🌐 Base URL: {config['base_url']}")
    print(f"🔑 API Key: {'Provided' if config['api_key'] else 'Not provided (optional)'}")
    print(f"📂 Input: {config['input']}")
    print(f"📁 Output: {config['output']}")
    print(f"📂 Prompts Dir: {config['prompts_dir']}")
    print(f"🔢 Outputs per task: {config['num_outputs']}")
    print(f"🌡️  Temperature: {config['temperature']}")
    print(f"🎯 Max Tokens: {config['max_tokens']}")
    print(f"🔝 Top-p: {config['top_p']}")
    print(f"⚡ Max Concurrent Requests: {config['max_concurrent_requests']}")
    print("="*60)
    
    print("\n✅ Starting inference with loaded configuration...")
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(
        api_key=config['api_key'] if config['api_key'] else "dummy-key",
        base_url=config['base_url']
    )

    # Load input JSON
    try:
        with open(config['input'], "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {config['input']}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}")
        return

    if isinstance(data, dict):
        data = [data]

    # Load any existing results to avoid re-processing
    results = load_existing_results(config['output'], config['model'])
    
    # Filter out results that have errors so they can be re-processed
    valid_results = []
    processed_ids = set()
    
    for result in results:
        has_error = False
        # Check all keys that look like outputX
        for key, value in result.items():
            if key.startswith("output") and isinstance(value, str) and "Error during generation" in value:
                has_error = True
                break
        
        if not has_error:
            valid_results.append(result)
            processed_ids.add(result["id"])
        else:
            print(f"  ⚠️  Found error in existing result for ID {result.get('id')}, will re-process.")
            
    # Update results list to only include valid ones
    results = valid_results
    
    print(f"\n📊 Found {len(results)} valid existing results.")
    print(f"🎯 Processing {len(data)} tasks...")

    # Concurrency control
    semaphore = asyncio.Semaphore(config['max_concurrent_requests'])
    save_lock = asyncio.Lock()

    async def process_item(item):
        item_id = item.get('id')
        task_type = item.get('task_type', 'unknown')
        
        async with semaphore:
            print(f"\n🧠 Processing ID: {item_id}")
            print(f"  📝 Task Type: {task_type}")
            
            # Construct prompt based on task type
            full_prompt = construct_prompt(item)
            
            # Save prompt to file
            save_prompt_to_file(item_id, full_prompt, config['prompts_dir'])
            
            # Determine appropriate system message
            system_message = determine_system_message(task_type)

            # Generate multiple responses (N outputs for same input)
            num_outputs = config['num_outputs']
            print(f"  🎯 Generating {num_outputs} response(s) for ID: {item_id}")
            multiple_outputs = await generate_multiple_responses(client, config, system_message, full_prompt, num_outputs)
            print(f"  ✅ All {num_outputs} generation(s) complete for ID: {item_id}")

            # Build result entry - preserve all input fields
            result_item = {
                "id": item["id"],
                "file_path": item.get("file_path", ""),
                "task_type": item.get("task_type", ""),
            }
            
            # Add context_code if present
            if "context_code" in item:
                result_item["context_code"] = item["context_code"]
            
            # Add buggy_code if present
            if "buggy_code" in item:
                result_item["buggy_code"] = item["buggy_code"]
            
            # Add prompt if present
            if "prompt" in item:
                result_item["prompt"] = item["prompt"]
            
            # Add constraints if present
            if "constraints" in item:
                result_item["constraints"] = item["constraints"]
            
            # Add error_handling_pattern if present
            if "error_handling_pattern" in item:
                result_item["error_handling_pattern"] = item["error_handling_pattern"]
            
            # Add expected_output if present
            if "expected_output" in item:
                result_item["expected_output"] = item["expected_output"]
            
            # Add metadata if present
            if "metadata" in item:
                result_item["metadata"] = item["metadata"]
            
            # Add multiple model outputs
            result_item.update(multiple_outputs)

            # Save immediately after each generation
            await save_result_immediately(result_item, results, config['output'], config['model'], save_lock)

    tasks = []
    for i, item in enumerate(data):
        item_id = item.get('id')
        
        # Skip if already processed
        if item_id in processed_ids:
            print(f"  ⏭️  Skipping ID {item_id} - already processed")
            continue

        tasks.append(process_item(item))

    if tasks:
        await asyncio.gather(*tasks)
    else:
        print("No new tasks to process.")

    # Generate summary
    print(f"\n{'='*60}")
    print(f"📊 GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total Items Generated: {len(results)}")
    print(f"📁 Output saved to: {config['output']}")
    
    # Task type breakdown
    task_types = {}
    for result in results:
        task_type = result.get('task_type', 'unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    if len(task_types) > 1:
        print(f"\n📋 Breakdown by Task Type:")
        for task_type, count in task_types.items():
            print(f"   {task_type}: {count} items")
    
    print(f"\n✅ Generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
