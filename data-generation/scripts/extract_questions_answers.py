import json
import argparse
import os

def extract_questions_answers(input_file, output_file):
    """
    Extracts questions and answers from a JSONL file and saves them to another JSONL file.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSONL file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                question = data.get("question")
                answer_raw = data.get("answer_raw")

                if question and answer_raw:
                    try:
                        # The answer_raw field is a JSON string, so it needs to be parsed again.
                        # It might also contain markdown code fences.
                        answer_data_str = answer_raw.strip()
                        if answer_data_str.startswith("```json"):
                            answer_data_str = answer_data_str[7:-3].strip()
                        elif answer_data_str.startswith("```"):
                            answer_data_str = answer_data_str[3:-3].strip()
                        
                        answer_data = json.loads(answer_data_str)
                        answer = answer_data.get("answer")

                        if answer:
                            result = {"question": question, "answer": answer}
                            outfile.write(json.dumps(result) + '\n')
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse answer_raw for question: {question}")
                    except (TypeError, AttributeError):
                        print(f"Warning: Unexpected format in answer_raw for question: {question}")

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions and answers from a JSONL file.")
    parser.add_argument("input_file", help="The path to the input JSONL file.")
    parser.add_argument("output_file", help="The path to the output JSONL file.")
    args = parser.parse_args()

    extract_questions_answers(args.input_file, args.output_file)
    print(f"Successfully extracted questions and answers to {args.output_file}")
