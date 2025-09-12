import json
import argparse
from pathlib import Path

def extract_confidence_from_answer(answer_raw):
    """Extract confidence score from the answer_raw JSON string"""
    import re
    
    try:
        # Try to parse as JSON first
        answer_data = json.loads(answer_raw.strip())
        return answer_data.get('confidence', 0.0)
    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, try to extract JSON from code blocks
        try:
            # Remove code block markers (```json and ```)
            cleaned = re.sub(r'```json\s*', '', answer_raw)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
            cleaned = cleaned.strip()
            
            # Try parsing the cleaned version
            answer_data = json.loads(cleaned)
            return answer_data.get('confidence', 0.0)
        except:
            pass
        
        # If still fails, try regex extraction as last resort
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', answer_raw)
        if confidence_match:
            return float(confidence_match.group(1))
        
        # If all methods fail, return 0.0 as default
        return 0.0

def filter_qna_by_confidence(input_file, high_confidence_file, low_confidence_file, threshold=0.7):
    """
    Filter QNA pairs based on confidence scores
    
    Args:
        input_file: Path to input JSONL file
        high_confidence_file: Path to output file for confidence >= threshold
        low_confidence_file: Path to output file for confidence < threshold
        threshold: Confidence threshold (default: 0.7)
    """
    
    input_path = Path(input_file)
    high_conf_path = Path(high_confidence_file)
    low_conf_path = Path(low_confidence_file)
    
    # Create output directories if they don't exist
    high_conf_path.parent.mkdir(parents=True, exist_ok=True)
    low_conf_path.parent.mkdir(parents=True, exist_ok=True)
    
    high_confidence_count = 0
    low_confidence_count = 0
    error_count = 0
    
    print(f"Filtering QNA pairs with confidence threshold: {threshold}")
    print(f"Input file: {input_path}")
    print(f"High confidence output (>= {threshold}): {high_conf_path}")
    print(f"Low confidence output (< {threshold}): {low_conf_path}")
    print("-" * 60)
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(high_conf_path, 'w', encoding='utf-8') as high_file, \
         open(low_conf_path, 'w', encoding='utf-8') as low_file:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the main QNA record
                qna_record = json.loads(line)
                
                # Extract confidence from answer_raw
                confidence = extract_confidence_from_answer(qna_record.get('answer_raw', ''))
                
                # Add confidence to the record for easier access
                qna_record['extracted_confidence'] = confidence
                
                # Write to appropriate file based on confidence
                if confidence >= threshold:
                    json.dump(qna_record, high_file, ensure_ascii=False)
                    high_file.write('\n')
                    high_confidence_count += 1
                else:
                    json.dump(qna_record, low_file, ensure_ascii=False)
                    low_file.write('\n')
                    low_confidence_count += 1
                    
                # Print progress every 100 records
                if line_num % 100 == 0:
                    print(f"Processed {line_num} records...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                error_count += 1
                continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("FILTERING COMPLETE")
    print("=" * 60)
    print(f"Total records processed: {high_confidence_count + low_confidence_count}")
    print(f"High confidence records (>= {threshold}): {high_confidence_count}")
    print(f"Low confidence records (< {threshold}): {low_confidence_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nOutput files created:")
    print(f"- High confidence: {high_conf_path} ({high_confidence_count} records)")
    print(f"- Low confidence: {low_conf_path} ({low_confidence_count} records)")

def analyze_confidence_distribution(input_file):
    """Analyze the distribution of confidence scores in the dataset"""
    
    confidence_scores = []
    error_count = 0
    
    print("Analyzing confidence score distribution...")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                qna_record = json.loads(line)
                confidence = extract_confidence_from_answer(qna_record.get('answer_raw', ''))
                confidence_scores.append(confidence)
            except Exception as e:
                error_count += 1
                continue
    
    if confidence_scores:
        confidence_scores.sort()
        total_records = len(confidence_scores)
        
        print(f"\nConfidence Score Analysis:")
        print(f"Total records: {total_records}")
        print(f"Min confidence: {min(confidence_scores):.2f}")
        print(f"Max confidence: {max(confidence_scores):.2f}")
        print(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")
        print(f"Median confidence: {confidence_scores[total_records//2]:.2f}")
        
        # Show distribution by ranges
        ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0), (1.0, 1.0)]
        print(f"\nDistribution by confidence ranges:")
        for low, high in ranges:
            if low == high:
                count = sum(1 for score in confidence_scores if score == low)
                print(f"  {low:.1f}: {count} records ({count/total_records*100:.1f}%)")
            else:
                count = sum(1 for score in confidence_scores if low <= score < high)
                print(f"  {low:.1f}-{high:.1f}: {count} records ({count/total_records*100:.1f}%)")
        
        # Show count above/below 0.7 threshold
        above_threshold = sum(1 for score in confidence_scores if score >= 0.7)
        below_threshold = total_records - above_threshold
        print(f"\nWith 0.7 threshold:")
        print(f"  >= 0.7: {above_threshold} records ({above_threshold/total_records*100:.1f}%)")
        print(f"  < 0.7: {below_threshold} records ({below_threshold/total_records*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Filter QNA pairs by confidence scores')
    parser.add_argument('--input', '-i', 
                       default='data/generated/qna_raw.jsonl',
                       help='Input JSONL file (default: data/generated/qna_raw.jsonl)')
    parser.add_argument('--high-output', '--high',
                       default='data/generated/qna_high_confidence.jsonl',
                       help='Output file for high confidence records (default: data/generated/qna_high_confidence.jsonl)')
    parser.add_argument('--low-output', '--low',
                       default='data/generated/qna_low_confidence.jsonl', 
                       help='Output file for low confidence records (default: data/generated/qna_low_confidence.jsonl)')
    parser.add_argument('--threshold', '-t',
                       type=float, default=0.7,
                       help='Confidence threshold (default: 0.7)')
    parser.add_argument('--analyze', '-a',
                       action='store_true',
                       help='Show confidence score distribution analysis only')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist!")
        return 1
    
    if args.analyze:
        analyze_confidence_distribution(args.input)
    else:
        filter_qna_by_confidence(args.input, args.high_output, args.low_output, args.threshold)
    
    return 0

if __name__ == "__main__":
    exit(main())
