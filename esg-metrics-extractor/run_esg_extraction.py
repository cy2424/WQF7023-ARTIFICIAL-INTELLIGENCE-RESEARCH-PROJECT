import argparse
import json
import time
import psutil
import os
from esg_extractor import ESGExtractor

def main():
    parser = argparse.ArgumentParser(description='Extract ESG metrics from sustainability report')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--output', default='esg_metrics.json', help='Output JSON file')
    parser.add_argument('--ground_truth', help='Path to ground truth JSON for evaluation')
    parser.add_argument('--max_pages', type=int, help='Maximum pages to process')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing pages')
    args = parser.parse_args()
    
    # Initialize the extractor
    extractor = ESGExtractor()
    
    # Track resource usage
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process the report
    results = extractor.process_report(args.pdf, args.max_pages)
    
    # Measure resource usage
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Add performance metrics
    results["performance"] = {
        "runtime_seconds": end_time - start_time,
        "memory_usage_mb": end_memory - start_memory
    }
    
    # Evaluate if ground truth is provided
    if args.ground_truth:
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        evaluation = extractor.evaluate(results["consolidated_metrics"], ground_truth)
        results["evaluation"] = evaluation
        
        print(f"Evaluation Results:")
        print(f"Precision: {evaluation['precision']:.2f}")
        print(f"Recall: {evaluation['recall']:.2f}")
        print(f"F1 Score: {evaluation['f1_score']:.2f}")
    
    # Save results
    extractor.save_results(results, args.output)
    
    print(f"Processing completed in {results['performance']['runtime_seconds']:.2f} seconds")
    print(f"Memory usage: {results['performance']['memory_usage_mb']:.2f} MB")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
