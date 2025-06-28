from esg_extractor import ESGExtractor
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import sys

def visualize_extraction(pdf_path, output_dir, max_pages=None):
    """
    Visualize the ESG metrics extraction results
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save visualization results
        max_pages: Maximum number of pages to process
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize extractor
    extractor = ESGExtractor()
    
    # Extract pages from PDF
    pages = extractor.extract_pages_from_pdf(pdf_path)
    
    if max_pages:
        pages = pages[:max_pages]
    
    # Process each page
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}")
        
        # Analyze page
        result = extractor.analyze_page(page)
        
        # Save original image
        img_path = os.path.join(output_dir, f"page_{i+1}_original.jpg")
        page["image"].save(img_path)
        
        # Visualize metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(page["image"])
        ax.axis('off')
        
        # Add metric annotations
        y_pos = 0.05
        for category, metrics in result["structured_metrics"].items():
            if metrics:  # Only add annotation if metrics found
                for metric, values in metrics.items():
                    # Create annotation text
                    annotation = f"{category.upper()}: {metric} = {', '.join(values)}"
                    ax.text(0.05, y_pos, annotation, transform=ax.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                    y_pos += 0.05
        
        # Save visualization
        viz_path = os.path.join(output_dir, f"page_{i+1}_annotated.jpg")
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        # Save raw extraction result
        json_path = os.path.join(output_dir, f"page_{i+1}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    print(f"Visualization complete. Results saved to {output_dir}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <pdf_path> <output_dir> [max_pages]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    max_pages = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    visualize_extraction(pdf_path, output_dir, max_pages)

if __name__ == "__main__":
    main()
