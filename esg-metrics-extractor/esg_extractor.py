import os
import json
import fitz  # PyMuPDF
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import base64
from io import BytesIO
import re
from huggingface_hub import scan_cache_dir

class ESGExtractor:
    def __init__(self, model_path="ibm-granite/granite-vision-3.3-2b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Check if model is available offline
        self.check_model_availability(model_path)
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path).to(self.device)
        
        # ESG metrics schema
        self.esg_schema = {
            "emissions": ["scope 1", "scope 2", "scope 3", "carbon footprint", "ghg"],
            "energy": ["renewable", "consumption", "efficiency", "kwh", "mwh"],
            "water": ["consumption", "recycled", "discharge", "withdrawal"],
            "waste": ["recycled", "landfill", "hazardous", "non-hazardous"],
            "diversity": ["gender", "ethnicity", "inclusion", "equality"]
        }
    
    def check_model_availability(self, model_path):
  
        try:
            cache_info = scan_cache_dir()
            model_cached = False
            
            for repo in cache_info.repos:
                if model_path in repo.repo_id:
                    model_cached = True
                    print(f"Model {model_path} found in cache at {repo.repo_path}")
                    break
            
            if not model_cached:
                print(f"Warning: Model {model_path} not found in local cache.")
                print("This might require internet connection for first-time download.")
                print("For fully offline use, pre-download the model:")
                print(f"  from huggingface_hub import snapshot_download")
                print(f"  snapshot_download(repo_id='{model_path}')")
            
            return model_cached
        except Exception as e:
            print(f"Error checking model cache: {e}")
            return False
    
    def extract_pages_from_pdf(self, pdf_path):
        """Extract pages from PDF as images and text"""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get text
            text = page.get_text()
            
            # Get image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pages.append({"page_num": page_num, "text": text, "image": img})
            
        return pages
    
    def analyze_page(self, page_data):
        """Analyze a single page using VLM"""
        img = page_data["image"]
        
        # Convert PIL image to base64 for the conversation template
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_b64 = f"data:image/jpeg;base64,{img_str}"
        
        # First prompt - general ESG metric identification
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_b64},  # Base64 encoded image
                    {"type": "text", "text": "Extract all ESG metrics mentioned on this page. Include all numeric values for emissions, energy, water, waste, and diversity metrics. If there are tables or charts, describe their content in detail."},
                ],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        # Move each tensor to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        output = self.model.generate(**inputs, max_new_tokens=500)
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Process and structure the response
        structured_metrics = self._process_response_to_structured_data(response)
        
        return {
            "page_num": page_data["page_num"],
            "raw_response": response,
            "structured_metrics": structured_metrics
        }
    
    def _process_response_to_structured_data(self, response):
        """Convert model response to structured ESG data"""
        structured_data = {category: {} for category in self.esg_schema}
        
        # Improved rule-based extraction
        for category, keywords in self.esg_schema.items():
            for keyword in keywords:
                if keyword in response.lower():
                    sentences = [s for s in response.split('.') if keyword in s.lower()]
                    for sentence in sentences:
                        # Improved regex to capture values with units
                        numbers = re.findall(r'\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?', sentence)
                        if numbers:
                            structured_data[category][keyword] = numbers
        
        return structured_data
    
    def process_report(self, pdf_path, max_pages=None):
        """Process entire ESG report"""
        pages = self.extract_pages_from_pdf(pdf_path)
        
        if max_pages:
            pages = pages[:max_pages]
        
        results = []
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)}")
            page_result = self.analyze_page(page)
            results.append(page_result)
        
        # Consolidate results
        consolidated = {category: {} for category in self.esg_schema}
        for page_result in results:
            for category, metrics in page_result["structured_metrics"].items():
                for metric_name, values in metrics.items():
                    if metric_name not in consolidated[category]:
                        consolidated[category][metric_name] = values
                    else:
                        consolidated[category][metric_name].extend(values)
        
        return {
            "page_results": results,
            "consolidated_metrics": consolidated
        }
    
    def process_pages_batch(self, pages, batch_size=4):
        """Process pages in batches to manage memory"""
        results = []
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i+batch_size]
            batch_results = []
            
            for page in batch:
                page_result = self.analyze_page(page)
                batch_results.append(page_result)
                
            results.extend(batch_results)
            
            # Optional: force garbage collection after each batch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
    
    def evaluate(self, extracted_data, ground_truth):
        """Evaluate extraction performance against ground truth"""
        # Flatten both extracted data and ground truth for comparison
        extracted_flat = []
        truth_flat = []
        
        for category in self.esg_schema:
            for metric in ground_truth.get(category, {}):
                truth_values = ground_truth[category][metric]
                extracted_values = extracted_data.get(category, {}).get(metric, [])
                
                # Convert to sets for comparison
                truth_set = set(map(str, truth_values))
                extracted_set = set(map(str, extracted_values))
                
                # Add to flat lists for precision/recall calculation
                for value in truth_set:
                    truth_flat.append(f"{category}.{metric}.{value}")
                
                for value in extracted_set:
                    extracted_flat.append(f"{category}.{metric}.{value}")
        
        # Calculate metrics
        true_positives = len(set(truth_flat) & set(extracted_flat))
        precision = true_positives / len(extracted_flat) if extracted_flat else 0
        recall = true_positives / len(truth_flat) if truth_flat else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "extracted_count": len(extracted_flat),
            "ground_truth_count": len(truth_flat)
        }
        
    def save_results(self, results, output_path):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
