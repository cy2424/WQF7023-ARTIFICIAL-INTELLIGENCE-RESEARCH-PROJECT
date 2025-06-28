from datasets import Dataset
import torch
from transformers import Trainer, TrainingArguments
import json
import os
from esg_extractor import ESGExtractor

class ESGFineTuner:
    def __init__(self, model_path="ibm-granite/granite-vision-3.3-2b", output_dir="./esg-fine-tuned-model"):
        self.extractor = ESGExtractor(model_path)
        self.model = self.extractor.model
        self.processor = self.extractor.processor
        self.output_dir = output_dir
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def prepare_dataset(self, dataset_path):
        """
        Prepare training dataset from annotated ESG reports
        
        Expected format:
        [
            {
                "image_path": "chunyew/dataset/page_image.jpg",
                "prompt": "Extract all ESG metrics from this page",
                "response": "Scope 1 emissions: 25,000 tCO2e"
            },
            
        ]
        """
        with open(dataset_path, 'r') as f:
            dataset_records = json.load(f)
        
        dataset_dict = {
            "image_paths": [],
            "prompts": [],
            "responses": []
        }
        
        for record in dataset_records:
            dataset_dict["image_paths"].append(record["image_path"])
            dataset_dict["prompts"].append(record["prompt"])
            dataset_dict["responses"].append(record["response"])
        
        return Dataset.from_dict(dataset_dict)
    
    def preprocess_function(self, examples):
        """Process dataset examples for training"""
        from PIL import Image
        
        images = [Image.open(image_path).convert("RGB") for image_path in examples["image_paths"]]
        
        # Create conversation format
        conversations = []
        for i in range(len(images)):
            import base64
            from io import BytesIO
            
            # Convert image to base64
            buffered = BytesIO()
            images[i].save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_b64 = f"data:image/jpeg;base64,{img_str}"
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_b64},
                        {"type": "text", "text": examples["prompts"][i]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": examples["responses"][i]
                }
            ]
            conversations.append(conversation)
        
        # Process the examples
        inputs = []
        for conv in conversations:
            processed = self.processor.apply_chat_template(
                conv,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt"
            )
            inputs.append(processed)
        
        # Create PyTorch tensors for training
        model_inputs = {
            "input_ids": torch.stack([x["input_ids"] for x in inputs]),
            "attention_mask": torch.stack([x["attention_mask"] for x in inputs])
        }
        
        return model_inputs
    
    def fine_tune(self, train_dataset_path, eval_dataset_path=None, batch_size=4, epochs=3):
        """Fine-tune the model on ESG-specific data"""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset_path)
        
        # If eval dataset provided, use it, otherwise split train dataset
        if eval_dataset_path:
            eval_dataset = self.prepare_dataset(eval_dataset_path)
        else:
            # Split train dataset 80/20
            train_eval = train_dataset.train_test_split(test_size=0.2)
            train_dataset = train_eval["train"]
            eval_dataset = train_eval["test"]
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=batch_size
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=batch_size
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=5e-5,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none"  # Disable wandb, tensorboard etc.
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda data: self.processor.pad(data, return_tensors="pt"),
        )
        
        # Start training
        print("Starting fine-tuning")
        trainer.train()
        
        # Save fine-tuned model
        print(f"Training complete. Saving model to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
        return self.model, self.processor


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune VLM for ESG metrics extraction')
    parser.add_argument('--train_data', required=True, help='Path to training data JSON file')
    parser.add_argument('--eval_data', help='Path to evaluation data JSON file (optional)')
    parser.add_argument('--model_path', default='ibm-granite/granite-vision-3.3-2b', help='Base model path')
    parser.add_argument('--output_dir', default='./esg-fine-tuned-model', help='Output directory for fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = ESGFineTuner(model_path=args.model_path, output_dir=args.output_dir)
    
    # Fine-tune model
    fine_tuner.fine_tune(
        train_dataset_path=args.train_data,
        eval_dataset_path=args.eval_data,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()
