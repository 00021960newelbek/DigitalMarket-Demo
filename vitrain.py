from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
import torch
from PIL import Image

dataset_dir = r"dataset_split"  # your folder with subfolders
dataset = load_dataset("imagefolder", data_dir=dataset_dir)

print("Original splits:", list(dataset.keys()))


all_splits = list(dataset.keys())
print(f"Found splits: {all_splits}")

# Combine all splits
datasets_to_combine = [dataset[split] for split in all_splits]
combined = concatenate_datasets(datasets_to_combine)
print(f"Combined dataset size: {len(combined)}")

# Split into train (80%) and validation (20%)
dataset_split = combined.train_test_split(test_size=0.2, seed=42)
print(f"New train size: {len(dataset_split['train'])}")
print(f"New validation size: {len(dataset_split['test'])}")


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

print("\nDataset features:", combined.features)
print("First item keys:", combined[0].keys())


def preprocess_images(examples):

    # Process images
    images = examples['image']
    inputs = processor(images, return_tensors='pt')

    # Add labels
    inputs['labels'] = examples['label']

    return inputs


print("\nPreprocessing train dataset...")
train_dataset = dataset_split['train'].map(
    preprocess_images,
    batched=True,
    batch_size=32,
    remove_columns=['image']
)

print("Preprocessing validation dataset...")
eval_dataset = dataset_split['test'].map(
    preprocess_images,
    batched=True,
    batch_size=32,
    remove_columns=['image']
)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
eval_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

print("Preprocessing complete!")


num_classes = len(combined.features["label"].names)
print(f"\nNumber of classes: {num_classes}")
print(f"Class names: {combined.features['label'].names}")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./vit_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=3,
    push_to_hub=False,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
)



def collate_fn(batch):
    """
    Custom collate function to batch preprocessed data.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'labels': labels,
    }



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=collate_fn,
)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save final model
    model.save_pretrained("./vit_model")
    processor.save_pretrained("./vit_model")

    print("\n" + "=" * 50)
    print("Training completed and model saved!")
    print("=" * 50)