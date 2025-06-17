import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load and process training data
json_data = "full_30k_training_data.json"
with open(json_data, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = []
for para in data:
    for question in para['questions']:
        pairs.append({'question': question, 'answer': para['paragraph_text']})

print(f"Total pairs: {len(pairs)}")
for i in range(10):
    print(pairs[i])

# Prepare InputExamples
train_examples = [InputExample(texts=[pair["question"], pair["answer"]]) for pair in pairs]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

# Load PhoBERT model using SentenceTransformers modules
phobert_model = models.Transformer('vinai/phobert-base', max_seq_length=256)
pooling_model = models.Pooling(phobert_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[phobert_model, pooling_model])

# Move model to CUDA
model.to(device)

# Training with MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Fine-tuning
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=10,
    warmup_steps=100,
    use_amp=True  # Mixed precision training for faster performance on GPU
)

# Save the fine-tuned model
model.save('fine-tuned-phobert-sentence-transformer')
