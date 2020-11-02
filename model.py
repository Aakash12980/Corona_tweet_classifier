from transformers import XLNetForSequenceClassification
import torch.nn as nn
from tokenizer import Tokenizer
import torch

class CoronaClassifier(nn.Module):
    def __init__(self, max_len=150):
        super(CoronaClassifier, self).__init__()
        self.model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', return_dict=True)
        self.tokenizer = Tokenizer(max_len)

    def forward(self, batch, device):
        inputs_ids, attention_mask, labels = self.tokenizer.encode_batch(batch)
        loss, logits = self.model(inputs_ids.to(device), attention_mask.to(device), labels=labels.to(device))[:2]
        return loss, logits
    
    def trainer(self, data_loader, log_every, device, optimizer=None, evaluate = False):
        total_loss = 0
        step = 0
        for batch in data_loader:
            step += 1
            if not evaluate:
                self.zero_grad()
                optimizer.zero_grad()
            loss, _ = self(batch, device)
            total_loss += loss.item()
            if not evaluate:
                loss.backward()
                optimizer.step()
            if (step)%log_every == 0:
                print(f"Step: {step} | loss: {total_loss/(step):.8f}")
        return total_loss/step, optimizer

    def save_checkpt(self, state, model_path):
        torch.save(state, model_path)
        print(f"Model saved at {model_path}")

    def load_model(self, model_path, device):
        if device == "cpu":
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(model_path)

        self.load_state_dict(checkpoint["model_state_dict"])
        loss = checkpoint["loss"]
        print("Model successfully loaded")
        return loss

