from data import CoronaData
from torch.utils.data import DataLoader
import torch
from model import CoronaClassifier
from torch.optim import Adam

batch_size = 4
num_epoch = 20
max_token_len = 130
log_every = 500
train_file = "./drive/My Drive/AAA folder/Corona_classifier/dataset/train.pkl"
valid_file = "./drive/My Drive/AAA folder/Corona_classifier/dataset/valid.pkl"
model_path = "./drive/My Drive/AAA folder/Corona_classifier/model/model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} as device")

def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list, label_list

def train():
    x, y = CoronaData.read_file(train_file)
    x_val, y_val = CoronaData.read_file(valid_file)
    print(len(x))
    print(len(x_val))
    train_data = CoronaData(x, y)
    valid_data = CoronaData(x_val, y_val)
    del x, y, x_val, y_val
    train_dl = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    corona_model = CoronaClassifier(max_token_len)
    corona_model.to(device)
    optimizer = torch.optim.Adadelta(corona_model.parameters(), lr=1e-4)
    best_loss = float("inf")
    for epoch in range(num_epoch):
        print(f"Epoch {epoch} running")
        corona_model.train()
        print("Model is training")
        train_loss, optimizer = corona_model.trainer(train_dl, log_every, device, optimizer=optimizer)
        print(f"Epoch: {epoch} | train loss: {train_loss:.8f}")
        print("Model is evaluating")
        corona_model.eval()
        eval_loss, _ = corona_model.trainer(valid_dl, log_every, device, evaluate=True)
        if eval_loss < best_loss:
            print("New best model found.")
            best_loss = eval_loss
            model_state = {
                'model_state_dict': corona_model.state_dict(),
                'loss': eval_loss
            }
            corona_model.save_checkpt(model_state, model_path)
        print(f"Epoch: {epoch} | eval loss: {eval_loss:.8f}")
    
    print(f"Model training completed with loss: {best_loss}")

if __name__ == "__main__":
    train()
        



