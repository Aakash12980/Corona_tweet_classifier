from transformers import XLNetTokenizer
import tqdm
import torch

class Tokenizer():
    def __init__(self, max_len=150):
        self.max_len = max_len
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    def encode_batch(self, batch):
        src_tokens = self.tokenizer(batch[0], return_tensors="pt", add_special_tokens=True,
                padding="longest", truncation=True)
        labels = torch.tensor(batch[1]).unsqueeze(dim=-1)
        return src_tokens.input_ids, src_tokens.attention_mask, labels

    def encode_sent(self, sents):
        src_tokens = []
        for s in sents:
            tokens = self.tokenizer(s, return_tensors="pt", add_special_tokens=True,
                padding="longest", truncation=True)
            src_tokens.append([tokens.input_ids, tokens.attention_mask])

        return src_tokens

    def decode_sent_tokens(self, data):
        sents_list = []
        for sent in data:
            s = self.tokenizer.decode(sent, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sents_list.append(s)

        return sents_list


# x = ["Hello there", "I am here to tell you this."]
# tokenizer = Tokenizer(12)
# out= tokenizer.encode_sent(x)
# print(out)