from transformers import AutoTokenizer
import torch
import numpy as np

LABEL_LIST = ['Attack on Reputation', 'Manipulative Wording']

class Classificator:
  def __init__(self, model_name, model_file_name, evaluation_threshold):
      self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)

      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

      self.model = torch.load(model_file_name, map_location=torch.device(self.device))
      self.model.to(self.device)
      self.model.eval()

      self.evaluation_threshold = evaluation_threshold
      self.label_list = LABEL_LIST

  def classify(self, text):
    ids, mask, token_type_ids = self.encode(text)

    with torch.no_grad():
      outputs = self.model(ids, mask, token_type_ids)
      outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
  
    outputs = np.array(outputs) >= self.evaluation_threshold
  
    techniques = []
    for has_label, label in zip(outputs[0], self.label_list):
      if has_label:
        techniques.append(label)
  
    return techniques


  def encode(self,text):
    MAX_LEN = self.tokenizer.model_max_length
    if MAX_LEN > 1024:
      MAX_LEN = 512

    inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True
    )

    ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(self.device)
    mask =  torch.tensor([inputs['attention_mask']], dtype=torch.long).to(self.device)
    token_type_ids = torch.tensor([inputs["token_type_ids"]], dtype=torch.long).to(self.device)

    return ids, mask, token_type_ids