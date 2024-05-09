from transformers import AutoTokenizer
import torch
import numpy as np
import re
from more_itertools import chunked
from fastapi_app.training_evaluation.splitter import Splitter

LABEL_LIST = ['AttackOnReputation', 'ManipulativeWording']

class Classificator:
  def __init__(self, model_name, model_file_name, thresholds, batch_size, logging_file):
      self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)

      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

      self.model = torch.load(model_file_name, map_location=torch.device(self.device))
      self.model.to(self.device)
      self.model.eval()

      self.labels_threshold = {label:threshold for label,threshold in zip(LABEL_LIST, thresholds)}
      self.label_list = LABEL_LIST
      self.batch_size = batch_size
      self.logging_file = logging_file

      self.splitter = Splitter()


  def classify(self, text):
    sentences = self.splitter.split_sentences(text)
    

    log_str = ''
    
    techniques = {
      label: [] for label in self.label_list
    }
     
    
    for batch in chunked(sentences, self.batch_size):
      ids, mask, token_type_ids = self.encode(batch)
      with torch.no_grad():
        outputs = self.model(ids, mask, token_type_ids)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()


      if self.logging_file:
        for i, output in  enumerate(outputs):
          log_str += batch[i] + ';' + str(output[0]) + ';' + str(output[1]) + '\n'



      for i, output in  enumerate(outputs):
        for output_label, label in zip(output, self.label_list):
          if output_label >= self.labels_threshold[label]:
            techniques[label].append(batch[i])

    if self.logging_file:
      with open(self.logging_file, "a") as f:
        f.write(log_str)        
  
    return techniques


  def encode(self,texts):
    MAX_LEN = self.tokenizer.model_max_length
    if MAX_LEN > 1024:
        MAX_LEN = 512
    ids = torch.empty((len(texts), MAX_LEN), dtype=torch.long)
    masks = torch.empty((len(texts), MAX_LEN), dtype=torch.long)
    token_type_ids = torch.empty((len(texts), MAX_LEN), dtype=torch.long)
    for i, text in enumerate(texts):
      inputs = self.tokenizer.encode_plus(
              text,
              None,
              add_special_tokens=True,
              max_length=MAX_LEN,
              pad_to_max_length=True,
              return_token_type_ids=True
      )

      ids[i] = torch.tensor(inputs['input_ids'], dtype=torch.long).to(self.device)
      masks[i] = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(self.device)
      token_type_ids[i] = torch.tensor(inputs["token_type_ids"], dtype=torch.long).to(self.device)

    return ids, masks, token_type_ids