#import the neccessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_id='gpt2'
tokenizer=GPT2Tokenizer.from_pretrained_model(model_id)
model_inst=GPT2LMHeadModel.from_pretrained_model(model_id)

# Set the model to evaluation mode
model_id.eval()