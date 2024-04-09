#import the neccessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_id='gpt2'
tokenizer=GPT2Tokenizer.from_pretrained_model(model_id)
model_inst=GPT2LMHeadModel.from_pretrained_model(model_id)

# Set the model to evaluation mode
model_id.eval()

def generate_response(prompt,max_length=50):

    #tokenize the prompt
    input=tokenizer.encode(prompt,return_tensor='pt')

    #Generate response
    with torch.no_grad():
        output=model_id.generate(input,max_length=max_length,num_return_sequance=1,pad_token_id=50256)

    #Decoding the ouput
    response=tokenizer.decode(output[0],skip_special_tokens=True)
    return response
