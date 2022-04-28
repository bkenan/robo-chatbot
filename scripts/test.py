import numpy as np
import pandas as pd
import re
import torch
import random
import json
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, BertTokenizerFast
from scripts.model import BERT

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # Import BERT-base pretrained model
    bert = AutoModel.from_pretrained("bert-base-uncased")
    
    # freeze all the parameters.
    for param in bert.parameters():
        param.requires_grad = False
    model = BERT(bert)

    # Loading the model
    model.load_state_dict(torch.load("./models/model.pt", map_location=device))
    
    # Initializing the label encoder
    df = pd.read_excel("./data/patterns.xlsx")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # Reading the json file
    with open('./assets/intent.json', 'r') as json_data:
        intents = json.load(json_data)
        
    def get_prediction(str):
        str = re.sub(r"[^a-zA-Z ]+", "", str)
        test_text = [str]
        model.eval()
        
        tokens_test_data = tokenizer(
        test_text,
        max_length = 10,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
        )
        test_seq = torch.tensor(tokens_test_data["input_ids"])
        test_mask = torch.tensor(tokens_test_data["attention_mask"])
        
        preds = None
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis = 1)
        return le.inverse_transform(preds)[0]
    
    def get_response(message): 
        intent = get_prediction(message)
        for i in intents['intents']: 
            if i["tag"] == intent:
                result = random.choice(i["responses"])
                break
        if message == "*":
            result = "Ending chat!"
        return result
    
    def chat():
        bot_name = "KB"
        print("Robo-Advisor chat! (Press '*' to exit)")
        while True:
            sentence = input("You: ")
            print(f"{bot_name}: {get_response(sentence)}")
            if sentence == "*":
                break
    
    #print(chat())


if __name__ == '__main__':
    main()