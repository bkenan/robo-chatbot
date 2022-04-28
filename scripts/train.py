import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW
from transformers import AutoModel, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
from model import BERT
from torch.utils.data import TensorDataset, DataLoader, RandomSampler



def main():
    """
    Performs model training.
    Output: The saved model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    
    df = pd.read_excel("./data/patterns.xlsx")

    # Converting the labels into encodings
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Defining training set:
    train_text, train_labels = df["text"], df["label"]

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # Import BERT-base pretrained model
    bert = AutoModel.from_pretrained("bert-base-uncased")

    text = ["this is a test for the bert model"]
    # Encode the text
    encoded_input = tokenizer(text, padding=True,truncation=True, return_tensors='pt')
    print(encoded_input)

    # tokenize and encode sequences in the training set
    max_seq_len = 10
    tokens_train = tokenizer(
        train_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )


    # for train set
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    #define a batch size
    batch_size = 8
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)
    # DataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    # freeze all the parameters. This will prevent updating of model weights during fine-tuning.
    for param in bert.parameters():
        param.requires_grad = False
    model = BERT(bert)
    model = model.to(device)
    
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-3)
    
    #compute the class weights

    class_wts = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_labels),
                                            y = train_labels                                                    
                                        )
    
    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)
    # loss function
    cross_entropy = nn.NLLLoss(weight=weights) 
    
    # function to train the model
    def train():
        
        model.train()
        total_loss = 0
        
        # empty list to save model predictions
        total_preds=[]
        
        # iterate over batches
        for step,batch in enumerate(train_dataloader):
            
            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step,    len(train_dataloader)))
            # push the batch to gpu
            batch = [r.to(device) for r in batch] 
            sent_id, mask, labels = batch
            # get model predictions for the current batch
            preds = model(sent_id, mask)
            # compute the loss between actual and predicted values
            loss = cross_entropy(preds, labels)
            # add on to the total loss
            total_loss = total_loss + loss.item()
            # backward pass to calculate the gradients
            loss.backward()
            # clip the the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # update parameters
            optimizer.step()
            # clear calculated gradients
            optimizer.zero_grad()
            # model predictions are stored on GPU. So, push it to CPU
            preds=preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)
        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)
            
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds  = np.concatenate(total_preds, axis=0)
        #returns the loss and predictions
        return avg_loss, total_preds

    
    epochs = 100
    train_losses = []
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        #train model
        train_loss, _ = train()
        
        # append training
        train_losses.append(train_loss)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'\nTraining Loss: {train_loss:.3f}')
        
    # Visualization of Negative Log Likelihood loss:

    Epoch_n = list(range(1, 101))

    # plotting the points
    plt.plot(Epoch_n, train_losses)
    
    # naming the x axis
    plt.xlabel('Epoch')
    # naming the y axis
    plt.ylabel('Loss')
    
    # giving a title to the graph
    plt.title('Metrics')
    
    # function to show the plot
    plt.show()
    
    
    #Saving the model

    PATH = "model.pt"

    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    main()