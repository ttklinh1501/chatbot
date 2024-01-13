import random
import json
from query import tokenize, bag_of_words
import numpy as np
import torch
from model import NeuralNet

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

#import time

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Hy"

def chat():
    print("Vui lòng nhập để bắt đầu, nhập 'Thoát' nếu bạn muốn kết thúc.")
    while True:
        sentence = input ("Bạn: ")
        if sentence == "Thoát":
            break

        sentence = tokenize(sentence)
        x = bag_of_words(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        #localtime = time.asctime(time.localtime(time.time()))

        probs = torch.softmax(output, dim = 1)
        prob = probs[0][predicted.item()]
            
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print( f"{bot_name}: Xin lỗi, mình không hiểu...")
    
chat()
