import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
'''
DATA PREPROCESSING
'''
# data load
all_data = np.loadtxt("data/international-airline-passengers.csv", delimiter=",", usecols=1)

# data slicing (train, test)
test_data_size = 12
train_data = all_data[:-test_data_size] # 132
test_data = all_data[-test_data_size:]  # 12

# nomalize data
scaler = MinMaxScaler(feature_range=(-1, 1)) # normalizes our data
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# data reshaping
# input data : [sequence length, batch size, input size]
# input data : [12, 1, 1]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        # input sequence 와 output 을 묶어서 하나로 사용한다.
        # for 문에서 두 값을 떼어서 사용
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_window = 12   # input sequence length for training to 12.
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

'''
MODEL
'''
# Though our sequence length is 12, for each month we have only 1 value i.e. total number of passengers,
# therefore the input size will be 1.
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # hidden_cell shape: [(2 if bidirectional=True otherwise 1)*num_layers,
        #                       batch size,
        #                       (proj_size if proj_size > 0 otherwise hidden_size)]
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        # reshape
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

'''
TRAINING
'''
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150

# 데이터를 한 번에 가져오지 않고 하나씩 가져옴
# => for seq, labels in train_inout_seq:
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        print(seq)
        print(labels)

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

'''
TESTING
'''
fut_pred = 12
test_inputs = train_data_normalized[-train_window:].tolist()
#??

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)

'''
PRINT
'''
x = np.arange(132, 144, 1)  # numeric values for the last 12 months.

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(all_data)
plt.plot(x, actual_predictions)
plt.show()

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(all_data)
plt.plot(x, actual_predictions)
plt.show()

