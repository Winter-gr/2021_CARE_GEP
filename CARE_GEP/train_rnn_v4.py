import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
'''
DATA PREPROCESSING
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'weights/%s.pth' % 'LSTM_passengers'
# 할일
#   저장 경로 더 자세히 정리 하기
#   loading 경로도 그에 맞춰서 정리
#   GPU, CPU간 모델 불러오기 정리해서 적용
#   CUDA 적용해야 하는 부분 모두 찾아서 넣기

# data load ','로 구분, col=1 을 데이터로 가져옴
all_data = np.loadtxt("data/international-airline-passengers.csv", delimiter=",", usecols=1)

# data slicing (train, test)
test_data_size = 12
train_data = all_data[:-test_data_size] # 132
test_data = all_data[-test_data_size:]  # 12

# nomalize data
scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# data reshaping
# input data : [sequence length, batch size, input size]
# input data : [12, 1, 1]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]          # input sequence
        train_label = input_data[i+tw:i+tw+1]   # output, lable
        # input sequence 와 output 을 한 행에 묶어서 하나씩 사용한다.
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_window = 12   # input sequence length for training to 12.
train_inout_seq = create_inout_sequences(train_data_normalized, train_window) #preprocessing

'''
MODEL
'''
# Though our sequence length is 12, for each month we have only 1 value i.e. total number of passengers,
# therefore the input size will be 1.
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        # === LSTM input 기본 param 정리 ===
        # input_size=한 시점에서 들어오는 특징값의 개수. 여기서는 승객 수(passengers) 1개의 특징
        # hidden_size=은닉층 하나의 셀 개수
        # num_layers=은닉층 개수
        # batch_first=batch size를 가장 먼저 할 것인지 확인하는 것 Default:False
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        # initialize hidden cell
        # hidden_cell shape: [D, num_layers, batch_size, H_out]
        # D = (2 if bidirectional=True otherwise 1) * num_layers
        # H_out = (proj_size if proj_size > 0 otherwise hidden_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))  # view(len(input_seq), -1) : reshape
        return predictions[-1]

'''
TRAINING
'''
model = LSTM()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

epochs = 150

# for 문으로 전체 에폭 돌리기
for i in range(epoch, epoch + epochs):
    # for 문으로 데이터 전체를 한 번 돌리기. 데이터 전체를 한번에 가져오지 않고 하나씩 가져온다.
    for seq, label in train_inout_seq:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()
        # initialize hidden cell
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        output = model(seq)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, PATH)


    if i%25 == 1:
        print(f'epoch: {i:3} loss: {loss.item():10.8f}')

print(f'epoch: {i:3} loss: {loss.item():10.10f}')

'''
TESTING
'''
fut_pred = 12   # future predict, train data 개수
test_inputs = train_data_normalized[-train_window:].tolist()
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size)) #initialize hidden cell
        test_inputs.append(model(seq).item())

# Nomalization 했던 값을 다시 바꿔준다. # 범위가 없이 다음 값을 예측하는 방식일 때만 필요하다. one-hot 인코딩을 하는 경우 안 필요할 수도?
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

