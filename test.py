from numpy import *
import numpy as np
from init import functions
import matplotlib.pyplot as plt

input_size = 16
hidden_size = 32
hidden2_size = 26
output_size = 26
mu = 0.15
epochs = 2000

batch_all = 20000
filename = "letter-recognition.data"
inputs = []
targets = []

letters = []
with open(filename) as f:
    for line in f:
        letters.append([str(n) for n in line.strip().split(',')])

for x in letters:
    a_z = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    convert = dict((c, i) for i, c in enumerate(a_z))
    int_encode = [convert[x[0]]]
    onehot_encode = []
    for val in int_encode:
        letter = [0 for _ in range(len(a_z))]
        letter[val] = 1
        onehot_encode = letter
    targets.append(onehot_encode)
    inputs.append([int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]),
                    int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]), int(x[12]),
                    int(x[13]), int(x[14]), int(x[15]), int(x[16])])

targets = np.array(targets)
train_batch = int((batch_all / 5) * 4)
test_batch = batch_all / 5
train_set = inputs[:train_batch]
train_outputs = targets[:train_batch]
test_set = inputs[train_batch:batch_all]
test_outputs = targets[train_batch:batch_all]

ANN = functions(train_set, train_outputs, hidden_size, hidden2_size)
ANN.randomize()
all_error = []

for x in range(epochs):
    ANN.forward(ANN.I)
    error = ANN.backprop()
    all_error.append(error)
    ANN.update_weights(mu)

    if mod(x, 1000) == 0:
        print("error at epoch: ", x, ": ", error)

expected_outs = []
batch_all = shape(test_set)[0]
test_input = concatenate((test_set, -ones((batch_all, 1))), axis=1)
expected_output = ANN.forward(test_input)
print("ex: ", expected_output)
expected_outs.append(expected_output)
error = 0.5 * sum((expected_output - test_outputs) ** 2)

print("Test set error: ", error)
print("Accuracy: ", )

fig, ax = plt.subplots()

ax.set_xlabel('Epochs')
ax.set_ylabel('Error')
ax.set_title('Epochs vs. Error')
ax.plot([n for n in range(len(all_error))], all_error)

plt.show()