from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, ones, matmul
from torch.nn.functional import relu, softmax
from torch import movedim

# Q1 — Phân loại hai nhãn
class PerceptronModel(Module):

    def __init__(self, dimensions):
        super().__init__()
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        return self.w

    def forward(self, x):
        return matmul(self.w, x.reshape(-1))

    def get_prediction(self, x):
        return 1 if self(x).item() >= 0 else -1

# Q2 — Hồi quy hàm sin
class RegressionModel(Module):

    def __init__(self):
        super().__init__()
        self.hidden1 = Linear(1, 128)
        self.hidden2 = Linear(128, 128)
        self.output = Linear(128, 1)

    def forward(self, x):
        x = relu(self.hidden1(x))
        x = relu(self.hidden2(x))
        return self.output(x)

# Q3 — Nhận dạng chữ số
class DigitClassificationModel(Module):

    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        hidden_size = 200
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = relu(self.fc1(x))
        return self.fc2(x)

# Q4 — RNN nhận diện ngôn ngữ
class LanguageIDModel(Module):

    def __init__(self, hidden_size=192):
        super().__init__()
        self.num_chars = 47
        self.languages = ['English', 'Spanish', 'Finnish', 'Dutch', 'Polish']
        self.hidden_size = hidden_size
        self.character_layer = Linear(self.num_chars, self.hidden_size)
        self.recurrent_layer = Linear(self.hidden_size, self.hidden_size)
        self.classifier = Linear(self.hidden_size, len(self.languages))

    def forward(self, xs):
        hidden = relu(self.character_layer(xs[0]))
        for character in xs[1:]:
            candidate = relu(self.character_layer(character) + self.recurrent_layer(hidden))
            is_character = character.sum(dim=1, keepdim=True)
            hidden = is_character * candidate + (1 - is_character) * hidden
        return self.classifier(hidden)

def Convolve(input: tensor, weight: tensor):
    h, w = weight.shape
    out_h = input.shape[0] - h + 1
    out_w = input.shape[1] - w + 1
    output = input.new_zeros((out_h, out_w))
    for dy in range(h):
        for dx in range(w):
            output = output + weight[dy, dx] * input[dy:dy + out_h, dx:dx + out_w]
    return output

# Q5 — CNN nhận dạng chữ số
class DigitConvolutionalModel(Module):

    def __init__(self):
        super().__init__()
        output_size = 10
        self.convolution_weights = Parameter(ones((3, 3)))
        self.fc1 = Linear(26 * 26, 100)
        self.fc2 = Linear(100, output_size)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack([Convolve(sample, self.convolution_weights) for sample in x])
        x = x.flatten(start_dim=1)
        x = relu(self.fc1(x))
        return self.fc2(x)

# Q6 — Attention nhân quả
class Attention(Module):

    def __init__(self, layer_size, block_size):
        super().__init__()
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)
        self.register_buffer('mask', tril(ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.layer_size = layer_size

    def forward(self, input):
        length = input.size(1)
        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)
        scores = matmul(Q, movedim(K, 1, 2)) / self.layer_size ** 0.5
        scores = scores.masked_fill(self.mask[0, 0, :length, :length] == 0, float('-inf'))
        weights = softmax(scores, dim=-1)
        return matmul(weights, V)
