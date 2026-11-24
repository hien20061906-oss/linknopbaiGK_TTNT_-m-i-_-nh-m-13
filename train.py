from torch import no_grad, randperm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss

def train_perceptron(model, dataset):
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        model.training_history = []
        while True:
            mistakes = 0
            for batch in dataloader:
                x, y = (batch['x'], batch['label'])
                label = y.item()
                if model.get_prediction(x) != label:
                    model.w.add_(label * x.reshape_as(model.w))
                    mistakes += 1
            model.training_history.append(mistakes)
            if mistakes == 0:
                break

def train_regression(model, dataset, *, learning_rate=0.003):
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    evaluation = DataLoader(dataset, batch_size=200, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.training_history = []
    for epoch in range(4000):
        model.train()
        for batch in dataloader:
            x, y = (batch['x'], batch['label'])
            optimizer.zero_grad()
            loss = regression_loss(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = (0.0, 0)
        with no_grad():
            for batch in evaluation:
                n = len(batch['x'])
                total += regression_loss(model(batch['x']), batch['label']).item() * n
                count += n
        mse = total / count
        model.training_history.append(mse)
        if mse <= 0.005:
            return
    raise RuntimeError('Q2 did not reach the chosen MSE target within 4000 epochs')

def train_digitclassifier(model, dataset):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy, best_parameters = (-1.0, None)
    model.training_history = []
    for epoch in range(30):
        model.train()
        total, count = (0.0, 0)
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            n = len(batch['x'])
            total += loss.item() * n
            count += n
        model.eval()
        with no_grad():
            accuracy = float(dataset.get_validation_accuracy())
        model.training_history.append({'epoch': epoch + 1, 'loss': total / count, 'dev_accuracy': accuracy})
        print(f'Q3 epoch {epoch + 1:02d} | loss={total / count:.4f} | dev_accuracy={accuracy:.2%}', flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = [p.detach().clone() for p in model.parameters()]
        if accuracy >= 0.98:
            break
    if best_parameters is not None:
        with no_grad():
            for p, best in zip(model.parameters(), best_parameters):
                p.copy_(best)
    model.eval()

def train_languageid(model, dataset, *, learning_rate=0.002, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    buckets = [
        dataset._encode(dataset.train_x[start:end], dataset.train_y[start:end])
        for start, end in dataset.train_buckets
    ]
    best_dev_accuracy = -1.0
    best_parameters = None
    model.training_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bucket_index in randperm(len(buckets)).tolist():
            words, labels = buckets[bucket_index]
            for indices in randperm(len(labels)).split(128):
                optimizer.zero_grad()
                logits = model(words[:, indices, :])
                loss = languageid_loss(logits, labels[indices])
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(indices)
        model.eval()
        with no_grad():
            _, predicted, correct = dataset._predict(split='dev')
            dev_accuracy = (tensor(predicted) == tensor(correct)).float().mean().item()
        model.training_history.append({'epoch': epoch+1, 'loss': total_loss/len(dataset), 'dev_accuracy': dev_accuracy})
        print(f'Q4 epoch {epoch + 1:02d} | loss={total_loss / len(dataset):.4f} | dev_accuracy={dev_accuracy:.2%}', flush=True)
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_parameters = [p.detach().clone() for p in model.parameters()]
    if best_parameters is not None:
        with no_grad():
            for parameter, best_parameter in zip(model.parameters(), best_parameters):
                parameter.copy_(best_parameter)
    model.eval()

def Train_DigitConvolution(model, dataset):
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_accuracy, best_parameters = (-1.0, None)
    model.training_history = []
    for epoch in range(15):
        model.train()
        total, count = (0.0, 0)
        for batch in dataloader:
            optimizer.zero_grad()
            loss = digitconvolution_Loss(model(batch['x']), batch['label'])
            loss.backward()
            optimizer.step()
            n = len(batch['x'])
            total += loss.item() * n
            count += n
        model.eval()
        with no_grad():
            accuracy = float(dataset.get_validation_accuracy())
        model.training_history.append({'epoch': epoch + 1, 'loss': total / count, 'dev_accuracy': accuracy})
        print(f'Q5 epoch {epoch + 1:02d} | loss={total / count:.4f} | dev_accuracy={accuracy:.2%}', flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = [p.detach().clone() for p in model.parameters()]
        if accuracy >= 0.85:
            break
    if best_parameters is not None:
        with no_grad():
            for p, best in zip(model.parameters(), best_parameters):
                p.copy_(best)
    model.eval()
