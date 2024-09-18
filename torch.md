参考https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html 这部分的learning

tensor的操作在这里https://pytorch.org/docs/stable/tensors.html#tensor-class-reference



# Tensor

Tensor很像多维数组。

## initializeing a Tensor

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```



```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```



```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```



## Attribute

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```



## Operation

**move to GPU**

```python
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

**Standard numpy-like indexing and slicing:**

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
```

**Joining tensors** You can use `torch.cat` to concatenate a sequence of tensors along a given dimension

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

**Arithmetic operations**

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```



**Single-element tensors** If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using `item()`:

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

**In-place operations** Operations that store the result into the operand are called in-place. They are denoted by a `_` suffix. For example: `x.copy_(y)`, `x.t_()`, will change `x`.\

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
```



>In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.



**Tensor to NumPy array**

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

**NumPy array to Tensor**

```python
n = np.ones(5)
t = torch.from_numpy(n)
```



# Dataset

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

-   `root` is the path where the train/test data is stored,
-   `train` specifies training or test dataset,
-   `download=True` downloads the data from the internet if it’s not available at `root`.
-   `transform` and `target_transform` specify the feature and label transformations



##  Iterate and visualize the dataset

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```



## customize your dataset

继承dataset，必须实现`__init__` `__len__` `__getitem__`

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```



# Transforms

数据并不可以直接用来训练，需要进行一些转换。

所有的TorchVision datesets有两个参数 `transform`来修饰features `target_transform`来修饰lables

The FashionMNIST features are in PIL Image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use `ToTensor` and `Lambda`.

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

## ToTensor()

[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor) converts a PIL image or NumPy `ndarray` into a `FloatTensor`. and scales the image’s pixel intensity values in the range [0., 1.]



## Lambda Transform

使用用户自定义的转换函数。

 It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) which assigns a `value=1` on the index as given by the label `y`.

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```



# Build the neural network

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```



## device

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```



## define the class

We define our neural network by subclassing `nn.Module`, and initialize the neural network layers in `__init__`. Every `nn.Module` subclass implements the operations on input data in the `forward` method.

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```



打印模型

```python
model = NeuralNetwork().to(device)
print(model)
```



如何使用model？我们直接传入input data，这个会自动执行forward函数和一些后台操作，不要直接调用forward()

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print(f"the logits is \n{logits}")
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

打印结果如下

```shell
the logits is 
tensor([[-0.0360,  0.1225,  0.0484, -0.0529,  0.0175, -0.0073, -0.0122, -0.1172,
          0.0487, -0.0214]], device='cuda:0', grad_fn=<AddmmBackward0>)
Predicted class: tensor([1], device='cuda:0')
```

因此后面的dim是从第一维而不是第0维开始。

softmax以后也是二维的

后面的argmax是返回在第一维中最大的值的下标

```python
import torch  
  
# 创建一个一维张量  
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])  
print(tensor.argmax())  # 输出: 3，因为最大值是4.0，索引为3  
  
# 创建一个二维张量  
tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  
print(tensor2.argmax())  # 默认在整个张量上查找，输出: 3  
print(tensor2.argmax(axis=0))  # 沿着行查找，输出: tensor([1, 1])，表示每列最大值的行索引  
print(tensor2.argmax(axis=1))  # 沿着列查找，输出: tensor([1, 1])，表示每行最大值的列索引
```



  

## model layers

 我们以下面的batch为3的28*28大小的Tensor为例

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

out:

```python
torch.Size([3, 28, 28])
```

### 

### nn.Flatten

dim=0会被保留，其他维会变成一维

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
#输出torch.Size([3, 784])
```



### nn.Linear

linear transformation on the input.

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

out

```python
torch.Size([3, 20])
```



`__init__`

```python
class Greeter:  
    def __init__(self, name):  
        self.name = name  
  
    def __call__(self, greeting):  
        print(f"{greeting}, {self.name}!")  
  
# 创建 Greeter 类的实例  
greet = Greeter("Alice")  
  
# 使用函数调用语法调用实例的 __call__ 方法  
greet("Hello")  # 输出: Hello, Alice!
```

### nn.ReLU



### nn.Sequential

有序排列，数据顺着顺序流动。

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```



### nn.Softmax

The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class. `dim` parameter indicates the dimension along which the values must sum to 1.



## Model parameters

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```





# Automatic Differentiation

To compute gradients, PyTorch has a built-in differentiation engine called `torch.autograd`. It supports automatic computation of gradient for any computational graph.

eg

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

In this network, `w` and `b` are **parameters**, which we need to optimize. Thus, we need to be able to compute the gradients of loss function with respect to those variables. In order to do that, we set the `requires_grad` property of those tensors.

>You can set the value of `requires_grad` when creating a tensor, or later by using `x.requires_grad_(True)` method.



## Compute Gradients

To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters, namely, we need ∂loss/∂w and ∂loss/∂b under some fixed values of `x` and `y`. To compute those derivatives, we call `loss.backward()`, and then retrieve the values from `w.grad` and `b.grad`:

```python
loss.backward()
print(w.grad)
print(b.grad)
```

-   We can only obtain the `grad` properties for the leaf nodes of the computational graph, which have `requires_grad` property set to `True`. For all other nodes in our graph, gradients will not be available.
-   We can only perform gradient calculations using `backward` once on a given graph, for performance reasons. If we need to do several `backward` calls on the same graph, we need to pass `retain_graph=True` to the `backward` call.



## More on Computational Graphs

在有向无环图中，autograd保存着数据（tensor）和执行过的操作。在这个有向无环图中，叶子是输入的tensor，根是输出的tensor，通过从根到叶子，我们可以通过链式法则自动计算梯度。

在forward中，autograd同时做

-   执行定义的操作，获得输出tensor
-   保留有向无环图中的操作的梯度函数

在backward过程中，autograd执行以下：

-   通过每个.grad_fn计算梯度
-   accumulates them in the respective tensor’s `.grad` attribute
-   using the chain rule, propagates all the way to the leaf tensors.

>   **DAGs are dynamic in PyTorch** An important thing to note is that the graph is recreated from scratch; after each `.backward()` call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed.



如果loss function不是scalar，可以用Jacobian Matrix和Jacobian Product



# Optimizing Model Parameters

## Hyperparameters

-   **Number of Epochs** - the number times to iterate over the dataset
-   **Batch Size** - the number of data samples propagated through the network before the parameters are updated
-   **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

## Loss Function

Common loss functions include [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) (Mean Square Error) for regression tasks, and [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (Negative Log Likelihood) for classification. [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines `nn.LogSoftmax` and `nn.NLLLoss`.



## Optimizer

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop, optimization happens in three steps:

-   Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
-   Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the gradients of the loss w.r.t. each parameter.
-   Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
epochs = 5
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```



# Save and Reload

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```



```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```



When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:

类和参数一起保存

```python
torch.save(model, 'model.pth')

model = torch.load('model.pth', weights_only=False),
```



>This approach uses Python [pickle](https://docs.python.org/3/library/pickle.html) module when serializing the model, thus it relies on the actual class definition to be available when loading the model.