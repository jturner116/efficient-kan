import torch
import torch.nn as nn
import torch.optim as optim
import time
from efficient_kan import KAN
from torch.profiler import profile, record_function, ProfilerActivity

torch.set_float32_matmul_precision("high")

# Define model
model = KAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Device: {device}")

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define loss
criterion = nn.CrossEntropyLoss()

# Synthetic data
batch_size = 64
input_data = torch.randn(batch_size, 28 * 28).to(device)
labels = torch.randint(0, 10, (batch_size,)).to(device)

# Warmup
# model = torch.compile(model)

for i in range(10):
    output = model(input_data)


# Profile forward pass

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        output = model(input_data)
        prof.step()

prof.export_chrome_trace("profile.json")


# Profile backward pass
loss = criterion(output, labels)
loss.backward()

# Memory usage (optional, only if on CUDA)
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e6} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e6} MB")
