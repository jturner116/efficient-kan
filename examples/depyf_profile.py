import torch
from torch import _dynamo as torchdynamo
from efficient_kan import KAN
from torch.profiler import profile, record_function, ProfilerActivity
import depyf

torch.set_float32_matmul_precision("high")

model = KAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
compiled_model = torch.compile(model)

# Warmup
for _ in range(10):
    compiled_model(torch.randn(64, 28 * 28).to(device))


def main(inputs):
    for _ in range(100):
        compiled_model(inputs)


if __name__ == "__main__":
    inputs = torch.randn(64, 28 * 28).to(device)
    with depyf.prepare_debug("./depyf_debug"):
        main(inputs)
