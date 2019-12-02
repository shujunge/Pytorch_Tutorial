import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()
print(model)
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./my_model.pt")
output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output.size())
print(output[0,:5])
