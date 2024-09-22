# VPR-Evaluation
This is a repo for evaluating VPR methods on various open-source datasets. This document will guide you on how to evaluate existing methods, how to import datasets for evaluation, and how to add new methods and datasets for evaluation from the code level.
# How to add new VPR methods to be evaluated
Create a new python script that includes the structure of your method and a get function in vpr_models directory, for example:
```
class TALANet(nn.Module):
    def __init__(self,
                ):
        super(TALANet, self).__init__()
        self.backbone
        self.aggregation

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

def get_tala():
    file_path = ""
    model_param = torch.load(file_path)

    model = TALANet()
    model.load_state_dict(model_param['state_dict'])
    model = model.eval()
```
Then add it to get_model() of \_init_.py in vpr_models directory, like:
```
elif method == "tala":
        model = tala.get_tala()
```

# Usage
If your datasets(check repo TALA to find out how to prepare your datasets) and method are prepared properly, simply run
```
python benchmark.py --method method_name
```

# TODO
Currently this tool is able to output the metric results(recall@k) correctly, but it seems there are a few problems in the logic of visualization, gonna make it right shortly. 
