import timm
from pprint import pprint

model_names = timm.list_models(pretrained=True)

pprint(model_names)

print()

for i, model in enumerate(model_names):
    if i > 195:
        m = timm.create_model(model, pretrained=True)
        print(model)
        pprint(m.default_cfg)
        print()
