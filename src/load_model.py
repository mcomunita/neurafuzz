import torch
from src.gcntfilm import GCNTF


def load_model(model_data, device):
    model_meta = model_data.pop('model_data')

    if model_meta["model_type"] == "gcntf":
        model = GCNTF(**model_meta, device=device)
    else:
        raise NotImplementedError

    if 'state_dict' in model_data:
        state_dict = model.state_dict()
        for each in model_data['state_dict']:
            state_dict[each] = torch.tensor(model_data['state_dict'][each])
        model.load_state_dict(state_dict)

    return model
