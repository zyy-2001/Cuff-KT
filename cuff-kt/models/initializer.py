import torch


# trunk model init
def default_weight_init(tensor):
    torch.nn.init.xavier_uniform(tensor)
    # torch.nn.init.kaiming_normal_(tensor)


def default_bias_init(tensor):
    torch.nn.init.constant_(tensor, 0)


# lite plugin model init
def default_lite_plugin_init(layer):
    torch.nn.init.xavier_uniform(layer.weight, gain=0.001)
    # torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)


# naive plugin model init
def default_naive_plugin_init(layer):
    torch.nn.init.constant_(layer.weight, 0)
    torch.nn.init.constant_(layer.bias, 0)


if __name__ == '__main__':
    print('test')