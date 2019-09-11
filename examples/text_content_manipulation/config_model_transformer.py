import copy
import texar as tx

random_seed = 1234
beam_width = 5
hidden_dim = 384
coverity_dim = 128
alpha = 0

def get_embedder_hparams(hidden_dim, name):
    return {
        'name': name,
        'dim': hidden_dim,
        'initializer': {
            'type': 'random_normal_initializer',
            'kwargs': {
                'mean': 0.0,
                'stddev': hidden_dim ** -0.5,
            },
        }
    }


embedders = {
    name: get_embedder_hparams(dim, '{}_embedder'.format(name))
    for name, dim in (
        ('sent', hidden_dim),
        ('entry', hidden_dim / 2),
        ('attribute', hidden_dim / 8),
        ('value', hidden_dim / 8 * 3))}

sent_encoder = {
    'dim': hidden_dim,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 384,
        'num_units': 384,
        'dropout_rate': 0.25
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': hidden_dim
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=hidden_dim)
}

sd_encoder = {
    'dim': hidden_dim,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 384,
        'num_units': 384,
        'dropout_rate': 0.25
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': hidden_dim
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim= hidden_dim)
}


decoder = copy.deepcopy(sd_encoder)
# rnn_cell = {
#     'type': 'LSTMBlockCell',
#     'kwargs': {
#         'num_units': hidden_dim,
#         'forget_bias': 0.
#     },
#     'dropout': {
#         'input_keep_prob': 0.8,
#         'state_keep_prob': 0.5,
#     },
#     'num_layers': 1
# }


# attention_decoder = {
#     'name': 'attention_decoder',
#     'attention': {
#         'type': 'LuongAttention',
#         'kwargs': {
#             'num_units': hidden_dim,
#         }
#     }
# }