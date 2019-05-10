dim = 384
coverage_state_dim = 128


def get_embedder_hparams(dim, name):
    return {
        'name': name,
        'dim': dim,
        'initializer': {
            'type': 'random_normal_initializer',
            'kwargs': {
                'mean': 0.0,
                'stddev': dim ** -0.5,
            },
        }
    }


embedders = {
    name: get_embedder_hparams(dim, '{}_embedder'.format(name))
    for name, dim in (
        ('y_aux', dim),
        ('x_value', dim / 2),
        ('x_type', dim / 8),
        ('x_associated', dim / 8 * 3))}

y_encoder = {
    'name': 'y_encoder',
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': dim
        }
    }
}

x_encoder = {
    'name': 'x_encoder',
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': dim
        }
    }
}

rnn_cell = {
    'type': 'LSTMBlockCell',
    'kwargs': {
        'num_units': dim,
        'forget_bias': 0.
    },
    'dropout': {
        'input_keep_prob': 0.8,
        'state_keep_prob': 0.5,
    },
    'num_layers': 1
}

decoder = {
    'name': 'decoder',
    'copying': {
        'copying_probability_history': True,
        'coverage': True,
        'selective_read': False,
    }
}

attention_decoder = {
    'name': 'attention_decoder',
    'attention': {
        'type': 'LuongAttention',
        'kwargs': {
            'num_units': dim,
        }
    }
}

coverage_rnn_cell = {
    'type': 'GRUCell',
    'kwargs': {
        'num_units': coverage_state_dim,
    },
}
