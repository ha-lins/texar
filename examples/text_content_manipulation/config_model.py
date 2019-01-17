dim = 384
import tensorflow as tf

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
        ('sent', dim),
        ('entry', dim / 2),
        ('attribute', dim / 8),
        ('value', dim / 8 * 3))}

sent_encoder = {
    'name': 'sent_encoder',
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': dim
        }
    }
}

sd_encoder = {
    'name': 'sd_encoder',
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
    'name': 'decoder'
}

attention_decoder = {
    'name': 'attention_decoder',
    'attention': {
        'type': tf.contrib.seq2seq.LuongMonotonicAttention,
        # 'type': 'LuongAttention',
        'kwargs': {
            'num_units': dim,
        }
    }
}


align_rnn_cell = {
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

align_attention_decoder = {
    'name': 'align_attention_decoder',
    'attention': {
        'type': tf.contrib.seq2seq.LuongMonotonicAttention,
        #'type': 'LuongAttention',
        'kwargs': {
            'num_units': dim,
        }
    }
}
