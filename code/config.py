from __future__ import division
from __future__ import print_function

import os.path as osp

import numpy as np

from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 4
__C.LOGDIR = None
__C.COMET_PROJECT_NAME = None

__C.EVAL = False
__C.TEST = False
__C.TEST_BATCH_SIZE = 256
__C.SAMPLE = ''
__C.resume_model = None
__C.resume_model_ema = None
__C.start_epoch = None
# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCHS = 25
__C.TRAIN.SNAPSHOT_INTERVAL = 5
__C.TRAIN.WEIGHT_INIT = "xavier_uniform"
__C.TRAIN.CLIP_GRADS = True
__C.TRAIN.CLIP = 8
# __C.TRAIN.MAX_STEPS = 4
__C.TRAIN.EALRY_STOPPING = True
__C.TRAIN.PATIENCE = 5
__C.TRAIN.VAR_DROPOUT = False
# __C.TRAIN = dict(__C.TRAIN)
__C.TRAIN.RADAM = False
__C.TRAIN.CURRICULUM = False
__C.TRAIN.CURRICULUM_STEP = 2

# Dataset options
__C.DATASET = edict()
__C.DATASET.DATASET = 'clevr'
__C.DATASET.DATA_DIR = ''
__C.DATASET.COGENT = ''
__C.DATASET.IGNORE_TOKEN = ''
# __C.DATASET = dict(__C.DATASET)
__C.pretrained_vocabs = edict(
    glove=edict(
        values_file='',
        tok2idx_file='',
    ),
)
__C.model = edict(
    init_mem='random',
    max_step=4,
    separate_syntax_semantics=False,
    pretrained_vocab='',
    common=edict(module_dim=512),
    input_unit=edict(
        wordvec_dim=300,
        rnn_dim=512,
        bidirectional=True,
        separate_syntax_semantics_embeddings=False,
        stem_act='ELU',
        in_channels=1024,
        num_blocks=0,
        use_stem=True,
        ),
    control_unit=edict(
        control_feed_prev=True,
        control_cont_activation='TANH',
    ),
    read_unit=edict(
        num_blocks=3,
        film_from=None,
        in_channels=1024,
        use_stem=True,
    ),
    write_unit=edict(
        rtom=False,
        self_attn=False,
        gate=False,
        gate_shared=False,
        ),
    output_unit=edict(),
)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)
