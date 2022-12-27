#!/usr/bin/env python3

from collections import OrderedDict
from metaworld import Benchmark
import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerDrawerCloseEnvV2,
    SawyerWindowOpenEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerWindowCloseEnvV2)

ALL_V2_ENVIRONMENTS = _env_dict.ALL_V2_ENVIRONMENTS

ML4 = OrderedDict([
    ('train', OrderedDict([
        ('drawer-close-v2', SawyerDrawerCloseEnvV2),
        ('window-open-v2', SawyerWindowOpenEnvV2),
        ('drawer-open-v2', SawyerDrawerOpenEnvV2)
    ])),
    # NOTE(ycho): real meta-testing env
    # ('test', OrderedDict([
    #     ('window-close-v2', SawyerWindowCloseEnvV2)
    # ]))
    # NOTE(ycho): for logging all envs while training.
    ('test', OrderedDict([
        ('drawer-close-v2', SawyerDrawerCloseEnvV2),
        ('window-open-v2', SawyerWindowOpenEnvV2),
        ('drawer-open-v2', SawyerDrawerOpenEnvV2),
        ('window-close-v2', SawyerWindowCloseEnvV2)
    ]))
])

ml4_train_args_kwargs = {
    key: dict(args=[],
              kwargs={
                  'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key),
    })
    for key, _ in ML4['train'].items()
}

ml4_test_args_kwargs = {
    key: dict(args=[],
              kwargs={'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML4['test'].items()
}

ML4_ARGS_KWARGS = dict(
    train=ml4_train_args_kwargs,
    test=ml4_test_args_kwargs,
)

