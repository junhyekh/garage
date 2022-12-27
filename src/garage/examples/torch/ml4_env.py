#!/usr/bin/env python3

from collections import OrderedDict
from metaworld import Benchmark, _make_tasks, _ML_OVERRIDE
import ml4_env_dict as _env_dict


class ML4(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = _env_dict.ML4['train']
        self._test_classes = _env_dict.ML4['test']
        train_kwargs = _env_dict.ml4_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _ML_OVERRIDE,
                                        seed=seed)
        test_kwargs = _env_dict.ml4_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes, test_kwargs,
                                       _ML_OVERRIDE,
                                       seed=seed)

