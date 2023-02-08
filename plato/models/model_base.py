"""
Model base
"""

import torch.nn as nn


class ModelBase(nn.Module):
    """
    Basic model wrapper for static graph and dygrpah.

    _registry, register, by_name, create用于管理不同的子类（具体模型）
    具体的模型继承父类ModelBase，使用其register方法将子类注册到父类的_registry属性中，用于父类管理所有的子类
    """
    _registry = dict()

    @classmethod
    def register(cls, name):
        ModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return ModelBase._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        model_cls = ModelBase.by_name(hparams.model)
        return model_cls(hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add cmdline argument. """
        group = parser.add_argument_group("Model")
        group.add_argument("--init_checkpoint", type=str, default=None)
        group.add_argument("--model", type=str, default="UnifiedTransformer",
                           choices=["UnifiedTransformer"])
        args, _ = parser.parse_known_args()
        model_cls = ModelBase.by_name(args.model)
        model_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams):
        super(ModelBase, self).__init__()
        self.forward_task = [self._forward_image_match,
                             self._forward_image_caption,
                             self._forward_poem_language,
                             self._forward_image_poem,
                             self._forward_image_poem_match]
        self.collect_metrics_task = [self._collect_metrics_image_match,
                                     self._collect_metrics_image_caption,
                                     self._collect_metrics_poem_language,
                                     self._collect_metrics_image_poem,
                                     self._collect_metrics_image_poem_match]
        self.infer_task = [self._infer_image_match,
                           self._infer_image_caption,
                           self._infer_poem_language,
                           self._infer_image_poem,
                           self._infer_image_poem_match]

        self.init_checkpoint = hparams.init_checkpoint
        self.use_gpu = hparams.use_gpu
        self.fp16 = hparams.fp16
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        raise NotImplementedError

    def _forward_image_match(self, inputs, is_training):
        raise NotImplementedError

    def _forward_image_caption(self, inputs, is_training):
        raise NotImplementedError

    def _forward_poem_language(self, inputs, is_training):
        raise NotImplementedError

    def _forward_image_poem(self, inputs, is_training):
        raise NotImplementedError

    def _forward_image_poem_match(self, inputs, is_training):
        raise NotImplementedError

    def _collect_metrics_image_match(self, inputs, outputs):
        raise NotImplementedError

    def _collect_metrics_image_caption(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _collect_metrics_poem_language(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _collect_metrics_image_poem(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _collect_metrics_image_poem_match(self, inputs, outputs):
        raise NotImplementedError

    def _forward(self, inputs, is_training):
        """ Real forward process of model in different mode(train/test). """
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs):
        """ Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _optimize(self, loss):
        """ Optimize loss function and update model. """
        raise NotImplementedError

    def _infer(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def _infer_image_match(self, inputs):
        raise NotImplementedError

    def _infer_image_caption(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def _infer_poem_language(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def _infer_image_poem(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def _infer_image_poem_match(self, inputs):
        """ Real inference process of model. """
        raise NotImplementedError

    def forward(self, inputs, task_id, is_training=False):
        """
        Forward process, include real forward, collect metrices and optimize(optional)

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        if is_training:
            self.train()
        else:
            self.eval()

        outputs = self.forward_task[task_id](inputs, is_training)
        metrics = self.collect_metrics_task[task_id](inputs, outputs)
        # outputs = self._forward(inputs, is_training)
        # metrics = self._collect_metrics(inputs, outputs)
        loss = metrics["loss"]
        if is_training:
            self._optimize(loss)

        metrics = {k: v.cpu().detach().numpy() for k, v in metrics.items()}
        return metrics

    def infer(self, inputs, task_id):
        """
        Inference process.

        @params : inputs : input data
        @type : dict of numpy.ndarray/int/float/...
        """
        self.eval()
        results = self.infer_task[task_id](inputs)
        results = {name: results[name].cpu().detach().numpy() for name in results}
        return results
