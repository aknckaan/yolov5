import logging
import pathlib
import sys
from typing import Dict, Tuple

from omegaconf.omegaconf import DictConfig, OmegaConf
import yaml

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

logger = logging.getLogger(__name__)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __to_str__(self):
        return str(self.__dict__)

class ConfigConverter:
    @classmethod
    def from_hydra(
        cls, hydra_config: DictConfig, cwd: pathlib.Path
    ) -> Dict:
        """Convert hydra config to mmcv config.

        Args:
            hydra_config (DictConfig): A hydra config.
            cwd (pathlib.Path): A path of current working directory
                aquired by running `hydra.utils.get_original_cwd()`.

        Returns:
            Config: A mmcv config which is conveted from hydra config.
            Dict: A meta data used in mmcv.

        Raises:
            ValueError: If `hydra_config` does not have the attribute
                `mmcv_config_path`.

        """
        OmegaConf.set_readonly(hydra_config, True)

        try:
            yolov5_hparams_path: Final = cwd / hydra_config.yolov5_hparams_path
            yolov5_args_path: Final = cwd / hydra_config.yolov5_arguments_path
            yolov5_datset_config: Final = cwd / hydra_config.yolov5_datset_config
            yolov5_model_config: Final = cwd / hydra_config.yolov5_model_config


        except Exception:
            logging.error("config should have attribute `yolov5_args_path` and `yolov5_hparams_path`.")
            raise ValueError()

        conf = AttrDict(yaml.load(open(str(yolov5_args_path))))
        conf.hyp = str(yolov5_hparams_path)
        conf.data = str(yolov5_datset_config)
        conf.cfg = str(yolov5_model_config)

        return conf

    @classmethod
    def _get_mmcv_meta(cls) -> Dict:
        """Create meta data from mmcv config.

        Args:
            mmcv_config (Config): A mmcv config.

        Returns:
            Dict: A meta data dict.

        """
        meta = dict()

        env_info_dict: Final = collect_env()
        env_info: Final = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        meta["env_info"] = env_info
        # meta["config"] = mmcv_config.pretty_text
        # meta["seed"] = mmcv_config.seed
        # meta["exp_name"] = mmcv_config.exp_name

        return meta
