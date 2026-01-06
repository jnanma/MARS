import json
import attr


@attr.s
class DataConfig:
    gene = attr.ib(type=str)
    pheno = attr.ib(type=str)
    plink = attr.ib(type=str)
    anno = attr.ib(type=str, default=None)


@attr.s
class GBLUPConfig:
    hiblup = attr.ib(type=str)
    threads = attr.ib(type=float)
    top = attr.ib(type=float, default=None)


@attr.s
class parameterConfig:
    xgbround = attr.ib(type=int, default=None)
    early_stopping = attr.ib(type=int, default=None)
    max_depth = attr.ib(type=int, default=None)
    eta = attr.ib(type=float, default=None)
    subsample = attr.ib(type=float, default=None)
    lm = attr.ib(type=int, default=None)
    alpha = attr.ib(type=float, default=None)
    l1_ratio = attr.ib(type=float, default=None)


@attr.s
class SnpSelectConfig:
    model = attr.ib(type=str, default=None)
    parameter = attr.ib(type=parameterConfig, default=None)


@attr.s
class svrConfig:
    kernel = attr.ib(type=str, default=None)


@attr.s
class xgbConfig:
    xgbround = attr.ib(type=int, default=None)
    early_stopping = attr.ib(type=int, default=None)
    max_depth = attr.ib(type=int, default=None)
    eta = attr.ib(type=float, default=None)
    lm = attr.ib(type=int, default=None)


@attr.s
class lgbConfig:
    round = attr.ib(type=int, default=None)
    early_stopping_round = attr.ib(type=int, default=None)
    max_depth = attr.ib(type=int, default=None)
    learning_rate = attr.ib(type=float, default=None)
    lambda_l2 = attr.ib(type=float, default=None)


@attr.s
class mlpConfig:
    lr = attr.ib(type=float, default=None)
    batch_size = attr.ib(type=int, default=None)
    epochs = attr.ib(type=int, default=None)
    early_stopping = attr.ib(type=int, default=None)
    loss = attr.ib(type=str, default=None)
    optimizer = attr.ib(type=str, default=None)
    weight_decay = attr.ib(type=float, default=None)
    layers = attr.ib(type=float, default=None)
    dropout = attr.ib(type=float, default=None)


@attr.s
class cnnConfig:
    lr = attr.ib(type=float, default=None)
    batch_size = attr.ib(type=int, default=None)
    epochs = attr.ib(type=int, default=None)
    early_stopping = attr.ib(type=int, default=None)
    loss = attr.ib(type=str, default=None)
    optimizer = attr.ib(type=str, default=None)
    weight_decay = attr.ib(type=float, default=None)
    dropout = attr.ib(type=float, default=None)
    

@attr.s
class swtfConfig:
    lr = attr.ib(type=float, default=None)
    batch_size = attr.ib(type=int, default=None)
    epochs = attr.ib(type=int, default=None)
    early_stopping = attr.ib(type=int, default=None)
    loss = attr.ib(type=str, default=None)
    optimizer = attr.ib(type=str, default=None)
    weight_decay = attr.ib(type=float, default=None)
    window_size = attr.ib(type=int, default=None)


@attr.s
class ModelConfig:
    SVR = attr.ib(type=svrConfig, default=None)
    RF = attr.ib(type=bool, default=None)
    XGBoost = attr.ib(type=xgbConfig, default=None)
    LightGBM = attr.ib(type=lgbConfig, default=None)
    MLP = attr.ib(type=mlpConfig, default=None)
    CNN = attr.ib(type=cnnConfig, default=None)
    SwimTransformer = attr.ib(type=swtfConfig, default=None)


@attr.s
class Config:
    trait = attr.ib(type=str)
    data = attr.ib(type=DataConfig)
    seed = attr.ib(type=int)
    ModelSetting = attr.ib(type=str)
    GBLUP = attr.ib(type=GBLUPConfig)
    SnpSelect = attr.ib(type=SnpSelectConfig, default=None)
    model = attr.ib(type=ModelConfig, default=None)

    @classmethod
    def from_json(cls, config_path):
        with open(config_path) as config_file:
            config = json.load(config_file)
            return Config.from_dict(config)

    @classmethod
    def from_dict(cls, config):
        config["trait"] = config["trait"]
        config["data"] = DataConfig(**config["data"])
        config["seed"] = config["seed"]
        config["ModelSetting"] = config["ModelSetting"]
        config["GBLUP"] = GBLUPConfig(**config["GBLUP"])
        config["SnpSelect"] = SnpSelectConfig(**config["SnpSelect"])
        config["model"] = ModelConfig(**config["model"])
        return cls(**config)
