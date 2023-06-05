import math
import numpy


op_types = {
    'StartOperation': 0,
    'EndOperation': 1,
    'MixingOperation': 2,
    'HeatingOperation': 3,
    'CoolingOperation': 4,
    'GrindingOperation': 5,
    'SinteringOperation': 6,
    'PressingOperation': 7
}

op_conds = {
    'HeatingOperation': {
        'temperature': 'num',
        'time': 'num'
    },
    'CoolingOperation': {
        'method': ['slow', 'rapid']
    },
    'GrindingOperation': {
        'method': ['mixer-mill', 'ball-mill', 'wet-ball-mill', 'HEBM', 'hand-grinding', 'cryomilling']
    },
    'SinteringOperation': {
        'method': ['SPS', 'HP', 'HPHT', 'DCS', 'NaN', 'PECS', 'PAS', 'DC-HP'],
        # 'pressure': 'num'
    },
    'PressingOperation': {
        'pressure': 'num'
    }
}


def str_to_float(string, log=False):
    val = float(string)

    if not math.isnan(val):
        if log:
            return numpy.log(val)
        else:
            return val
    else:
        return None


class Operation:
    def __init__(self, op_name, op_type, conditions, label):
        self.op_name = op_name
        self.op_type = op_type
        self.conditions = conditions
        self.label = label
        self.label_cond = None

        if self.op_type == 'HeatingOperation':
            self.label_cond = self.__get_cond_dict(self.conditions.split(' '))
        elif self.op_type == 'CoolingOperation':
            self.label_cond = self.__get_cond_dict(self.conditions)
        elif self.op_type == 'GrindingOperation':
            self.label_cond = self.__get_cond_dict(self.conditions)
        elif self.op_type == 'SinteringOperation':
            self.label_cond = self.__get_cond_dict(self.conditions.split(' ')[0])
        elif self.op_type == 'PressingOperation':
            self.label_cond = self.__get_cond_dict(self.conditions)
        else:
            self.label_cond = None

    def __get_cond_dict(self, cond_vals):
        _cond_vals = cond_vals if isinstance(cond_vals, list) else [cond_vals]
        conds = list(op_conds[self.op_type].keys())
        dict_conds = dict()

        for i in range(0, len(conds)):
            if isinstance(op_conds[self.op_type][conds[i]], str):
                dict_conds[conds[i]] = str_to_float(_cond_vals[i], log=True)
            else:
                for j in range(0, len(op_conds[self.op_type][conds[i]])):
                    if _cond_vals[i] == op_conds[self.op_type][conds[i]][j]:
                        dict_conds[conds[i]] = j

        if len(dict_conds.keys()) == 0:
            dict_conds = None

        return dict_conds
