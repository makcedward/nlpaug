from nlpaug.util.exception.exception import Exception, ExceptionType


class Warning(Exception):
    def __init__(self, name, code, msg):
        super(Warning, self).__init__(name=name, type=ExceptionType.WARNING, code=code, msg=msg)


class WarningName:
    INPUT_VALIDATION_WARNING = 'Input validation issue'
    OUT_OF_VOCABULARY = 'Out of vocabulary issue'


class WarningCode:
    WARNING_CODE_001 = 'W001'
    WARNING_CODE_002 = 'W002'

class WarningMessage:
    LENGTH_IS_ZERO = 'Length of input is 0'
    NO_WORD = 'No other word except stop words and OOV. Returning input data without augmentation'
