from nlpaug.util.exception.exception_info import ExceptionInfo, ExceptionType


class WarningException(ExceptionInfo):
    def __init__(self, name, code, msg):
        super(WarningException, self).__init__(name=name, exp_type=ExceptionType.WARNING, code=code, msg=msg)


class WarningName:
    INPUT_VALIDATION_WARNING = 'Input validation issue'
    OUT_OF_VOCABULARY = 'Out of vocabulary issue'


class WarningCode:
    WARNING_CODE_001 = 'W001'
    WARNING_CODE_002 = 'W002'


class WarningMessage:
    LENGTH_IS_ZERO = 'Length of input is 0'
    NO_WORD = 'No other word except stop words and OOV. Returning input data without augmentation'

    DEPRECATED = 'Warning: {} will be removed after {} release. Change to use {}'
