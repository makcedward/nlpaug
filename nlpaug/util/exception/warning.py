from nlpaug.util.exception.exception import Exception, ExceptionType


class Warning(Exception):
    def __init__(self, name, code, msg):
        super(Warning, self).__init__(name=name, type=ExceptionType.WARNING, code=code, msg=msg)


class WarningName:
    INPUT_VALIDATION_WARNING = 'Input validation exception'


class WarningCode:
    WARNING_CODE_001 = 'W001'

class WarningMessage:
    LENGTH_IS_ZERO = 'Length of input is 0'