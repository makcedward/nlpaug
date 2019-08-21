
class ExceptionInfo:
    def __init__(self, name, exp_type, code, msg):
        self.name = name
        self.exp_type = exp_type
        self.code = code
        self.msg = msg

    def output(self):
        msg = '[{}] Name:{}, Code:{}, Message:{}'.format(self.exp_type, self.name, self.code, self.msg)
        print(msg)


class ExceptionType:
    WARNING = 'Warning'