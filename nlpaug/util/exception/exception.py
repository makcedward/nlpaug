
class Exception:
    def __init__(self, name, type, code, msg):
        self.name = name
        self.type = type
        self.code = code
        self.msg = msg

    def output(self):
        msg = '[{}] Name:{}, Code:{}, Message:{}'.format(self.type, self.name, self.code, self.msg)
        print(msg)


class ExceptionType:
    WARNING = 'Warning'