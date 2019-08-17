import functools
import warnings


def deprecated(deprecate_from, deprecate_to, msg):
    def decorator(obj):
        if isinstance(obj, type):
            return _decorate_class(obj, deprecate_from, deprecate_to, msg)
        # # TODO:
        # elif isinstance(obj, property):
        #     return _decorate_prop(obj, msg)
        else:
            return _decorate_func(obj, deprecate_from, deprecate_to, msg)
    return decorator


def _decorate_class(cls, deprecate_from, deprecate_to, msg):
    msg_template = 'Class {name} is deprecated from {deprecate_from} version.'
    msg_template += ' It will be removed from {deprecate_to} version. {msg}'

    @functools.wraps(cls)
    def wrapped(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            msg_template.format(
                name=cls.__name__, deprecate_from=deprecate_from, deprecate_to=deprecate_to, msg=msg),
            category=DeprecationWarning
        )
        warnings.simplefilter('default', DeprecationWarning)
        return cls(*args, **kwargs)

    return wrapped


def _decorate_func(func, deprecate_from, deprecate_to, msg):
    msg_template = 'Function {name} is deprecated from {deprecate_from} version.'
    msg_template += ' It will be removed from {deprecate_to} version. {msg}'

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            msg_template.format(
                name=func.__name__, deprecate_from=deprecate_from, deprecate_to=deprecate_to, msg=msg),
            category=DeprecationWarning
        )
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)

    return wrapped


def _decorate_prop(prop, msg):
    @functools.wraps(prop)
    @property
    def wrapped(*args, **kwargs):
        msg_template = 'Property {name} is deprecated. {msg}'
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            msg_template.format(name=prop.__name__, msg=msg), category=DeprecationWarning
        )
        warnings.simplefilter('default', DeprecationWarning)
        return prop.fget(*args, **kwargs)

    return wrapped
