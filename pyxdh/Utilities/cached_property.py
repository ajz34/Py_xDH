# https://ajz34.readthedocs.io/zh_CN/latest/Simple_Notes/cached_property.html

def cached_property(f):
    def wrap(*args):
        self = args[0]
        _f = "_" + f.__name__
        if not hasattr(self, _f) or getattr(self, _f) is NotImplemented:
            setattr(self, _f, f(*args))
        return getattr(self, _f)
    return property(wrap)
