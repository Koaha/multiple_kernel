import inspect
def instance_convert(X):
    if inspect.isclass(X):
        return X()
    return X
