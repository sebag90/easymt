class InvalidArgument(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class FileError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
