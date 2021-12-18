import configparser

from utils.errors import InvalidArgument


class Section:
    """
    Class to store parameters from a section
    of the configuration file
    """
    def __init__(self, section):
        for key, value in section.items():
            try:
                setattr(self, key, eval(value))
            except NameError:
                setattr(self, key, value)

    def __repr__(self):
        return f"Section({', '.join(vars(self).keys())})"


class Parameters:
    """
    Class to store parameters from configuration file
    """
    def __init__(self, config):
        for key, value in config.items():
            if len(dict(value)) > 0:
                setattr(self, key.lower(), Section(dict(value)))

    def __repr__(self):
        return f"Parameters({', '.join(vars(self).keys())})"

    @classmethod
    def from_config(cls, path):
        # argument parser
        config = configparser.ConfigParser()
        config.read(path)

        # create empty class to store parameters
        params = cls(config)

        if (params.rnn.input_feed
                and params.rnn.attention.lower() == "none"):
            raise InvalidArgument(
                "Input feed needs attention"
            )

        return params
