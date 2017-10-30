"""Useful exceptions."""


class InvalidArchitectureException(ValueError):
    """Exception used when architecture value is not correct."""

    def __init__(self, architecture):
        """Define the message displayed to the user."""
        self.supported_architectures = ['inception', 'vgg16', 'vgg19', 'resnet']
        super(InvalidArchitectureException, self).__init__('Invalid value for architectute. Got {0} but needs to be one of {1}'.format(architecture, self.supported_architectures))
