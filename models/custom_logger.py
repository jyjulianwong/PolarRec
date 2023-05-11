"""
Methods for unifying and cleaning up log messages for server monitoring.
"""


def log(message, source_name, message_type="info"):
    """
    Logs the message by printing it onto the server console.

    :param message: The message to be logged.
    :type message: str
    :param source_name: The module or class this was called from.
    :type source_name: str
    :param message_type: ``"error"``, ``"warning"``, ``"info"``
    :type message_type: str
    """
    assert message_type in ["error", "warning", "info"]
    print(f"{message_type.upper()}: {source_name}: {message}")


def log_extended_line(message):
    """
    Assuming log(...) was called before this is called.
    Extends a pre-existing log message to become a multi-line message.

    :param message: The message to be logged.
    :type message: str
    """
    print(f"\t{message}")
