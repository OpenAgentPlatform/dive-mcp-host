from enum import StrEnum


class ClientState(StrEnum):
    """The state of the client.

    States and transitions:
    """

    INIT = "init"
    RUNNING = "running"
    CLOSED = "closed"
    RESTARTING = "restarting"
    FAILED = "failed"


type ChatID = str
