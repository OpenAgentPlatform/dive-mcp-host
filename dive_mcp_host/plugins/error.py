class PluginError(Exception):
    """Plugin error."""


class PluginAlreadyRegisteredError(PluginError):
    """Plugin already registered."""


class PluginHookNameAlreadyRegisteredError(PluginError):
    """Hook name already registered."""


class PluginHookNotFoundError(PluginError):
    """Hook not found."""


class PluginLoadError(PluginError):
    """Plugin load error."""
