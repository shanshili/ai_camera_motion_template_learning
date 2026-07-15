"""触发各阶段注册（import 即 @register 生效）。"""
from . import probe      # noqa: F401
from . import analysis   # noqa: F401
from . import shotplan   # noqa: F401
from . import camera     # noqa: F401
from . import render     # noqa: F401
