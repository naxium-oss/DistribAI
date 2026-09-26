"""gridlink: expose the DistribAI grid and let anyone join it, for free.

Two halves of one thing:

* **expose**: an operator runs one command to publish the coordinator's control
  plane through a free Cloudflare quick tunnel (no account, no custom domain) or
  ngrok (free account). It prints a shareable **join link**.
* **join**: a contributor pastes that link (or a host:port) into any machine,
  a local GPU, Colab, Kaggle or Molab, and gets a ready-to-run worker.

Local bind stays the default: nothing is exposed unless a provider is asked for.
"""

# NB: exported as ``expose_grid`` (not ``expose``) so the ``gridlink.expose``
# submodule stays addressable for monkeypatching in tests.
from .expose import ExposeResult, ExposeSession
from .expose import expose as expose_grid
from .join import JoinPlan, plan_join
from .link import GridLink
from .providers import PROVIDERS, available_providers, get_provider

__all__ = [
    "PROVIDERS",
    "ExposeResult",
    "ExposeSession",
    "GridLink",
    "JoinPlan",
    "available_providers",
    "expose_grid",
    "get_provider",
    "plan_join",
]
