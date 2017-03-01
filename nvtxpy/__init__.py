"""This submodule adds support for nvtx if the lib is available.

This allows integration with nvidia tools like nvvp (the visual
profiler) by providing tools to install markers and ranges in
a pythonic way.
"""

from __future__ import absolute_import, print_function, division

from .nvtx import (profile_range, 
                   profile_range_push, 
                   profile_range_pop, 
                   profile_mark,
                   profiled,
                   getstats,
                   colors)
