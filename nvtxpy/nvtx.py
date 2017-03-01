"""
NVTX driver implementation

"""

from __future__ import absolute_import, print_function, division

import os
import sys
from time import time
from copy import deepcopy
import contextlib
import ctypes
import numbers

def _get_cuda_nvtx_path():
    base = os.environ.get('NVTXPY_CUDA_TOOLKIT')

    # if no environment variable is set, try a sensible default.
    # In our case, the default install path for CUDA
    # TODO: implement something more reliable that auto updates
    if base is None:
        if sys.platform == 'darwin':
            base = '/Developer/NVIDIA/CUDA-7.5/lib'
        elif sys.platform.startswith('linux'):
            base = '/usr/local/cuda-7.0/lib64'
        else:
            # no windows default
            base = None

    if sys.platform == 'darwin':
        ext = '.dylib'
    elif sys.platform.startswith('linux'):
        ext = '.so'
    elif sys.platform == 'win32':
        ext = '.dll'
    else:
        ext = ''
            
    return os.path.join(base, 'libnvToolsExt'+ext) if base is not None else None
        

def _load_lib():
    loader = ctypes.CDLL if sys.platform != 'win32' else ctypes.WinDLL

    try:
        return loader(_get_cuda_nvtx_path())
    except OSError as e:
        # placeholder for more informative error... just reraise for now
        raise


try:
    _lib = _load_lib()

    # map nvtxMarkA, nvtxRangePushA and nvtxRangePop
    _NVTX_VERSION = 1

    _NVTX_COLOR_UNKNOWN = 0
    _NVTX_COLOR_ARGB = 1

    _NVTX_PAYLOAD_UNKNOWN = 0
    _NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1
    _NVTX_PAYLOAD_TYPE_INT64 = 2
    _NVTX_PAYLOAD_TYPE_DOUBLE = 3

    _NVTX_MESSAGE_UNKNOWN = 0
    _NVTX_MESSAGE_TYPE_ASCII = 1
    _NVTX_MESSAGE_TYPE_UNICODE = 2

    class _Payload(ctypes.Union):
        _fields_ = [('ull_value', ctypes.c_uint64),
                    ('ll_value', ctypes.c_int64),
                    ('d_value', ctypes.c_double)]

    class _Message(ctypes.Union):
        _fields_ = [('ascii', ctypes.c_char_p),
                   ('unicode', ctypes.c_wchar_p)]

    class _EventAttributes_v1(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint16),
                    ('size', ctypes.c_uint16),
                    ('category', ctypes.c_uint32),
                    ('_color_type', ctypes.c_int32),
                    ('_color', ctypes.c_uint32),
                    ('_payload_type', ctypes.c_int32),
                    ('_reserved0', ctypes.c_int32),
                    ('_payload', _Payload),
                    ('_message_type', ctypes.c_int32),
                    ('_message', _Message)]

        @property
        def payload(self):
            if self._payload_type == _NVTX_PAYLOAD_UNKNOWN:
                return None
            elif self._payload_type ==  _NVTX_PAYLOAD_TYPE_UNSIGNED_INT64:
                return self._payload.ull_value
            elif self._payload_type == _NVTX_PAYLOAD_TYPE_INT64:
                return self._payload.ll_value
            elif self._payload_type == _NVTX_PAYLOAD_TYPE_DOUBLE:
                return self._payload.d_value
            else:
                raise AttributeError('Payload of unknown type')

        @property
        def color(self):
            if self._color_type == _NVTX_COLOR_UNKNOWN:
                return None
            elif self._color_type == _NVTX_COLOR_ARGB:
                return hex(self._color)
            else:
                raise AttributeError('Color of unknown type')

        @property
        def message(self):
            if self._message_type == _NVTX_MESSAGE_UNKNOWN:
                return None
            elif self._message_type == _NVTX_MESSAGE_TYPE_ASCII:
                return self._message.ascii
            elif self._message_type ==  _NVTX_MESSAGE_TYPE_UNICODE:
                return self._message.unicode
            else:
                raise AttributeError('Message of unknown type')

        def __init__(self):
            ctypes.memset(ctypes.byref(self), 0, ctypes.sizeof(self))
            self.version = _NVTX_VERSION
            self.size = ctypes.sizeof(self)

        def __str__(self):
            attrs = ['version', 'size', 'category', 'message',
                     'color', 'payload']

            str = '\n\t'.join('{0}: {1}'.format(x, getattr(self, x)) for x in attrs)
            return 'EventAttributes_v1:\n\t'+str

    _EventAttr_p = ctypes.POINTER(_EventAttributes_v1)


    _lib.nvtxMarkA.restype = None
    _lib.nvtxMarkA.argtypes = [ ctypes.c_char_p ]
    _lib.nvtxMarkEx.restype = None
    _lib.nvtxMarkEx.argtypes = [_EventAttr_p]

    _lib.nvtxRangePushEx.restype = ctypes.c_int
    _lib.nvtxRangePushEx.argtypes = [_EventAttr_p]
    _lib.nvtxRangePushA.restype = ctypes.c_int
    _lib.nvtxRangePushA.argtypes = [ ctypes.c_char_p ]
    _lib.nvtxRangePop.restype = ctypes.c_int
    _lib.nvtxRangePop.argtypes = []

    def _create_event(message, color, payload, category):
        ea = _EventAttributes_v1()

        if category is not None:
            ea.category = category

        if color is not None:
            ea._color_type = _NVTX_COLOR_ARGB
            ea._color = color

        if payload is not None:
            if isinstance(payload, numbers.Integral):
                ea._payload_type = _NVTX_PAYLOAD_TYPE_INT64
                ea._payload.ll_value = payload
            elif isinstance(payload, numbers.Real):
                ea._payload_type = _NVTX_PAYLOAD_TYPE_DOUBLE
                ea._payload.d_value = payload

        if message is not None:
            ea._message_type = _NVTX_MESSAGE_TYPE_ASCII
            ea._message.ascii = message

        return ea


    # everything is ok... export the symbols
    # the simple ascii version will be used when no color payload or category is
    # specified to avoid overhead... otherwise the event_attribute version of
    # the calls will be used
    def profile_mark(message, color=None, payload=None, category=None):
        if all(x is None for x in (color, payload, category)):
            _lib.nvtxMarkA(message)
        else:
            evnt = _create_event(message, color, payload, category)
            _lib.nvtxMarkEx(ctypes.byref(evnt))


    def profile_range_push(message, color=None, payload=None, category=None):
        if all(x is None for x in (color, payload, category)):
            _lib.nvtxRangePushA(message)
        else:
            evnt = _create_event(message, color, payload, category)
            _lib.nvtxRangePushEx(ctypes.byref(evnt))


    profile_range_pop = _lib.nvtxRangePop

except Exception as e:
    def profile_mark(message, color=None, payload=None, category=None):
        pass

    def profile_range_push(message, color=None, payload=None, category=None):
        pass

    def profile_range_pop():
        pass


class _Colors(object):
    red = 0xffff0000
    green = 0xff00ff00
    blue = 0xff0000ff
    yellow = 0xffffff00
    magenta = 0xffff00ff
    cyan = 0xff00ffff
    white = 0xffffffff
    black = 0xff000000


colors = _Colors()

_stats = {}

@contextlib.contextmanager
def profile_range_nvtx_only(name, color=None, payload=None, category=None):
    profile_range_push(name, color=color, payload=payload, category=None)
    yield
    profile_range_pop()

@contextlib.contextmanager
def profile_range(name, color=None, payload=None, category=None):
    profile_range_push(name, color=color, payload=payload, category=None)
    t = time()
    yield
    t = time() - t

    stat = _stats.get(name)
    if stat:
        stat[0] += 1
        stat[1] += t
    else:
        _stats[name] = [1, t]
    profile_range_pop()


def profiled(tag, category=None, color=None, payload=None):
    from functools import wraps
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            with profile_range(tag, category=category, color=color, payload=payload):
                return func(*args, **kwargs)
        return _wrapper
    return _decorator


def getstats():
    return deepcopy(_stats)
