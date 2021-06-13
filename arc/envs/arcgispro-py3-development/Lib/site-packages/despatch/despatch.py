# -*- coding: utf-8 -*-

"""Main module."""

from os import environ
from x86cpu import info

# only looking at SSE2+, using this on x64 where this is guaranteed
INSTRUCTIONS = ("sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2")
# dispatch levels actually supported by MKL
ISA_MKL = ("sss3", "sse4_2", "avx", "avx2")


class UnhandledInstructionSet(NotImplementedError):
    def __init__(self, instruction, errors):
        self.message = "Don't know how to handle instruction set {}".format(instruction)
        super().__init__(self.message)
        self.errors = errors


def set_env(var, value):
    environ[var] = str(value)


def highest_supported():
    """Return the highest supported instruction set for this CPU."""
    # Note that x86cpu doesn't currently support AVX512, but we will leave the
    # dispatcher to its default path if above AVX2

    # this descrepancy should be fixed upstream
    prefix = ["has"] * 5 + ["supports"] * 2
    for (pos, isa) in enumerate(INSTRUCTIONS):
        info_attr = "_".join((prefix[pos], isa))
        if not getattr(info, info_attr):
            return INSTRUCTIONS[pos - 1]
    return isa


def supported_idx(instruction):
    """Based on position in the ISA list (won't work for AVX512)"""
    if instruction not in INSTRUCTIONS:
        raise UnhandledInstructionSet(instruction)

    return INSTRUCTIONS.index(instruction)


def is_supported(instruction):
    """Based on position in the ISA list (won't work for AVX512)"""
    if instruction not in INSTRUCTIONS:
        raise UnhandledInstructionSet(instruction)

    return supported_idx(instruction) >= INSTRUCTIONS.index(highest_supported())


def mkl_map_to_type(isa):
    """Map an ISA to a particular underlying MKL DLL. Note that these would be
       better specified with the supported MKL_ENABLE_INSTRUCTIONS variable,
       but this has no effect for non-Intel CPUs."""
    # map instruction set level to MKL_DEBUG_CPU_TYPE
    mkl_mapping = {"sse2": 1, "ssse3": 2, "sse4_2": 3, "avx": 4, "avx2": 5}
    # less than SSE4.2, use mkl_mc
    if isa not in mkl_mapping.keys():
        return 2
    else:
        return mkl_mapping[isa]


def set_cpu_type(max_isa):
    """Set the CPU dispatcher to the level specified by max_isa."""

    # undocumented environment variable for forcing dispatch to this level isa
    env_var = "MKL_DEBUG_CPU_TYPE"
    cpu_type = mkl_map_to_type(max_isa)
    if cpu_type not in range(2, 6):
        raise NotImplementedError("Only can set for SSE2 -> AVX2")

    set_env(env_var, cpu_type)


def set_optimal(verbose=False):
    """ Chose an optimal dispatch mechanism for MKL. This currently only
        will modify the MKL automatically determined dispatcher when
        either an AMD CPU is detected, or when the maximum instruction set
        supported by an Intel CPU is detected as SSE2, SSE3 or SSSE3, when it will
        use the mkl_mc.dll dispatcher, or the SSSE3 optimized code."""

    max_isa = highest_supported()
    if verbose:
        set_env("MKL_VERBOSE", 1)

    # if less than or equal to this isa, we want to modify
    if info.vendor == "AuthenticAMD" or supported_idx(max_isa) <= supported_idx("ssse3"):
        set_cpu_type(max_isa)
