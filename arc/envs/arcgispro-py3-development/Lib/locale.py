"""Locale support module.

The module provides low-level access to the C lib's locale APIs and adds high
level number formatting APIs as well as a locale aliasing engine to complement
these.

The aliasing engine includes support for many commonly used locale names and
maps them to values suitable for passing to the C lib's setlocale() function. It
also includes default encodings for all supported locale names.

"""

import sys
import encodings
import encodings.aliases
import copy
import re
import _collections_abc
from builtins import str as _builtin_str
import functools

# Try importing the _locale module.
#
# If this fails, fall back on a basic 'C' locale emulation.

# Yuck:  LC_MESSAGES is non-standard:  can't tell whether it exists before
# trying the import.  So __all__ is also fiddled at the end of the file.
__all__ = ["getlocale", "getdefaultlocale", "getpreferredencoding", "Error",
           "setlocale", "resetlocale", "localeconv", "strcoll", "strxfrm",
           "str", "atof", "atoi", "format", "format_string", "currency",
           "normalize", "LC_CTYPE", "LC_COLLATE", "LC_TIME", "LC_MONETARY",
           "LC_NUMERIC", "LC_ALL", "CHAR_MAX"]

def _strcoll(a,b):
    """ strcoll(string,string) -> int.
        Compares two strings according to the locale.
    """
    return (a > b) - (a < b)

def _strxfrm(s):
    """ strxfrm(string) -> string.
        Returns a string that behaves for cmp locale-aware.
    """
    return s

try:

    from _locale import *

except ImportError:

    # Locale emulation

    CHAR_MAX = 127
    LC_ALL = 6
    LC_COLLATE = 3
    LC_CTYPE = 0
    LC_MESSAGES = 5
    LC_MONETARY = 4
    LC_NUMERIC = 1
    LC_TIME = 2
    Error = ValueError

    def localeconv():
        """ localeconv() -> dict.
            Returns numeric and monetary locale-specific parameters.
        """
        # 'C' locale default values
        return {'grouping': [127],
                'currency_symbol': '',
                'n_sign_posn': 127,
                'p_cs_precedes': 127,
                'n_cs_precedes': 127,
                'mon_grouping': [],
                'n_sep_by_space': 127,
                'decimal_point': '.',
                'negative_sign': '',
                'positive_sign': '',
                'p_sep_by_space': 127,
                'int_curr_symbol': '',
                'p_sign_posn': 127,
                'thousands_sep': '',
                'mon_thousands_sep': '',
                'frac_digits': 127,
                'mon_decimal_point': '',
                'int_frac_digits': 127}

    def setlocale(category, value=None):
        """ setlocale(integer,string=None) -> string.
            Activates/queries locale processing.
        """
        if value not in (None, '', 'C'):
            raise Error('_locale emulation only supports "C" locale')
        return 'C'

# These may or may not exist in _locale, so be sure to set them.
if 'strxfrm' not in globals():
    strxfrm = _strxfrm
if 'strcoll' not in globals():
    strcoll = _strcoll


_localeconv = localeconv

# With this dict, you can override some items of localeconv's return value.
# This is useful for testing purposes.
_override_localeconv = {}

@functools.wraps(_localeconv)
def localeconv():
    d = _localeconv()
    if _override_localeconv:
        d.update(_override_localeconv)
    return d


### Number formatting APIs

# Author: Martin von Loewis
# improved by Georg Brandl

# Iterate over grouping intervals
def _grouping_intervals(grouping):
    last_interval = None
    for interval in grouping:
        # if grouping is -1, we are done
        if interval == CHAR_MAX:
            return
        # 0: re-use last group ad infinitum
        if interval == 0:
            if last_interval is None:
                raise ValueError("invalid grouping")
            while True:
                yield last_interval
        yield interval
        last_interval = interval

#perform the grouping from right to left
def _group(s, monetary=False):
    conv = localeconv()
    thousands_sep = conv[monetary and 'mon_thousands_sep' or 'thousands_sep']
    grouping = conv[monetary and 'mon_grouping' or 'grouping']
    if not grouping:
        return (s, 0)
    if s[-1] == ' ':
        stripped = s.rstrip()
        right_spaces = s[len(stripped):]
        s = stripped
    else:
        right_spaces = ''
    left_spaces = ''
    groups = []
    for interval in _grouping_intervals(grouping):
        if not s or s[-1] not in "0123456789":
            # only non-digit characters remain (sign, spaces)
            left_spaces = s
            s = ''
            break
        groups.append(s[-interval:])
        s = s[:-interval]
    if s:
        groups.append(s)
    groups.reverse()
    return (
        left_spaces + thousands_sep.join(groups) + right_spaces,
        len(thousands_sep) * (len(groups) - 1)
    )

# Strip a given amount of excess padding from the given string
def _strip_padding(s, amount):
    lpos = 0
    while amount and s[lpos] == ' ':
        lpos += 1
        amount -= 1
    rpos = len(s) - 1
    while amount and s[rpos] == ' ':
        rpos -= 1
        amount -= 1
    return s[lpos:rpos+1]

_percent_re = re.compile(r'%(?:\((?P<key>.*?)\))?'
                         r'(?P<modifiers>[-#0-9 +*.hlL]*?)[eEfFgGdiouxXcrs%]')

def _format(percent, value, grouping=False, monetary=False, *additional):
    if additional:
        formatted = percent % ((value,) + additional)
    else:
        formatted = percent % value
    # floats and decimal ints need special action!
    if percent[-1] in 'eEfFgG':
        seps = 0
        parts = formatted.split('.')
        if grouping:
            parts[0], seps = _group(parts[0], monetary=monetary)
        decimal_point = localeconv()[monetary and 'mon_decimal_point'
                                              or 'decimal_point']
        formatted = decimal_point.join(parts)
        if seps:
            formatted = _strip_padding(formatted, seps)
    elif percent[-1] in 'diu':
        seps = 0
        if grouping:
            formatted, seps = _group(formatted, monetary=monetary)
        if seps:
            formatted = _strip_padding(formatted, seps)
    return formatted

def format_string(f, val, grouping=False, monetary=False):
    """Formats a string in the same way that the % formatting would use,
    but takes the current locale into account.

    Grouping is applied if the third parameter is true.
    Conversion uses monetary thousands separator and grouping strings if
    forth parameter monetary is true."""
    percents = list(_percent_re.finditer(f))
    new_f = _percent_re.sub('%s', f)

    if isinstance(val, _collections_abc.Mapping):
        new_val = []
        for perc in percents:
            if perc.group()[-1]=='%':
                new_val.append('%')
            else:
                new_val.append(_format(perc.group(), val, grouping, monetary))
    else:
        if not isinstance(val, tuple):
            val = (val,)
        new_val = []
        i = 0
        for perc in percents:
            if perc.group()[-1]=='%':
                new_val.append('%')
            else:
                starcount = perc.group('modifiers').count('*')
                new_val.append(_format(perc.group(),
                                      val[i],
                                      grouping,
                                      monetary,
                                      *val[i+1:i+1+starcount]))
                i += (1 + starcount)
    val = tuple(new_val)

    return new_f % val

def format(percent, value, grouping=False, monetary=False, *additional):
    """Deprecated, use format_string instead."""
    import warnings
    warnings.warn(
        "This method will be removed in a future version of Python. "
        "Use 'locale.format_string()' instead.",
        DeprecationWarning, stacklevel=2
    )

    match = _percent_re.match(percent)
    if not match or len(match.group())!= len(percent):
        raise ValueError(("format() must be given exactly one %%char "
                         "format specifier, %s not valid") % repr(percent))
    return _format(percent, value, grouping, monetary, *additional)

def currency(val, symbol=True, grouping=False, international=False):
    """Formats val according to the currency settings
    in the current locale."""
    conv = localeconv()

    # check for illegal values
    digits = conv[international and 'int_frac_digits' or 'frac_digits']
    if digits == 127:
        raise ValueError("Currency formatting is not possible using "
                         "the 'C' locale.")

    s = _format('%%.%if' % digits, abs(val), grouping, monetary=True)
    # '<' and '>' are markers if the sign must be inserted between symbol and value
    s = '<' + s + '>'

    if symbol:
        smb = conv[international and 'int_curr_symbol' or 'currency_symbol']
        precedes = conv[val<0 and 'n_cs_precedes' or 'p_cs_precedes']
        separated = conv[val<0 and 'n_sep_by_space' or 'p_sep_by_space']

        if precedes:
            s = smb + (separated and ' ' or '') + s
        else:
            s = s + (separated and ' ' or '') + smb

    sign_pos = conv[val<0 and 'n_sign_posn' or 'p_sign_posn']
    sign = conv[val<0 and 'negative_sign' or 'positive_sign']

    if sign_pos == 0:
        s = '(' + s + ')'
    elif sign_pos == 1:
        s = sign + s
    elif sign_pos == 2:
        s = s + sign
    elif sign_pos == 3:
        s = s.replace('<', sign)
    elif sign_pos == 4:
        s = s.replace('>', sign)
    else:
        # the default if nothing specified;
        # this should be the most fitting sign position
        s = sign + s

    return s.replace('<', '').replace('>', '')

def str(val):
    """Convert float to string, taking the locale into account."""
    return _format("%.12g", val)

def delocalize(string):
    "Parses a string as a normalized number according to the locale settings."

    conv = localeconv()

    #First, get rid of the grouping
    ts = conv['thousands_sep']
    if ts:
        string = string.replace(ts, '')

    #next, replace the decimal point with a dot
    dd = conv['decimal_point']
    if dd:
        string = string.replace(dd, '.')
    return string

def atof(string, func=float):
    "Parses a string as a float according to the locale settings."
    return func(delocalize(string))

def atoi(string):
    "Converts a string to an integer according to the locale settings."
    return int(delocalize(string))

def _test():
    setlocale(LC_ALL, "")
    #do grouping
    s1 = format_string("%d", 123456789,1)
    print(s1, "is", atoi(s1))
    #standard formatting
    s1 = str(3.14)
    print(s1, "is", atof(s1))

### Locale name aliasing engine

# Author: Marc-Andre Lemburg, mal@lemburg.com
# Various tweaks by Fredrik Lundh <fredrik@pythonware.com>

# store away the low-level version of setlocale (it's
# overridden below)
_setlocale = setlocale

def _replace_encoding(code, encoding):
    if '.' in code:
        langname = code[:code.index('.')]
    else:
        langname = code
    # Convert the encoding to a C lib compatible encoding string
    norm_encoding = encodings.normalize_encoding(encoding)
    #print('norm encoding: %r' % norm_encoding)
    norm_encoding = encodings.aliases.aliases.get(norm_encoding.lower(),
                                                  norm_encoding)
    #print('aliased encoding: %r' % norm_encoding)
    encoding = norm_encoding
    norm_encoding = norm_encoding.lower()
    if norm_encoding in locale_encoding_alias:
        encoding = locale_encoding_alias[norm_encoding]
    else:
        norm_encoding = norm_encoding.replace('_', '')
        norm_encoding = norm_encoding.replace('-', '')
        if norm_encoding in locale_encoding_alias:
            encoding = locale_encoding_alias[norm_encoding]
    #print('found encoding %r' % encoding)
    return langname + '.' + encoding

def _append_modifier(code, modifier):
    if modifier == 'euro':
        if '.' not in code:
            return code + '.ISO8859-15'
        _, _, encoding = code.partition('.')
        if encoding in ('ISO8859-15', 'UTF-8'):
            return code
        if encoding == 'ISO8859-1':
            return _replace_encoding(code, 'ISO8859-15')
    return code + '@' + modifier

def normalize(localename):

    """ Returns a normalized locale code for the given locale
        name.

        The returned locale code is formatted for use with
        setlocale().

        If normalization fails, the original name is returned
        unchanged.

        If the given encoding is not known, the function defaults to
        the default encoding for the locale code just like setlocale()
        does.

    """
    # Normalize the locale name and extract the encoding and modifier
    code = localename.lower()

    # if we're on windows, try win names first.
    if sys.platform.startswith("win"):
        # skip already fully qualified entries
        if localename in win_locale_alias.values():
            return localename

        win_code = win_locale_alias.get(code, None)
        if win_code:
            return win_code

    if ':' in code:
        # ':' is sometimes used as encoding delimiter.
        code = code.replace(':', '.')
    if '@' in code:
        code, modifier = code.split('@', 1)
    else:
        modifier = ''
    if '.' in code:
        langname, encoding = code.split('.')[:2]
    else:
        langname = code
        encoding = ''

    # First lookup: fullname (possibly with encoding and modifier)
    lang_enc = langname
    if encoding:
        norm_encoding = encoding.replace('-', '')
        norm_encoding = norm_encoding.replace('_', '')
        lang_enc += '.' + norm_encoding
    lookup_name = lang_enc
    if modifier:
        lookup_name += '@' + modifier
    code = locale_alias.get(lookup_name, None)
    if code is not None:
        return code
    #print('first lookup failed')

    if modifier:
        # Second try: fullname without modifier (possibly with encoding)
        code = locale_alias.get(lang_enc, None)
        if code is not None:
            #print('lookup without modifier succeeded')
            if '@' not in code:
                return _append_modifier(code, modifier)
            if code.split('@', 1)[1].lower() == modifier:
                return code
        #print('second lookup failed')

    if encoding:
        # Third try: langname (without encoding, possibly with modifier)
        lookup_name = langname
        if modifier:
            lookup_name += '@' + modifier
        code = locale_alias.get(lookup_name, None)
        if code is not None:
            #print('lookup without encoding succeeded')
            if '@' not in code:
                return _replace_encoding(code, encoding)
            code, modifier = code.split('@', 1)
            return _replace_encoding(code, encoding) + '@' + modifier

        if modifier:
            # Fourth try: langname (without encoding and modifier)
            code = locale_alias.get(langname, None)
            if code is not None:
                #print('lookup without modifier and encoding succeeded')
                if '@' not in code:
                    code = _replace_encoding(code, encoding)
                    return _append_modifier(code, modifier)
                code, defmod = code.split('@', 1)
                if defmod.lower() == modifier:
                    return _replace_encoding(code, encoding) + '@' + defmod

    return localename

def _parse_localename(localename):

    """ Parses the locale code for localename and returns the
        result as tuple (language code, encoding).

        The localename is normalized and passed through the locale
        alias engine. A ValueError is raised in case the locale name
        cannot be parsed.

        The language code corresponds to RFC 1766.  code and encoding
        can be None in case the values cannot be determined or are
        unknown to this implementation.

    """
    code = normalize(localename)
    if '@' in code:
        # Deal with locale modifiers
        code, modifier = code.split('@', 1)
        if modifier == 'euro' and '.' not in code:
            # Assume Latin-9 for @euro locales. This is bogus,
            # since some systems may use other encodings for these
            # locales. Also, we ignore other modifiers.
            return code, 'iso-8859-15'

    if '.' in code:
        return tuple(code.split('.')[:2])
    elif code == 'C':
        return None, None
    elif code == 'UTF-8':
        # On macOS "LC_CTYPE=UTF-8" is a valid locale setting
        # for getting UTF-8 handling for text.
        return None, 'UTF-8'
    raise ValueError('unknown locale: %s' % localename)

def _build_localename(localetuple):

    """ Builds a locale code from the given tuple (language code,
        encoding).

        No aliasing or normalizing takes place.

    """
    try:
        language, encoding = localetuple

        if language is None:
            language = 'C'
        if encoding is None:
            return language
        else:
            return language + '.' + encoding
    except (TypeError, ValueError):
        raise TypeError('Locale must be None, a string, or an iterable of '
                        'two strings -- language code, encoding.') from None

def getdefaultlocale(envvars=('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE')):

    """ Tries to determine the default locale settings and returns
        them as tuple (language code, encoding).

        According to POSIX, a program which has not called
        setlocale(LC_ALL, "") runs using the portable 'C' locale.
        Calling setlocale(LC_ALL, "") lets it use the default locale as
        defined by the LANG variable. Since we don't want to interfere
        with the current locale setting we thus emulate the behavior
        in the way described above.

        To maintain compatibility with other platforms, not only the
        LANG variable is tested, but a list of variables given as
        envvars parameter. The first found to be defined will be
        used. envvars defaults to the search path used in GNU gettext;
        it must always contain the variable name 'LANG'.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """

    try:
        # check if it's supported by the _locale module
        import _locale
        code, encoding = _locale._getdefaultlocale()
    except (ImportError, AttributeError):
        pass
    else:
        # make sure the code/encoding values are valid
        if sys.platform == "win32" and code and code[:2] == "0x":
            # map windows language identifier to language name
            code = windows_locale.get(int(code, 0))
        # ...add other platform-specific processing here, if
        # necessary...
        return code, encoding

    # fall back on POSIX behaviour
    import os
    lookup = os.environ.get
    for variable in envvars:
        localename = lookup(variable,None)
        if localename:
            if variable == 'LANGUAGE':
                localename = localename.split(':')[0]
            break
    else:
        localename = 'C'
    return _parse_localename(localename)


def getlocale(category=LC_CTYPE):

    """ Returns the current setting for the given locale category as
        tuple (language code, encoding).

        category may be one of the LC_* value except LC_ALL. It
        defaults to LC_CTYPE.

        Except for the code 'C', the language code corresponds to RFC
        1766.  code and encoding can be None in case the values cannot
        be determined.

    """
    localename = _setlocale(category)
    if category == LC_ALL and ';' in localename:
        raise TypeError('category LC_ALL is not supported')
    return _parse_localename(localename)

def setlocale(category, locale=None):

    """ Set the locale for the given category.  The locale can be
        a string, an iterable of two strings (language code and encoding),
        or None.

        Iterables are converted to strings using the locale aliasing
        engine.  Locale strings are passed directly to the C lib.

        category may be given as one of the LC_* values.

    """
    if locale and not isinstance(locale, _builtin_str):
        # convert to string
        locale = normalize(_build_localename(locale))

    # If on Windows and passed a POSIX style locale, try to convert it
    if sys.platform.startswith('win'):
        locale = _normalize_winlocalename(locale)

    return _setlocale(category, locale)

def _normalize_winlocalename(locale):
    """Build a normalized Windows locale name."""
    if not locale:
        locale = ''

    code = locale.lower()
    # lang-country and shortcode names, skip short lang and non-nls
    if not win_non_nls_locales.get(code, None) and len(code) != 3:
        win_code = win_locale_alias.get(code, None)
        if win_code:
            # still use modified form, return value will include full form
            locale = win_code
        else:
            # simple modifications to get basic POSIX names through
            code = code.replace('_', '-')
            code = code.split('.')[0]

            win_code = win_locale_alias.get(code, None)
            if win_code:
                locale = win_code
    return locale

def _build_winlocalename(localetuple):
    locale = _build_localename(localetuple)
    norm_locale = _normalize_winlocalename(locale)

    codepage = None
    if len(localetuple) == 2:
        codepage = _convert_getdefaultlocale_codepage(localetuple[1])
        try:
            parts = norm_locale.split(".")
            if len(parts) == 2:
                (norm_language, norm_codepage) = parts
                locale = norm_language + "." + codepage
        except (ValueError, TypeError):
            raise TypeError('Locale must be None, a string, or an iterable of two strings -- language code, encoding.')

    return locale

def _convert_getdefaultlocale_codepage(codepage):
    # Currently, this just unmangles the output from
    # PyLocale_getdefaultlocale, which always prepends
    # 'cp' to the values of GetACP(). In the future, it should
    # be replaced with a correct implementation and
    # _convert_codepage_to_posix().

    if codepage.startswith('cp'):
        codepage = codepage[2:]
    return codepage

def _convert_codepage_to_win(codepage):
    # convert POSIX style codepage into a windows value.
    for (win_page, posix_page) in win_codepages.items():
        if codepage == posix_page:
            codepage = win_page
    return codepage

def _convert_codepage_to_posix(codepage):
    # convert windows style codepage into a POSIX value.
    return win_codepages.get(codepage, codepage)

def resetlocale(category=LC_ALL):

    """ Sets the locale for category to the default setting.

        The default setting is determined by calling
        getdefaultlocale(). category defaults to LC_ALL.

    """
    default_locale = getdefaultlocale()
    if sys.platform.startswith("win"):
        # we can't directly use the returns from 'getdefaultlocale()'
        # because the lang_name + codepage combo isn't valid for
        # setlocale. Long term fix is to update the function, for now
        # set it to defaults we can use.
       locale = _build_winlocalename(default_locale)
    else:
        locale = _build_localename(default_locale)
    print("setting to {}".format(locale))
    _setlocale(category, locale)

if sys.platform.startswith("win"):
    # On Win32, this will return the ANSI code page
    def getpreferredencoding(do_setlocale = True):
        """Return the charset that the user is likely using."""
        if sys.flags.utf8_mode:
            return 'UTF-8'
        import _bootlocale
        return _bootlocale.getpreferredencoding(False)
else:
    # On Unix, if CODESET is available, use that.
    try:
        CODESET
    except NameError:
        if hasattr(sys, 'getandroidapilevel'):
            # On Android langinfo.h and CODESET are missing, and UTF-8 is
            # always used in mbstowcs() and wcstombs().
            def getpreferredencoding(do_setlocale = True):
                return 'UTF-8'
        else:
            # Fall back to parsing environment variables :-(
            def getpreferredencoding(do_setlocale = True):
                """Return the charset that the user is likely using,
                by looking at environment variables."""
                if sys.flags.utf8_mode:
                    return 'UTF-8'
                res = getdefaultlocale()[1]
                if res is None:
                    # LANG not set, default conservatively to ASCII
                    res = 'ascii'
                return res
    else:
        def getpreferredencoding(do_setlocale = True):
            """Return the charset that the user is likely using,
            according to the system configuration."""
            if sys.flags.utf8_mode:
                return 'UTF-8'
            import _bootlocale
            if do_setlocale:
                oldloc = setlocale(LC_CTYPE)
                try:
                    setlocale(LC_CTYPE, "")
                except Error:
                    pass
            result = _bootlocale.getpreferredencoding(False)
            if do_setlocale:
                setlocale(LC_CTYPE, oldloc)
            return result


### Database
#
# The following data was extracted from the locale.alias file which
# comes with X11 and then hand edited removing the explicit encoding
# definitions and adding some more aliases. The file is usually
# available as /usr/lib/X11/locale/locale.alias.
#

#
# The local_encoding_alias table maps lowercase encoding alias names
# to C locale encoding names (case-sensitive). Note that normalize()
# first looks up the encoding in the encodings.aliases dictionary and
# then applies this mapping to find the correct C lib name for the
# encoding.
#
locale_encoding_alias = {

    # Mappings for non-standard encoding names used in locale names
    '437':                          'C',
    'c':                            'C',
    'en':                           'ISO8859-1',
    'jis':                          'JIS7',
    'jis7':                         'JIS7',
    'ajec':                         'eucJP',
    'koi8c':                        'KOI8-C',
    'microsoftcp1251':              'CP1251',
    'microsoftcp1255':              'CP1255',
    'microsoftcp1256':              'CP1256',
    '88591':                        'ISO8859-1',
    '88592':                        'ISO8859-2',
    '88595':                        'ISO8859-5',
    '885915':                       'ISO8859-15',

    # Mappings from Python codec names to C lib encoding names
    'ascii':                        'ISO8859-1',
    'latin_1':                      'ISO8859-1',
    'iso8859_1':                    'ISO8859-1',
    'iso8859_10':                   'ISO8859-10',
    'iso8859_11':                   'ISO8859-11',
    'iso8859_13':                   'ISO8859-13',
    'iso8859_14':                   'ISO8859-14',
    'iso8859_15':                   'ISO8859-15',
    'iso8859_16':                   'ISO8859-16',
    'iso8859_2':                    'ISO8859-2',
    'iso8859_3':                    'ISO8859-3',
    'iso8859_4':                    'ISO8859-4',
    'iso8859_5':                    'ISO8859-5',
    'iso8859_6':                    'ISO8859-6',
    'iso8859_7':                    'ISO8859-7',
    'iso8859_8':                    'ISO8859-8',
    'iso8859_9':                    'ISO8859-9',
    'iso2022_jp':                   'JIS7',
    'shift_jis':                    'SJIS',
    'tactis':                       'TACTIS',
    'euc_jp':                       'eucJP',
    'euc_kr':                       'eucKR',
    'utf_8':                        'UTF-8',
    'koi8_r':                       'KOI8-R',
    'koi8_t':                       'KOI8-T',
    'koi8_u':                       'KOI8-U',
    'kz1048':                       'RK1048',
    'cp1251':                       'CP1251',
    'cp1255':                       'CP1255',
    'cp1256':                       'CP1256',

    # XXX This list is still incomplete. If you know more
    # mappings, please file a bug report. Thanks.
}

for k, v in sorted(locale_encoding_alias.items()):
    k = k.replace('_', '')
    locale_encoding_alias.setdefault(k, v)

#
# The locale_alias table maps lowercase alias names to C locale names
# (case-sensitive). Encodings are always separated from the locale
# name using a dot ('.'); they should only be given in case the
# language name is needed to interpret the given encoding alias
# correctly (CJK codes often have this need).
#
# Note that the normalize() function which uses this tables
# removes '_' and '-' characters from the encoding part of the
# locale name before doing the lookup. This saves a lot of
# space in the table.
#
# MAL 2004-12-10:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.4
# and older):
#
#    updated 'bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bg_bg' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'bulgarian' -> 'bg_BG.ISO8859-5' to 'bg_BG.CP1251'
#    updated 'cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'cz_cz' -> 'cz_CZ.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'czech' -> 'cs_CS.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'dutch' -> 'nl_BE.ISO8859-1' to 'nl_NL.ISO8859-1'
#    updated 'et' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'et_ee' -> 'et_EE.ISO8859-4' to 'et_EE.ISO8859-15'
#    updated 'fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'fi_fi' -> 'fi_FI.ISO8859-1' to 'fi_FI.ISO8859-15'
#    updated 'iw' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'iw_il' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'japanese' -> 'ja_JP.SJIS' to 'ja_JP.eucJP'
#    updated 'lt' -> 'lt_LT.ISO8859-4' to 'lt_LT.ISO8859-13'
#    updated 'lv' -> 'lv_LV.ISO8859-4' to 'lv_LV.ISO8859-13'
#    updated 'sl' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'slovene' -> 'sl_CS.ISO8859-2' to 'sl_SI.ISO8859-2'
#    updated 'th_th' -> 'th_TH.TACTIS' to 'th_TH.ISO8859-11'
#    updated 'zh_cn' -> 'zh_CN.eucCN' to 'zh_CN.gb2312'
#    updated 'zh_cn.big5' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#    updated 'zh_tw' -> 'zh_TW.eucTW' to 'zh_TW.big5'
#
# MAL 2008-05-30:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.5
# and older):
#
#    updated 'cs_cs.iso88592' -> 'cs_CZ.ISO8859-2' to 'cs_CS.ISO8859-2'
#    updated 'serbocroatian' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh_hr.iso88592' -> 'sh_HR.ISO8859-2' to 'hr_HR.ISO8859-2'
#    updated 'sh_sp' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sh_yu' -> 'sh_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sp' -> 'sp_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sp_yu' -> 'sp_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_sp' -> 'sr_SP.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sr_yu' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.cp1251@cyrillic' -> 'sr_YU.CP1251' to 'sr_CS.CP1251'
#    updated 'sr_yu.iso88592' -> 'sr_YU.ISO8859-2' to 'sr_CS.ISO8859-2'
#    updated 'sr_yu.iso88595' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.iso88595@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#    updated 'sr_yu.microsoftcp1251@cyrillic' -> 'sr_YU.CP1251' to 'sr_CS.CP1251'
#    updated 'sr_yu.utf8@cyrillic' -> 'sr_YU.UTF-8' to 'sr_CS.UTF-8'
#    updated 'sr_yu@cyrillic' -> 'sr_YU.ISO8859-5' to 'sr_CS.ISO8859-5'
#
# AP 2010-04-12:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 2.6.5
# and older):
#
#    updated 'ru' -> 'ru_RU.ISO8859-5' to 'ru_RU.UTF-8'
#    updated 'ru_ru' -> 'ru_RU.ISO8859-5' to 'ru_RU.UTF-8'
#    updated 'serbocroatian' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sh' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sh_yu' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#    updated 'sr@cyrillic' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#    updated 'sr@latn' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr_cs.utf8@latn' -> 'sr_CS.UTF-8' to 'sr_RS.UTF-8@latin'
#    updated 'sr_cs@latn' -> 'sr_CS.ISO8859-2' to 'sr_RS.UTF-8@latin'
#    updated 'sr_yu' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8@latin'
#    updated 'sr_yu.utf8@cyrillic' -> 'sr_CS.UTF-8' to 'sr_RS.UTF-8'
#    updated 'sr_yu@cyrillic' -> 'sr_CS.ISO8859-5' to 'sr_RS.UTF-8'
#
# SS 2013-12-20:
# Updated alias mapping to most recent locale.alias file
# from X.org distribution using makelocalealias.py.
#
# These are the differences compared to the old mapping (Python 3.3.3
# and older):
#
#    updated 'a3' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'a3_az' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'a3_az.koi8c' -> 'a3_AZ.KOI8-C' to 'az_AZ.KOI8-C'
#    updated 'cs_cs.iso88592' -> 'cs_CS.ISO8859-2' to 'cs_CZ.ISO8859-2'
#    updated 'hebrew' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'hebrew.iso88598' -> 'iw_IL.ISO8859-8' to 'he_IL.ISO8859-8'
#    updated 'sd' -> 'sd_IN@devanagari.UTF-8' to 'sd_IN.UTF-8'
#    updated 'sr@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#    updated 'sr_cs' -> 'sr_RS.UTF-8' to 'sr_CS.UTF-8'
#    updated 'sr_cs.utf8@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#    updated 'sr_cs@latn' -> 'sr_RS.UTF-8@latin' to 'sr_CS.UTF-8@latin'
#
# SS 2014-10-01:
# Updated alias mapping with glibc 2.19 supported locales.
#
# SS 2018-05-05:
# Updated alias mapping with glibc 2.27 supported locales.
#
# These are the differences compared to the old mapping (Python 3.6.5
# and older):
#
#    updated 'ca_es@valencia' -> 'ca_ES.ISO8859-15@valencia' to 'ca_ES.UTF-8@valencia'
#    updated 'kk_kz' -> 'kk_KZ.RK1048' to 'kk_KZ.ptcp154'
#    updated 'russian' -> 'ru_RU.ISO8859-5' to 'ru_RU.KOI8-R'

locale_alias = {
    'a3':                                   'az_AZ.KOI8-C',
    'a3_az':                                'az_AZ.KOI8-C',
    'a3_az.koic':                           'az_AZ.KOI8-C',
    'aa_dj':                                'aa_DJ.ISO8859-1',
    'aa_er':                                'aa_ER.UTF-8',
    'aa_et':                                'aa_ET.UTF-8',
    'af':                                   'af_ZA.ISO8859-1',
    'af_za':                                'af_ZA.ISO8859-1',
    'agr_pe':                               'agr_PE.UTF-8',
    'ak_gh':                                'ak_GH.UTF-8',
    'am':                                   'am_ET.UTF-8',
    'am_et':                                'am_ET.UTF-8',
    'american':                             'en_US.ISO8859-1',
    'an_es':                                'an_ES.ISO8859-15',
    'anp_in':                               'anp_IN.UTF-8',
    'ar':                                   'ar_AA.ISO8859-6',
    'ar_aa':                                'ar_AA.ISO8859-6',
    'ar_ae':                                'ar_AE.ISO8859-6',
    'ar_bh':                                'ar_BH.ISO8859-6',
    'ar_dz':                                'ar_DZ.ISO8859-6',
    'ar_eg':                                'ar_EG.ISO8859-6',
    'ar_in':                                'ar_IN.UTF-8',
    'ar_iq':                                'ar_IQ.ISO8859-6',
    'ar_jo':                                'ar_JO.ISO8859-6',
    'ar_kw':                                'ar_KW.ISO8859-6',
    'ar_lb':                                'ar_LB.ISO8859-6',
    'ar_ly':                                'ar_LY.ISO8859-6',
    'ar_ma':                                'ar_MA.ISO8859-6',
    'ar_om':                                'ar_OM.ISO8859-6',
    'ar_qa':                                'ar_QA.ISO8859-6',
    'ar_sa':                                'ar_SA.ISO8859-6',
    'ar_sd':                                'ar_SD.ISO8859-6',
    'ar_ss':                                'ar_SS.UTF-8',
    'ar_sy':                                'ar_SY.ISO8859-6',
    'ar_tn':                                'ar_TN.ISO8859-6',
    'ar_ye':                                'ar_YE.ISO8859-6',
    'arabic':                               'ar_AA.ISO8859-6',
    'as':                                   'as_IN.UTF-8',
    'as_in':                                'as_IN.UTF-8',
    'ast_es':                               'ast_ES.ISO8859-15',
    'ayc_pe':                               'ayc_PE.UTF-8',
    'az':                                   'az_AZ.ISO8859-9E',
    'az_az':                                'az_AZ.ISO8859-9E',
    'az_az.iso88599e':                      'az_AZ.ISO8859-9E',
    'az_ir':                                'az_IR.UTF-8',
    'be':                                   'be_BY.CP1251',
    'be@latin':                             'be_BY.UTF-8@latin',
    'be_bg.utf8':                           'bg_BG.UTF-8',
    'be_by':                                'be_BY.CP1251',
    'be_by@latin':                          'be_BY.UTF-8@latin',
    'bem_zm':                               'bem_ZM.UTF-8',
    'ber_dz':                               'ber_DZ.UTF-8',
    'ber_ma':                               'ber_MA.UTF-8',
    'bg':                                   'bg_BG.CP1251',
    'bg_bg':                                'bg_BG.CP1251',
    'bhb_in.utf8':                          'bhb_IN.UTF-8',
    'bho_in':                               'bho_IN.UTF-8',
    'bho_np':                               'bho_NP.UTF-8',
    'bi_vu':                                'bi_VU.UTF-8',
    'bn_bd':                                'bn_BD.UTF-8',
    'bn_in':                                'bn_IN.UTF-8',
    'bo_cn':                                'bo_CN.UTF-8',
    'bo_in':                                'bo_IN.UTF-8',
    'bokmal':                               'nb_NO.ISO8859-1',
    'bokm\xe5l':                            'nb_NO.ISO8859-1',
    'br':                                   'br_FR.ISO8859-1',
    'br_fr':                                'br_FR.ISO8859-1',
    'brx_in':                               'brx_IN.UTF-8',
    'bs':                                   'bs_BA.ISO8859-2',
    'bs_ba':                                'bs_BA.ISO8859-2',
    'bulgarian':                            'bg_BG.CP1251',
    'byn_er':                               'byn_ER.UTF-8',
    'c':                                    'C',
    'c-french':                             'fr_CA.ISO8859-1',
    'c.ascii':                              'C',
    'c.en':                                 'C',
    'c.iso88591':                           'en_US.ISO8859-1',
    'c.utf8':                               'en_US.UTF-8',
    'c_c':                                  'C',
    'c_c.c':                                'C',
    'ca':                                   'ca_ES.ISO8859-1',
    'ca_ad':                                'ca_AD.ISO8859-1',
    'ca_es':                                'ca_ES.ISO8859-1',
    'ca_es@valencia':                       'ca_ES.UTF-8@valencia',
    'ca_fr':                                'ca_FR.ISO8859-1',
    'ca_it':                                'ca_IT.ISO8859-1',
    'catalan':                              'ca_ES.ISO8859-1',
    'ce_ru':                                'ce_RU.UTF-8',
    'cextend':                              'en_US.ISO8859-1',
    'chinese-s':                            'zh_CN.eucCN',
    'chinese-t':                            'zh_TW.eucTW',
    'chr_us':                               'chr_US.UTF-8',
    'ckb_iq':                               'ckb_IQ.UTF-8',
    'cmn_tw':                               'cmn_TW.UTF-8',
    'crh_ua':                               'crh_UA.UTF-8',
    'croatian':                             'hr_HR.ISO8859-2',
    'cs':                                   'cs_CZ.ISO8859-2',
    'cs_cs':                                'cs_CZ.ISO8859-2',
    'cs_cz':                                'cs_CZ.ISO8859-2',
    'csb_pl':                               'csb_PL.UTF-8',
    'cv_ru':                                'cv_RU.UTF-8',
    'cy':                                   'cy_GB.ISO8859-1',
    'cy_gb':                                'cy_GB.ISO8859-1',
    'cz':                                   'cs_CZ.ISO8859-2',
    'cz_cz':                                'cs_CZ.ISO8859-2',
    'czech':                                'cs_CZ.ISO8859-2',
    'da':                                   'da_DK.ISO8859-1',
    'da_dk':                                'da_DK.ISO8859-1',
    'danish':                               'da_DK.ISO8859-1',
    'dansk':                                'da_DK.ISO8859-1',
    'de':                                   'de_DE.ISO8859-1',
    'de_at':                                'de_AT.ISO8859-1',
    'de_be':                                'de_BE.ISO8859-1',
    'de_ch':                                'de_CH.ISO8859-1',
    'de_de':                                'de_DE.ISO8859-1',
    'de_it':                                'de_IT.ISO8859-1',
    'de_li.utf8':                           'de_LI.UTF-8',
    'de_lu':                                'de_LU.ISO8859-1',
    'deutsch':                              'de_DE.ISO8859-1',
    'doi_in':                               'doi_IN.UTF-8',
    'dutch':                                'nl_NL.ISO8859-1',
    'dutch.iso88591':                       'nl_BE.ISO8859-1',
    'dv_mv':                                'dv_MV.UTF-8',
    'dz_bt':                                'dz_BT.UTF-8',
    'ee':                                   'ee_EE.ISO8859-4',
    'ee_ee':                                'ee_EE.ISO8859-4',
    'eesti':                                'et_EE.ISO8859-1',
    'el':                                   'el_GR.ISO8859-7',
    'el_cy':                                'el_CY.ISO8859-7',
    'el_gr':                                'el_GR.ISO8859-7',
    'el_gr@euro':                           'el_GR.ISO8859-15',
    'en':                                   'en_US.ISO8859-1',
    'en_ag':                                'en_AG.UTF-8',
    'en_au':                                'en_AU.ISO8859-1',
    'en_be':                                'en_BE.ISO8859-1',
    'en_bw':                                'en_BW.ISO8859-1',
    'en_ca':                                'en_CA.ISO8859-1',
    'en_dk':                                'en_DK.ISO8859-1',
    'en_dl.utf8':                           'en_DL.UTF-8',
    'en_gb':                                'en_GB.ISO8859-1',
    'en_hk':                                'en_HK.ISO8859-1',
    'en_ie':                                'en_IE.ISO8859-1',
    'en_il':                                'en_IL.UTF-8',
    'en_in':                                'en_IN.ISO8859-1',
    'en_ng':                                'en_NG.UTF-8',
    'en_nz':                                'en_NZ.ISO8859-1',
    'en_ph':                                'en_PH.ISO8859-1',
    'en_sc.utf8':                           'en_SC.UTF-8',
    'en_sg':                                'en_SG.ISO8859-1',
    'en_uk':                                'en_GB.ISO8859-1',
    'en_us':                                'en_US.ISO8859-1',
    'en_us@euro@euro':                      'en_US.ISO8859-15',
    'en_za':                                'en_ZA.ISO8859-1',
    'en_zm':                                'en_ZM.UTF-8',
    'en_zw':                                'en_ZW.ISO8859-1',
    'en_zw.utf8':                           'en_ZS.UTF-8',
    'eng_gb':                               'en_GB.ISO8859-1',
    'english':                              'en_EN.ISO8859-1',
    'english.iso88591':                     'en_US.ISO8859-1',
    'english_uk':                           'en_GB.ISO8859-1',
    'english_united-states':                'en_US.ISO8859-1',
    'english_united-states.437':            'C',
    'english_us':                           'en_US.ISO8859-1',
    'eo':                                   'eo_XX.ISO8859-3',
    'eo.utf8':                              'eo.UTF-8',
    'eo_eo':                                'eo_EO.ISO8859-3',
    'eo_us.utf8':                           'eo_US.UTF-8',
    'eo_xx':                                'eo_XX.ISO8859-3',
    'es':                                   'es_ES.ISO8859-1',
    'es_ar':                                'es_AR.ISO8859-1',
    'es_bo':                                'es_BO.ISO8859-1',
    'es_cl':                                'es_CL.ISO8859-1',
    'es_co':                                'es_CO.ISO8859-1',
    'es_cr':                                'es_CR.ISO8859-1',
    'es_cu':                                'es_CU.UTF-8',
    'es_do':                                'es_DO.ISO8859-1',
    'es_ec':                                'es_EC.ISO8859-1',
    'es_es':                                'es_ES.ISO8859-1',
    'es_gt':                                'es_GT.ISO8859-1',
    'es_hn':                                'es_HN.ISO8859-1',
    'es_mx':                                'es_MX.ISO8859-1',
    'es_ni':                                'es_NI.ISO8859-1',
    'es_pa':                                'es_PA.ISO8859-1',
    'es_pe':                                'es_PE.ISO8859-1',
    'es_pr':                                'es_PR.ISO8859-1',
    'es_py':                                'es_PY.ISO8859-1',
    'es_sv':                                'es_SV.ISO8859-1',
    'es_us':                                'es_US.ISO8859-1',
    'es_uy':                                'es_UY.ISO8859-1',
    'es_ve':                                'es_VE.ISO8859-1',
    'estonian':                             'et_EE.ISO8859-1',
    'et':                                   'et_EE.ISO8859-15',
    'et_ee':                                'et_EE.ISO8859-15',
    'eu':                                   'eu_ES.ISO8859-1',
    'eu_es':                                'eu_ES.ISO8859-1',
    'eu_fr':                                'eu_FR.ISO8859-1',
    'fa':                                   'fa_IR.UTF-8',
    'fa_ir':                                'fa_IR.UTF-8',
    'fa_ir.isiri3342':                      'fa_IR.ISIRI-3342',
    'ff_sn':                                'ff_SN.UTF-8',
    'fi':                                   'fi_FI.ISO8859-15',
    'fi_fi':                                'fi_FI.ISO8859-15',
    'fil_ph':                               'fil_PH.UTF-8',
    'finnish':                              'fi_FI.ISO8859-1',
    'fo':                                   'fo_FO.ISO8859-1',
    'fo_fo':                                'fo_FO.ISO8859-1',
    'fr':                                   'fr_FR.ISO8859-1',
    'fr_be':                                'fr_BE.ISO8859-1',
    'fr_ca':                                'fr_CA.ISO8859-1',
    'fr_ch':                                'fr_CH.ISO8859-1',
    'fr_fr':                                'fr_FR.ISO8859-1',
    'fr_lu':                                'fr_LU.ISO8859-1',
    'fran\xe7ais':                          'fr_FR.ISO8859-1',
    'fre_fr':                               'fr_FR.ISO8859-1',
    'french':                               'fr_FR.ISO8859-1',
    'french.iso88591':                      'fr_CH.ISO8859-1',
    'french_france':                        'fr_FR.ISO8859-1',
    'fur_it':                               'fur_IT.UTF-8',
    'fy_de':                                'fy_DE.UTF-8',
    'fy_nl':                                'fy_NL.UTF-8',
    'ga':                                   'ga_IE.ISO8859-1',
    'ga_ie':                                'ga_IE.ISO8859-1',
    'galego':                               'gl_ES.ISO8859-1',
    'galician':                             'gl_ES.ISO8859-1',
    'gd':                                   'gd_GB.ISO8859-1',
    'gd_gb':                                'gd_GB.ISO8859-1',
    'ger_de':                               'de_DE.ISO8859-1',
    'german':                               'de_DE.ISO8859-1',
    'german.iso88591':                      'de_CH.ISO8859-1',
    'german_germany':                       'de_DE.ISO8859-1',
    'gez_er':                               'gez_ER.UTF-8',
    'gez_et':                               'gez_ET.UTF-8',
    'gl':                                   'gl_ES.ISO8859-1',
    'gl_es':                                'gl_ES.ISO8859-1',
    'greek':                                'el_GR.ISO8859-7',
    'gu_in':                                'gu_IN.UTF-8',
    'gv':                                   'gv_GB.ISO8859-1',
    'gv_gb':                                'gv_GB.ISO8859-1',
    'ha_ng':                                'ha_NG.UTF-8',
    'hak_tw':                               'hak_TW.UTF-8',
    'he':                                   'he_IL.ISO8859-8',
    'he_il':                                'he_IL.ISO8859-8',
    'hebrew':                               'he_IL.ISO8859-8',
    'hi':                                   'hi_IN.ISCII-DEV',
    'hi_in':                                'hi_IN.ISCII-DEV',
    'hi_in.isciidev':                       'hi_IN.ISCII-DEV',
    'hif_fj':                               'hif_FJ.UTF-8',
    'hne':                                  'hne_IN.UTF-8',
    'hne_in':                               'hne_IN.UTF-8',
    'hr':                                   'hr_HR.ISO8859-2',
    'hr_hr':                                'hr_HR.ISO8859-2',
    'hrvatski':                             'hr_HR.ISO8859-2',
    'hsb_de':                               'hsb_DE.ISO8859-2',
    'ht_ht':                                'ht_HT.UTF-8',
    'hu':                                   'hu_HU.ISO8859-2',
    'hu_hu':                                'hu_HU.ISO8859-2',
    'hungarian':                            'hu_HU.ISO8859-2',
    'hy_am':                                'hy_AM.UTF-8',
    'hy_am.armscii8':                       'hy_AM.ARMSCII_8',
    'ia':                                   'ia.UTF-8',
    'ia_fr':                                'ia_FR.UTF-8',
    'icelandic':                            'is_IS.ISO8859-1',
    'id':                                   'id_ID.ISO8859-1',
    'id_id':                                'id_ID.ISO8859-1',
    'ig_ng':                                'ig_NG.UTF-8',
    'ik_ca':                                'ik_CA.UTF-8',
    'in':                                   'id_ID.ISO8859-1',
    'in_id':                                'id_ID.ISO8859-1',
    'is':                                   'is_IS.ISO8859-1',
    'is_is':                                'is_IS.ISO8859-1',
    'iso-8859-1':                           'en_US.ISO8859-1',
    'iso-8859-15':                          'en_US.ISO8859-15',
    'iso8859-1':                            'en_US.ISO8859-1',
    'iso8859-15':                           'en_US.ISO8859-15',
    'iso_8859_1':                           'en_US.ISO8859-1',
    'iso_8859_15':                          'en_US.ISO8859-15',
    'it':                                   'it_IT.ISO8859-1',
    'it_ch':                                'it_CH.ISO8859-1',
    'it_it':                                'it_IT.ISO8859-1',
    'italian':                              'it_IT.ISO8859-1',
    'iu':                                   'iu_CA.NUNACOM-8',
    'iu_ca':                                'iu_CA.NUNACOM-8',
    'iu_ca.nunacom8':                       'iu_CA.NUNACOM-8',
    'iw':                                   'he_IL.ISO8859-8',
    'iw_il':                                'he_IL.ISO8859-8',
    'iw_il.utf8':                           'iw_IL.UTF-8',
    'ja':                                   'ja_JP.eucJP',
    'ja_jp':                                'ja_JP.eucJP',
    'ja_jp.euc':                            'ja_JP.eucJP',
    'ja_jp.mscode':                         'ja_JP.SJIS',
    'ja_jp.pck':                            'ja_JP.SJIS',
    'japan':                                'ja_JP.eucJP',
    'japanese':                             'ja_JP.eucJP',
    'japanese-euc':                         'ja_JP.eucJP',
    'japanese.euc':                         'ja_JP.eucJP',
    'jp_jp':                                'ja_JP.eucJP',
    'ka':                                   'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge':                                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianacademy':                'ka_GE.GEORGIAN-ACADEMY',
    'ka_ge.georgianps':                     'ka_GE.GEORGIAN-PS',
    'ka_ge.georgianrs':                     'ka_GE.GEORGIAN-ACADEMY',
    'kab_dz':                               'kab_DZ.UTF-8',
    'kk_kz':                                'kk_KZ.ptcp154',
    'kl':                                   'kl_GL.ISO8859-1',
    'kl_gl':                                'kl_GL.ISO8859-1',
    'km_kh':                                'km_KH.UTF-8',
    'kn':                                   'kn_IN.UTF-8',
    'kn_in':                                'kn_IN.UTF-8',
    'ko':                                   'ko_KR.eucKR',
    'ko_kr':                                'ko_KR.eucKR',
    'ko_kr.euc':                            'ko_KR.eucKR',
    'kok_in':                               'kok_IN.UTF-8',
    'korean':                               'ko_KR.eucKR',
    'korean.euc':                           'ko_KR.eucKR',
    'ks':                                   'ks_IN.UTF-8',
    'ks_in':                                'ks_IN.UTF-8',
    'ks_in@devanagari.utf8':                'ks_IN.UTF-8@devanagari',
    'ku_tr':                                'ku_TR.ISO8859-9',
    'kw':                                   'kw_GB.ISO8859-1',
    'kw_gb':                                'kw_GB.ISO8859-1',
    'ky':                                   'ky_KG.UTF-8',
    'ky_kg':                                'ky_KG.UTF-8',
    'lb_lu':                                'lb_LU.UTF-8',
    'lg_ug':                                'lg_UG.ISO8859-10',
    'li_be':                                'li_BE.UTF-8',
    'li_nl':                                'li_NL.UTF-8',
    'lij_it':                               'lij_IT.UTF-8',
    'lithuanian':                           'lt_LT.ISO8859-13',
    'ln_cd':                                'ln_CD.UTF-8',
    'lo':                                   'lo_LA.MULELAO-1',
    'lo_la':                                'lo_LA.MULELAO-1',
    'lo_la.cp1133':                         'lo_LA.IBM-CP1133',
    'lo_la.ibmcp1133':                      'lo_LA.IBM-CP1133',
    'lo_la.mulelao1':                       'lo_LA.MULELAO-1',
    'lt':                                   'lt_LT.ISO8859-13',
    'lt_lt':                                'lt_LT.ISO8859-13',
    'lv':                                   'lv_LV.ISO8859-13',
    'lv_lv':                                'lv_LV.ISO8859-13',
    'lzh_tw':                               'lzh_TW.UTF-8',
    'mag_in':                               'mag_IN.UTF-8',
    'mai':                                  'mai_IN.UTF-8',
    'mai_in':                               'mai_IN.UTF-8',
    'mai_np':                               'mai_NP.UTF-8',
    'mfe_mu':                               'mfe_MU.UTF-8',
    'mg_mg':                                'mg_MG.ISO8859-15',
    'mhr_ru':                               'mhr_RU.UTF-8',
    'mi':                                   'mi_NZ.ISO8859-1',
    'mi_nz':                                'mi_NZ.ISO8859-1',
    'miq_ni':                               'miq_NI.UTF-8',
    'mjw_in':                               'mjw_IN.UTF-8',
    'mk':                                   'mk_MK.ISO8859-5',
    'mk_mk':                                'mk_MK.ISO8859-5',
    'ml':                                   'ml_IN.UTF-8',
    'ml_in':                                'ml_IN.UTF-8',
    'mn_mn':                                'mn_MN.UTF-8',
    'mni_in':                               'mni_IN.UTF-8',
    'mr':                                   'mr_IN.UTF-8',
    'mr_in':                                'mr_IN.UTF-8',
    'ms':                                   'ms_MY.ISO8859-1',
    'ms_my':                                'ms_MY.ISO8859-1',
    'mt':                                   'mt_MT.ISO8859-3',
    'mt_mt':                                'mt_MT.ISO8859-3',
    'my_mm':                                'my_MM.UTF-8',
    'nan_tw':                               'nan_TW.UTF-8',
    'nb':                                   'nb_NO.ISO8859-1',
    'nb_no':                                'nb_NO.ISO8859-1',
    'nds_de':                               'nds_DE.UTF-8',
    'nds_nl':                               'nds_NL.UTF-8',
    'ne_np':                                'ne_NP.UTF-8',
    'nhn_mx':                               'nhn_MX.UTF-8',
    'niu_nu':                               'niu_NU.UTF-8',
    'niu_nz':                               'niu_NZ.UTF-8',
    'nl':                                   'nl_NL.ISO8859-1',
    'nl_aw':                                'nl_AW.UTF-8',
    'nl_be':                                'nl_BE.ISO8859-1',
    'nl_nl':                                'nl_NL.ISO8859-1',
    'nn':                                   'nn_NO.ISO8859-1',
    'nn_no':                                'nn_NO.ISO8859-1',
    'no':                                   'no_NO.ISO8859-1',
    'no@nynorsk':                           'ny_NO.ISO8859-1',
    'no_no':                                'no_NO.ISO8859-1',
    'no_no.iso88591@bokmal':                'no_NO.ISO8859-1',
    'no_no.iso88591@nynorsk':               'no_NO.ISO8859-1',
    'norwegian':                            'no_NO.ISO8859-1',
    'nr':                                   'nr_ZA.ISO8859-1',
    'nr_za':                                'nr_ZA.ISO8859-1',
    'nso':                                  'nso_ZA.ISO8859-15',
    'nso_za':                               'nso_ZA.ISO8859-15',
    'ny':                                   'ny_NO.ISO8859-1',
    'ny_no':                                'ny_NO.ISO8859-1',
    'nynorsk':                              'nn_NO.ISO8859-1',
    'oc':                                   'oc_FR.ISO8859-1',
    'oc_fr':                                'oc_FR.ISO8859-1',
    'om_et':                                'om_ET.UTF-8',
    'om_ke':                                'om_KE.ISO8859-1',
    'or':                                   'or_IN.UTF-8',
    'or_in':                                'or_IN.UTF-8',
    'os_ru':                                'os_RU.UTF-8',
    'pa':                                   'pa_IN.UTF-8',
    'pa_in':                                'pa_IN.UTF-8',
    'pa_pk':                                'pa_PK.UTF-8',
    'pap_an':                               'pap_AN.UTF-8',
    'pap_aw':                               'pap_AW.UTF-8',
    'pap_cw':                               'pap_CW.UTF-8',
    'pd':                                   'pd_US.ISO8859-1',
    'pd_de':                                'pd_DE.ISO8859-1',
    'pd_us':                                'pd_US.ISO8859-1',
    'ph':                                   'ph_PH.ISO8859-1',
    'ph_ph':                                'ph_PH.ISO8859-1',
    'pl':                                   'pl_PL.ISO8859-2',
    'pl_pl':                                'pl_PL.ISO8859-2',
    'polish':                               'pl_PL.ISO8859-2',
    'portuguese':                           'pt_PT.ISO8859-1',
    'portuguese_brazil':                    'pt_BR.ISO8859-1',
    'posix':                                'C',
    'posix-utf2':                           'C',
    'pp':                                   'pp_AN.ISO8859-1',
    'pp_an':                                'pp_AN.ISO8859-1',
    'ps_af':                                'ps_AF.UTF-8',
    'pt':                                   'pt_PT.ISO8859-1',
    'pt_br':                                'pt_BR.ISO8859-1',
    'pt_pt':                                'pt_PT.ISO8859-1',
    'quz_pe':                               'quz_PE.UTF-8',
    'raj_in':                               'raj_IN.UTF-8',
    'ro':                                   'ro_RO.ISO8859-2',
    'ro_ro':                                'ro_RO.ISO8859-2',
    'romanian':                             'ro_RO.ISO8859-2',
    'ru':                                   'ru_RU.UTF-8',
    'ru_ru':                                'ru_RU.UTF-8',
    'ru_ua':                                'ru_UA.KOI8-U',
    'rumanian':                             'ro_RO.ISO8859-2',
    'russian':                              'ru_RU.KOI8-R',
    'rw':                                   'rw_RW.ISO8859-1',
    'rw_rw':                                'rw_RW.ISO8859-1',
    'sa_in':                                'sa_IN.UTF-8',
    'sat_in':                               'sat_IN.UTF-8',
    'sc_it':                                'sc_IT.UTF-8',
    'sd':                                   'sd_IN.UTF-8',
    'sd_in':                                'sd_IN.UTF-8',
    'sd_in@devanagari.utf8':                'sd_IN.UTF-8@devanagari',
    'sd_pk':                                'sd_PK.UTF-8',
    'se_no':                                'se_NO.UTF-8',
    'serbocroatian':                        'sr_RS.UTF-8@latin',
    'sgs_lt':                               'sgs_LT.UTF-8',
    'sh':                                   'sr_RS.UTF-8@latin',
    'sh_ba.iso88592@bosnia':                'sr_CS.ISO8859-2',
    'sh_hr':                                'sh_HR.ISO8859-2',
    'sh_hr.iso88592':                       'hr_HR.ISO8859-2',
    'sh_sp':                                'sr_CS.ISO8859-2',
    'sh_yu':                                'sr_RS.UTF-8@latin',
    'shn_mm':                               'shn_MM.UTF-8',
    'shs_ca':                               'shs_CA.UTF-8',
    'si':                                   'si_LK.UTF-8',
    'si_lk':                                'si_LK.UTF-8',
    'sid_et':                               'sid_ET.UTF-8',
    'sinhala':                              'si_LK.UTF-8',
    'sk':                                   'sk_SK.ISO8859-2',
    'sk_sk':                                'sk_SK.ISO8859-2',
    'sl':                                   'sl_SI.ISO8859-2',
    'sl_cs':                                'sl_CS.ISO8859-2',
    'sl_si':                                'sl_SI.ISO8859-2',
    'slovak':                               'sk_SK.ISO8859-2',
    'slovene':                              'sl_SI.ISO8859-2',
    'slovenian':                            'sl_SI.ISO8859-2',
    'sm_ws':                                'sm_WS.UTF-8',
    'so_dj':                                'so_DJ.ISO8859-1',
    'so_et':                                'so_ET.UTF-8',
    'so_ke':                                'so_KE.ISO8859-1',
    'so_so':                                'so_SO.ISO8859-1',
    'sp':                                   'sr_CS.ISO8859-5',
    'sp_yu':                                'sr_CS.ISO8859-5',
    'spanish':                              'es_ES.ISO8859-1',
    'spanish_spain':                        'es_ES.ISO8859-1',
    'sq':                                   'sq_AL.ISO8859-2',
    'sq_al':                                'sq_AL.ISO8859-2',
    'sq_mk':                                'sq_MK.UTF-8',
    'sr':                                   'sr_RS.UTF-8',
    'sr@cyrillic':                          'sr_RS.UTF-8',
    'sr@latn':                              'sr_CS.UTF-8@latin',
    'sr_cs':                                'sr_CS.UTF-8',
    'sr_cs.iso88592@latn':                  'sr_CS.ISO8859-2',
    'sr_cs@latn':                           'sr_CS.UTF-8@latin',
    'sr_me':                                'sr_ME.UTF-8',
    'sr_rs':                                'sr_RS.UTF-8',
    'sr_rs@latn':                           'sr_RS.UTF-8@latin',
    'sr_sp':                                'sr_CS.ISO8859-2',
    'sr_yu':                                'sr_RS.UTF-8@latin',
    'sr_yu.cp1251@cyrillic':                'sr_CS.CP1251',
    'sr_yu.iso88592':                       'sr_CS.ISO8859-2',
    'sr_yu.iso88595':                       'sr_CS.ISO8859-5',
    'sr_yu.iso88595@cyrillic':              'sr_CS.ISO8859-5',
    'sr_yu.microsoftcp1251@cyrillic':       'sr_CS.CP1251',
    'sr_yu.utf8':                           'sr_RS.UTF-8',
    'sr_yu.utf8@cyrillic':                  'sr_RS.UTF-8',
    'sr_yu@cyrillic':                       'sr_RS.UTF-8',
    'ss':                                   'ss_ZA.ISO8859-1',
    'ss_za':                                'ss_ZA.ISO8859-1',
    'st':                                   'st_ZA.ISO8859-1',
    'st_za':                                'st_ZA.ISO8859-1',
    'sv':                                   'sv_SE.ISO8859-1',
    'sv_fi':                                'sv_FI.ISO8859-1',
    'sv_se':                                'sv_SE.ISO8859-1',
    'sw_ke':                                'sw_KE.UTF-8',
    'sw_tz':                                'sw_TZ.UTF-8',
    'swedish':                              'sv_SE.ISO8859-1',
    'szl_pl':                               'szl_PL.UTF-8',
    'ta':                                   'ta_IN.TSCII-0',
    'ta_in':                                'ta_IN.TSCII-0',
    'ta_in.tscii':                          'ta_IN.TSCII-0',
    'ta_in.tscii0':                         'ta_IN.TSCII-0',
    'ta_lk':                                'ta_LK.UTF-8',
    'tcy_in.utf8':                          'tcy_IN.UTF-8',
    'te':                                   'te_IN.UTF-8',
    'te_in':                                'te_IN.UTF-8',
    'tg':                                   'tg_TJ.KOI8-C',
    'tg_tj':                                'tg_TJ.KOI8-C',
    'th':                                   'th_TH.ISO8859-11',
    'th_th':                                'th_TH.ISO8859-11',
    'th_th.tactis':                         'th_TH.TIS620',
    'th_th.tis620':                         'th_TH.TIS620',
    'thai':                                 'th_TH.ISO8859-11',
    'the_np':                               'the_NP.UTF-8',
    'ti_er':                                'ti_ER.UTF-8',
    'ti_et':                                'ti_ET.UTF-8',
    'tig_er':                               'tig_ER.UTF-8',
    'tk_tm':                                'tk_TM.UTF-8',
    'tl':                                   'tl_PH.ISO8859-1',
    'tl_ph':                                'tl_PH.ISO8859-1',
    'tn':                                   'tn_ZA.ISO8859-15',
    'tn_za':                                'tn_ZA.ISO8859-15',
    'to_to':                                'to_TO.UTF-8',
    'tpi_pg':                               'tpi_PG.UTF-8',
    'tr':                                   'tr_TR.ISO8859-9',
    'tr_cy':                                'tr_CY.ISO8859-9',
    'tr_tr':                                'tr_TR.ISO8859-9',
    'ts':                                   'ts_ZA.ISO8859-1',
    'ts_za':                                'ts_ZA.ISO8859-1',
    'tt':                                   'tt_RU.TATAR-CYR',
    'tt_ru':                                'tt_RU.TATAR-CYR',
    'tt_ru.tatarcyr':                       'tt_RU.TATAR-CYR',
    'tt_ru@iqtelif':                        'tt_RU.UTF-8@iqtelif',
    'turkish':                              'tr_TR.ISO8859-9',
    'ug_cn':                                'ug_CN.UTF-8',
    'uk':                                   'uk_UA.KOI8-U',
    'uk_ua':                                'uk_UA.KOI8-U',
    'univ':                                 'en_US.utf',
    'universal':                            'en_US.utf',
    'universal.utf8@ucs4':                  'en_US.UTF-8',
    'unm_us':                               'unm_US.UTF-8',
    'ur':                                   'ur_PK.CP1256',
    'ur_in':                                'ur_IN.UTF-8',
    'ur_pk':                                'ur_PK.CP1256',
    'uz':                                   'uz_UZ.UTF-8',
    'uz_uz':                                'uz_UZ.UTF-8',
    'uz_uz@cyrillic':                       'uz_UZ.UTF-8',
    've':                                   've_ZA.UTF-8',
    've_za':                                've_ZA.UTF-8',
    'vi':                                   'vi_VN.TCVN',
    'vi_vn':                                'vi_VN.TCVN',
    'vi_vn.tcvn':                           'vi_VN.TCVN',
    'vi_vn.tcvn5712':                       'vi_VN.TCVN',
    'vi_vn.viscii':                         'vi_VN.VISCII',
    'vi_vn.viscii111':                      'vi_VN.VISCII',
    'wa':                                   'wa_BE.ISO8859-1',
    'wa_be':                                'wa_BE.ISO8859-1',
    'wae_ch':                               'wae_CH.UTF-8',
    'wal_et':                               'wal_ET.UTF-8',
    'wo_sn':                                'wo_SN.UTF-8',
    'xh':                                   'xh_ZA.ISO8859-1',
    'xh_za':                                'xh_ZA.ISO8859-1',
    'yi':                                   'yi_US.CP1255',
    'yi_us':                                'yi_US.CP1255',
    'yo_ng':                                'yo_NG.UTF-8',
    'yue_hk':                               'yue_HK.UTF-8',
    'yuw_pg':                               'yuw_PG.UTF-8',
    'zh':                                   'zh_CN.eucCN',
    'zh_cn':                                'zh_CN.gb2312',
    'zh_cn.big5':                           'zh_TW.big5',
    'zh_cn.euc':                            'zh_CN.eucCN',
    'zh_hk':                                'zh_HK.big5hkscs',
    'zh_hk.big5hk':                         'zh_HK.big5hkscs',
    'zh_sg':                                'zh_SG.GB2312',
    'zh_sg.gbk':                            'zh_SG.GBK',
    'zh_tw':                                'zh_TW.big5',
    'zh_tw.euc':                            'zh_TW.eucTW',
    'zh_tw.euctw':                          'zh_TW.eucTW',
    'zu':                                   'zu_ZA.ISO8859-1',
    'zu_za':                                'zu_ZA.ISO8859-1',
}

# This list maps non-NLS codestrings, as included in the UCRT

# Note: some of these values from the NLS reference differ from what setlocale
# returns, in particular 'uk' returns "English_United Kingdom.1252", not the
# value for Ukraine. Here we match the UCRT implementation.
win_non_nls_locales = {
    # UCRT, non-NLS language strings
    "american": "English_United States.1252",
    "american english": "English_United States.1252",
    "american-english": "English_United States.1252",
    "australian": "English_Australia.1252",
    "belgian": "Dutch_Belgium.1252",
    "canadian": "English_Canada.1252",
    "chh": "Chinese (Traditional)_Hong Kong SAR.950",
    "chi": "Chinese (Simplified)_Singapore.936",
    "chinese": "Chinese_China.936",
    "chinese-hongkong": "Chinese (Traditional)_Hong Kong SAR.950",
    "chinese-simplified": "Chinese_China.936",
    "chinese-singapore": "Chinese (Simplified)_Singapore.936",
    "chinese-traditional": "Chinese (Traditional)_Hong Kong SAR.950",
    "dutch-belgian": "Dutch_Belgium.1252",
    "english-american": "English_United States.1252",
    "english-aus": "English_Australia.1252",
    "english-belize": "English_Belize.1252",
    "english-can": "English_Canada.1252",
    "english-caribbean": "English_Caribbean.1252",
    "english-ire": "English_Ireland.1252",
    "english-jamaica": "English_Jamaica.1252",
    "english-nz": "English_New Zealand.1252",
    "english-south africa": "English_South Africa.1252",
    "english-trinidad y tobago": "English_Trinidad and Tobago.1252",
    "english-uk": "English_United Kingdom.1252",
    "english-us": "English_United States.1252",
    "english-usa": "English_United States.1252",
    "french-belgian": "French_Belgium.1252",
    "french-canadian": "French_Canada.1252",
    "french-luxembourg": "French_Luxembourg.1252",
    "french-swiss": "French_Switzerland.1252",
    "german-austrian": "German_Austria.1252",
    "german-lichtenstein": "German_Liechtenstein.1252",
    "german-luxembourg": "German_Luxembourg.1252",
    "german-swiss": "German_Switzerland.1252",
    "irish-english": "English_Ireland.1252",
    "italian-swiss": "Italian_Switzerland.1252",
    "norwegian": "Norwegian_Norway.1252",
    "norwegian-bokmal": "Norwegian Bokm\xe5l_Norway.1252",
    "norwegian-nynorsk": "Norwegian Nynorsk_Norway.1252",
    "portuguese-brazilian": "Portuguese_Brazil.1252",
    "spanish-argentina": "Spanish_Argentina.1252",
    "spanish-bolivia": "Spanish_Bolivia.1252",
    "spanish-chile": "Spanish_Chile.1252",
    "spanish-colombia": "Spanish_Colombia.1252",
    "spanish-costa rica": "Spanish_Costa Rica.1252",
    "spanish-dominican republic": "Spanish_Dominican Republic.1252",
    "spanish-ecuador": "Spanish_Ecuador.1252",
    "spanish-el salvador": "Spanish_El Salvador.1252",
    "spanish-guatemala": "Spanish_Guatemala.1252",
    "spanish-honduras": "Spanish_Honduras.1252",
    "spanish-mexican": "Spanish_Mexico.1252",
    "spanish-modern": "Spanish_Spain.1252",
    "spanish-nicaragua": "Spanish_Nicaragua.1252",
    "spanish-panama": "Spanish_Panama.1252",
    "spanish-paraguay": "Spanish_Paraguay.1252",
    "spanish-peru": "Spanish_Peru.1252",
    "spanish-puerto rico": "Spanish_Puerto Rico.1252",
    "spanish-uruguay": "Spanish_Uruguay.1252",
    "spanish-venezuela": "Spanish_Venezuela.1252",
    "swedish-finland": "Swedish_Finland.1252",
    "swiss": "German_Switzerland.1252",
    "uk": "English_United Kingdom.1252",
    "us": "English_United States.1252"
}

# This maps windows locale names and language codes to their
# fully expanded forms. The availability of these languages varies
# depending on the version of Windows available, this list broadly
# follows what is implemented in Windows Vista. The NLS mappings are
# derived from Windows API calls, and this site:
# https://www.microsoft.com/resources/msdn/goglobal/default.mspx?OS=Windows+7

# these are NLS mapped values. There are four kinds of mappings:
# - language
# - language_country
# - ISO 3166 language codes
# - language code (Microsoft 3 character code)

win_nls_locales = {
    'afrikaans': 'Afrikaans_South Africa.1252',
    'afrikaans_south africa': 'Afrikaans_South Africa.1252',
    'af': 'Afrikaans_South Africa.1252',
    'af-za': 'Afrikaans_South Africa.1252',
    'afk': 'Afrikaans_South Africa.1252',
    'albanian': 'Albanian_Albania.1250',
    'albanian_albania': 'Albanian_Albania.1250',
    'sq': 'Albanian_Albania.1250',
    'sq-al': 'Albanian_Albania.1250',
    'sqi': 'Albanian_Albania.1250',
    'alsatian': 'Alsatian_France.1252',
    'alsatian_france': 'Alsatian_France.1252',
    'gsw': 'Alsatian_France.1252',
    'gsw-fr': 'Alsatian_France.1252',
    'amharic': 'Amharic_Ethiopia.1252',
    'amharic_ethiopia': 'Amharic_Ethiopia.1252',
    'am': 'Amharic_Ethiopia.1252',
    'am-et': 'Amharic_Ethiopia.1252',
    'amh': 'Amharic_Ethiopia.1252',
    'arabic': 'Arabic_Saudi Arabia.1256',
    'arabic_saudi arabia': 'Arabic_Saudi Arabia.1256',
    'ar': 'Arabic_Saudi Arabia.1256',
    'ar-sa': 'Arabic_Saudi Arabia.1256',
    'ara': 'Arabic_Saudi Arabia.1256',
    'arabic_algeria': 'Arabic_Algeria.1256',
    'ar-dz': 'Arabic_Algeria.1256',
    'arg': 'Arabic_Algeria.1256',
    'arabic_bahrain': 'Arabic_Bahrain.1256',
    'ar-bh': 'Arabic_Bahrain.1256',
    'arh': 'Arabic_Bahrain.1256',
    'arabic_egypt': 'Arabic_Egypt.1256',
    'ar-eg': 'Arabic_Egypt.1256',
    'are': 'Arabic_Egypt.1256',
    'arabic_iraq': 'Arabic_Iraq.1256',
    'ar-iq': 'Arabic_Iraq.1256',
    'ari': 'Arabic_Iraq.1256',
    'arabic_jordan': 'Arabic_Jordan.1256',
    'ar-jo': 'Arabic_Jordan.1256',
    'arj': 'Arabic_Jordan.1256',
    'arabic_kuwait': 'Arabic_Kuwait.1256',
    'ar-kw': 'Arabic_Kuwait.1256',
    'ark': 'Arabic_Kuwait.1256',
    'arabic_lebanon': 'Arabic_Lebanon.1256',
    'ar-lb': 'Arabic_Lebanon.1256',
    'arb': 'Arabic_Lebanon.1256',
    'arabic_libya': 'Arabic_Libya.1256',
    'ar-ly': 'Arabic_Libya.1256',
    'arl': 'Arabic_Libya.1256',
    'arabic_morocco': 'Arabic_Morocco.1256',
    'ar-ma': 'Arabic_Morocco.1256',
    'arm': 'Arabic_Morocco.1256',
    'arabic_oman': 'Arabic_Oman.1256',
    'ar-om': 'Arabic_Oman.1256',
    'aro': 'Arabic_Oman.1256',
    'arabic_qatar': 'Arabic_Qatar.1256',
    'ar-qa': 'Arabic_Qatar.1256',
    'arq': 'Arabic_Qatar.1256',
    'arabic_syria': 'Arabic_Syria.1256',
    'ar-sy': 'Arabic_Syria.1256',
    'ars': 'Arabic_Syria.1256',
    'arabic_tunisia': 'Arabic_Tunisia.1256',
    'ar-tn': 'Arabic_Tunisia.1256',
    'art': 'Arabic_Tunisia.1256',
    'arabic_united arab emirates':
    'Arabic_United Arab Emirates.1256',
    'ar-ae': 'Arabic_United Arab Emirates.1256',
    'aru': 'Arabic_United Arab Emirates.1256',
    'arabic_yemen': 'Arabic_Yemen.1256',
    'ar-ye': 'Arabic_Yemen.1256',
    'ary': 'Arabic_Yemen.1256',
    'armenian': 'Armenian_Armenia.1252',
    'armenian_armenia': 'Armenian_Armenia.1252',
    'hy': 'Armenian_Armenia.1252',
    'hy-am': 'Armenian_Armenia.1252',
    'hye': 'Armenian_Armenia.1252',
    'assamese': 'Assamese_India.1252',
    'assamese_india': 'Assamese_India.1252',
    'as': 'Assamese_India.1252',
    'as-in': 'Assamese_India.1252',
    'asm': 'Assamese_India.1252',
    'Azerbaijani (Latin)': 'Azerbaijani (Latin)_Azerbaijan.1254',
    'azerbaijani': 'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'azerbaijani (latin)_azerbaijan':
    'Azerbaijani (Latin)_Azerbaijan.1254',
    'az': 'Azerbaijani (Latin)_Azerbaijan.1254',
    'az-latn': 'Azerbaijani (Latin)_Azerbaijan.1254',
    'az-latn-az': 'Azerbaijani (Latin)_Azerbaijan.1254',
    'aze': 'Azerbaijani_Azerbaijan.1254',
    'Azerbaijani (Cyrillic)':
    'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'azerbaijani (cyrillic)_azerbaijan':
    'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'az-cyrl': 'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'az-cyrl-az': 'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'azc': 'Azerbaijani (Cyrillic)_Azerbaijan.1251',
    'bashkir': 'Bashkir_Russia.1251',
    'bashkir_russia': 'Bashkir_Russia.1251',
    'ba': 'Bashkir_Russia.1251',
    'ba-ru': 'Bashkir_Russia.1251',
    'bas': 'Bashkir_Russia.1251',
    'basque': 'Basque_Spain.1252',
    'basque_spain': 'Basque_Spain.1252',
    'eu': 'Basque_Spain.1252',
    'eu-es': 'Basque_Spain.1252',
    'euq': 'Basque_Spain.1252',
    'belarusian': 'Belarusian_Belarus.1251',
    'belarusian_belarus': 'Belarusian_Belarus.1251',
    'be': 'Belarusian_Belarus.1251',
    'be-by': 'Belarusian_Belarus.1251',
    'bel': 'Belarusian_Belarus.1251',
    'bangla': 'Bangla_Bangladesh.1252',
    'bangla_india': 'Bangla_India.1252',
    'bn': 'Bangla_India.1252',
    'bn-in': 'Bangla_India.1252',
    'bng': 'Bangla_India.1252',
    'bangla_bangladesh': 'Bangla_Bangladesh.1252',
    'bn-bd': 'Bangla_Bangladesh.1252',
    'bnb': 'Bangla_Bangladesh.1252',
    'Bosnian (Latin)': 'Bosnian (Latin)_Bosnia and Herzegovina.1250',
    'bosnian': 'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'bosnian (latin)_bosnia and herzegovina':
    'Bosnian (Latin)_Bosnia and Herzegovina.1250',
    'bs': 'Bosnian (Latin)_Bosnia and Herzegovina.1250',
    'bs-latn': 'Bosnian (Latin)_Bosnia and Herzegovina.1250',
    'bs-latn-ba': 'Bosnian (Latin)_Bosnia and Herzegovina.1250',
    'bsb': 'Bosnian_Bosnia and Herzegovina.1250',
    'Bosnian (Cyrillic)':
    'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'bosnian (cyrillic)_bosnia and herzegovina':
    'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'bs-cyrl': 'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'bs-cyrl-ba': 'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'bsc': 'Bosnian (Cyrillic)_Bosnia and Herzegovina.1251',
    'breton': 'Breton_France.1252',
    'breton_france': 'Breton_France.1252',
    'br': 'Breton_France.1252',
    'br-fr': 'Breton_France.1252',
    'bre': 'Breton_France.1252',
    'bulgarian': 'Bulgarian_Bulgaria.1251',
    'bulgarian_bulgaria': 'Bulgarian_Bulgaria.1251',
    'bg': 'Bulgarian_Bulgaria.1251',
    'bg-bg': 'Bulgarian_Bulgaria.1251',
    'bgr': 'Bulgarian_Bulgaria.1251',
    'catalan': 'Catalan_Spain.1252',
    'catalan_spain': 'Catalan_Spain.1252',
    'ca': 'Catalan_Spain.1252',
    'ca-es': 'Catalan_Spain.1252',
    'cat': 'Catalan_Spain.1252',
    'chinese_china': 'Chinese_China.936',
    'zh': 'Chinese_China.936',
    'zh-hans': 'Chinese_China.936',
    'zh-cn': 'Chinese_China.936',
    'chs': 'Chinese_China.936',
    'chinese_singapore': 'Chinese (Simplified)_Singapore.936',
    'zh-sg': 'Chinese (Simplified)_Singapore.936',
    'zhi': 'Chinese (Simplified)_Singapore.936',
    'chinese (Traditional)': 'Chinese (Traditional)_Taiwan.950',
    'chinese (traditional)_hong kong sar':
    'Chinese (Traditional)_Hong Kong SAR.950',
    'zh-hant': 'Chinese (Traditional)_Hong Kong SAR.950',
    'zh-hk': 'Chinese (Traditional)_Hong Kong SAR.950',
    'zhh': 'Chinese (Traditional)_Hong Kong SAR.950',
    'chinese (traditional)_macao sar':
    'Chinese (Traditional)_Macao SAR.950',
    'zh-mo': 'Chinese (Traditional)_Macao SAR.950',
    'zhm': 'Chinese (Traditional)_Macao SAR.950',
    'chinese (traditional)_taiwan':
    'Chinese (Traditional)_Taiwan.950',
    'zh-tw': 'Chinese (Traditional)_Taiwan.950',
    'cht': 'Chinese (Traditional)_Taiwan.950',
    'corsican': 'Corsican_France.1252',
    'corsican_france': 'Corsican_France.1252',
    'co': 'Corsican_France.1252',
    'co-fr': 'Corsican_France.1252',
    'cos': 'Corsican_France.1252',
    'croatian': 'Croatian_Bosnia and Herzegovina.1250',
    'croatian_croatia': 'Croatian_Croatia.1250',
    'hr': 'Croatian_Croatia.1250',
    'hr-hr': 'Croatian_Croatia.1250',
    'hrv': 'Croatian_Croatia.1250',
    'croatian_bosnia and herzegovina':
    'Croatian_Bosnia and Herzegovina.1250',
    'hr-ba': 'Croatian_Bosnia and Herzegovina.1250',
    'hrb': 'Croatian_Bosnia and Herzegovina.1250',
    'czech': 'Czech_Czech Republic.1250',
    'czech_czech republic': 'Czech_Czech Republic.1250',
    'cs': 'Czech_Czech Republic.1250',
    'cs-cz': 'Czech_Czech Republic.1250',
    'csy': 'Czech_Czech Republic.1250',
    'danish': 'Danish_Denmark.1252',
    'danish_denmark': 'Danish_Denmark.1252',
    'da': 'Danish_Denmark.1252',
    'da-dk': 'Danish_Denmark.1252',
    'dan': 'Danish_Denmark.1252',
    'dari': 'Dari_Afghanistan.1256',
    'dari_afghanistan': 'Dari_Afghanistan.1256',
    'prs': 'Dari_Afghanistan.1256',
    'prs-af': 'Dari_Afghanistan.1256',
    'divehi': 'Divehi_Maldives.1252',
    'divehi_maldives': 'Divehi_Maldives.1252',
    'dv': 'Divehi_Maldives.1252',
    'dv-mv': 'Divehi_Maldives.1252',
    'div': 'Divehi_Maldives.1252',
    'dutch': 'Dutch_Belgium.1252',
    'dutch_netherlands': 'Dutch_Netherlands.1252',
    'nl': 'Dutch_Netherlands.1252',
    'nl-nl': 'Dutch_Netherlands.1252',
    'nld': 'Dutch_Netherlands.1252',
    'dutch_belgium': 'Dutch_Belgium.1252',
    'nl-be': 'Dutch_Belgium.1252',
    'nlb': 'Dutch_Belgium.1252',
    'english': 'English_Zimbabwe.1252',
    'english_united states': 'English_United States.1252',
    'en': 'English_United States.1252',
    'en-us': 'English_United States.1252',
    'enu': 'English_United States.1252',
    'english_australia': 'English_Australia.1252',
    'en-au': 'English_Australia.1252',
    'ena': 'English_Australia.1252',
    'english_belize': 'English_Belize.1252',
    'en-bz': 'English_Belize.1252',
    'enl': 'English_Belize.1252',
    'english_canada': 'English_Canada.1252',
    'en-ca': 'English_Canada.1252',
    'enc': 'English_Canada.1252',
    'english_caribbean': 'English_Caribbean.1252',
    'en-029': 'English_Caribbean.1252',
    'enb': 'English_Caribbean.1252',
    'english_india': 'English_India.1252',
    'en-in': 'English_India.1252',
    'enn': 'English_India.1252',
    'english_ireland': 'English_Ireland.1252',
    'en-ie': 'English_Ireland.1252',
    'eni': 'English_Ireland.1252',
    'english_jamaica': 'English_Jamaica.1252',
    'en-jm': 'English_Jamaica.1252',
    'enj': 'English_Jamaica.1252',
    'english_malaysia': 'English_Malaysia.1252',
    'en-my': 'English_Malaysia.1252',
    'enm': 'English_Malaysia.1252',
    'english_new zealand': 'English_New Zealand.1252',
    'en-nz': 'English_New Zealand.1252',
    'enz': 'English_New Zealand.1252',
    'english_philippines': 'English_Philippines.1252',
    'en-ph': 'English_Philippines.1252',
    'enp': 'English_Philippines.1252',
    'english_singapore': 'English_Singapore.1252',
    'en-sg': 'English_Singapore.1252',
    'ene': 'English_Singapore.1252',
    'english_south africa': 'English_South Africa.1252',
    'en-za': 'English_South Africa.1252',
    'ens': 'English_South Africa.1252',
    'english_trinidad and tobago':
    'English_Trinidad and Tobago.1252',
    'en-tt': 'English_Trinidad and Tobago.1252',
    'ent': 'English_Trinidad and Tobago.1252',
    'english_united kingdom': 'English_United Kingdom.1252',
    'en-gb': 'English_United Kingdom.1252',
    'eng': 'English_United Kingdom.1252',
    'english_zimbabwe': 'English_Zimbabwe.1252',
    'en-zw': 'English_Zimbabwe.1252',
    'enw': 'English_Zimbabwe.1252',
    'estonian': 'Estonian_Estonia.1257',
    'estonian_estonia': 'Estonian_Estonia.1257',
    'et': 'Estonian_Estonia.1257',
    'et-ee': 'Estonian_Estonia.1257',
    'eti': 'Estonian_Estonia.1257',
    'faroese': 'Faroese_Faroe Islands.1252',
    'faroese_faroe islands': 'Faroese_Faroe Islands.1252',
    'fo': 'Faroese_Faroe Islands.1252',
    'fo-fo': 'Faroese_Faroe Islands.1252',
    'fos': 'Faroese_Faroe Islands.1252',
    'filipino': 'Filipino_Philippines.1252',
    'filipino_philippines': 'Filipino_Philippines.1252',
    'fil': 'Filipino_Philippines.1252',
    'fil-ph': 'Filipino_Philippines.1252',
    'fpo': 'Filipino_Philippines.1252',
    'finnish': 'Finnish_Finland.1252',
    'finnish_finland': 'Finnish_Finland.1252',
    'fi': 'Finnish_Finland.1252',
    'fi-fi': 'Finnish_Finland.1252',
    'fin': 'Finnish_Finland.1252',
    'french': 'French_France.1252',
    'french_france': 'French_France.1252',
    'fr': 'French_France.1252',
    'fr-fr': 'French_France.1252',
    'fra': 'French_France.1252',
    'french_belgium': 'French_Belgium.1252',
    'fr-be': 'French_Belgium.1252',
    'frb': 'French_Belgium.1252',
    'french_canada': 'French_Canada.1252',
    'fr-ca': 'French_Canada.1252',
    'frc': 'French_Canada.1252',
    'french_luxembourg': 'French_Luxembourg.1252',
    'fr-lu': 'French_Luxembourg.1252',
    'frl': 'French_Luxembourg.1252',
    'french_monaco': 'French_Monaco.1252',
    'fr-mc': 'French_Monaco.1252',
    'frm': 'French_Monaco.1252',
    'french_switzerland': 'French_Switzerland.1252',
    'fr-ch': 'French_Switzerland.1252',
    'frs': 'French_Switzerland.1252',
    'western frisian': 'Western Frisian_Netherlands.1252',
    'western frisian_netherlands': 'Western Frisian_Netherlands.1252',
    'fy': 'Western Frisian_Netherlands.1252',
    'fy-nl': 'Western Frisian_Netherlands.1252',
    'fyn': 'Western Frisian_Netherlands.1252',
    'galician': 'Galician_Spain.1252',
    'galician_spain': 'Galician_Spain.1252',
    'gl': 'Galician_Spain.1252',
    'gl-es': 'Galician_Spain.1252',
    'glc': 'Galician_Spain.1252',
    'georgian': 'Georgian_Georgia.1252',
    'georgian_georgia': 'Georgian_Georgia.1252',
    'ka': 'Georgian_Georgia.1252',
    'ka-ge': 'Georgian_Georgia.1252',
    'kat': 'Georgian_Georgia.1252',
    'german': 'German_Germany.1252',
    'german_germany': 'German_Germany.1252',
    'de': 'German_Germany.1252',
    'de-de': 'German_Germany.1252',
    'deu': 'German_Germany.1252',
    'german_austria': 'German_Austria.1252',
    'de-at': 'German_Austria.1252',
    'dea': 'German_Austria.1252',
    'german_liechtenstein': 'German_Liechtenstein.1252',
    'de-li': 'German_Liechtenstein.1252',
    'dec': 'German_Liechtenstein.1252',
    'german_luxembourg': 'German_Luxembourg.1252',
    'de-lu': 'German_Luxembourg.1252',
    'del': 'German_Luxembourg.1252',
    'german_switzerland': 'German_Switzerland.1252',
    'de-ch': 'German_Switzerland.1252',
    'des': 'German_Switzerland.1252',
    'greek': 'Greek_Greece.1253',
    'greek_greece': 'Greek_Greece.1253',
    'el': 'Greek_Greece.1253',
    'el-gr': 'Greek_Greece.1253',
    'ell': 'Greek_Greece.1253',
    'greenlandic': 'Greenlandic_Greenland.1252',
    'greenlandic_greenland': 'Greenlandic_Greenland.1252',
    'kl': 'Greenlandic_Greenland.1252',
    'kl-gl': 'Greenlandic_Greenland.1252',
    'kal': 'Greenlandic_Greenland.1252',
    'gujarati': 'Gujarati_India.1252',
    'gujarati_india': 'Gujarati_India.1252',
    'gu': 'Gujarati_India.1252',
    'gu-in': 'Gujarati_India.1252',
    'guj': 'Gujarati_India.1252',
    'hausa': 'Hausa_Nigeria.1252',
    'hausa_nigeria': 'Hausa_Nigeria.1252',
    'ha': 'Hausa_Nigeria.1252',
    'ha-latn': 'Hausa_Nigeria.1252',
    'ha-latn-ng': 'Hausa_Nigeria.1252',
    'hau': 'Hausa_Nigeria.1252',
    'hebrew': 'Hebrew_Israel.1255',
    'hebrew_israel': 'Hebrew_Israel.1255',
    'he': 'Hebrew_Israel.1255',
    'he-il': 'Hebrew_Israel.1255',
    'heb': 'Hebrew_Israel.1255',
    'hindi': 'Hindi_India.1252',
    'hindi_india': 'Hindi_India.1252',
    'hi': 'Hindi_India.1252',
    'hi-in': 'Hindi_India.1252',
    'hin': 'Hindi_India.1252',
    'hungarian': 'Hungarian_Hungary.1250',
    'hungarian_hungary': 'Hungarian_Hungary.1250',
    'hu': 'Hungarian_Hungary.1250',
    'hu-hu': 'Hungarian_Hungary.1250',
    'hun': 'Hungarian_Hungary.1250',
    'icelandic': 'Icelandic_Iceland.1252',
    'icelandic_iceland': 'Icelandic_Iceland.1252',
    'is': 'Icelandic_Iceland.1252',
    'is-is': 'Icelandic_Iceland.1252',
    'isl': 'Icelandic_Iceland.1252',
    'igbo': 'Igbo_Nigeria.1252',
    'igbo_nigeria': 'Igbo_Nigeria.1252',
    'ig': 'Igbo_Nigeria.1252',
    'ig-ng': 'Igbo_Nigeria.1252',
    'ibo': 'Igbo_Nigeria.1252',
    'indonesian': 'Indonesian_Indonesia.1252',
    'indonesian_indonesia': 'Indonesian_Indonesia.1252',
    'id': 'Indonesian_Indonesia.1252',
    'id-id': 'Indonesian_Indonesia.1252',
    'ind': 'Indonesian_Indonesia.1252',
    'inuktitut': 'Inuktitut (Syllabics)_Canada.1252',
    'inuktitut_canada': 'Inuktitut_Canada.1252',
    'iu': 'Inuktitut_Canada.1252',
    'iuk': 'Inuktitut_Canada.1252',
    'Inuktitut (latin)': 'Inuktitut (Latin)_Canada.1252',
    'inuktitut (latin)_canada': 'Inuktitut (Latin)_Canada.1252',
    'iu-latn': 'Inuktitut (Latin)_Canada.1252',
    'iu-latn-ca': 'Inuktitut (Latin)_Canada.1252',
    'Inuktitut (syllabics)': 'Inuktitut (Syllabics)_Canada.1252',
    'inuktitut (syllabics)_canada':
    'Inuktitut (Syllabics)_Canada.1252',
    'iu-cans': 'Inuktitut (Syllabics)_Canada.1252',
    'iu-cans-ca': 'Inuktitut (Syllabics)_Canada.1252',
    'ius': 'Inuktitut (Syllabics)_Canada.1252',
    'irish': 'Irish_Ireland.1252',
    'irish_ireland': 'Irish_Ireland.1252',
    'ga': 'Irish_Ireland.1252',
    'ga-ie': 'Irish_Ireland.1252',
    'ire': 'Irish_Ireland.1252',
    'isixhosa': 'isiXhosa_South Africa.1252',
    'isixhosa_south africa': 'isiXhosa_South Africa.1252',
    'xh': 'isiXhosa_South Africa.1252',
    'xh-za': 'isiXhosa_South Africa.1252',
    'xho': 'isiXhosa_South Africa.1252',
    'isizulu': 'isiZulu_South Africa.1252',
    'isizulu_south africa': 'isiZulu_South Africa.1252',
    'zu': 'isiZulu_South Africa.1252',
    'zu-za': 'isiZulu_South Africa.1252',
    'zul': 'isiZulu_South Africa.1252',
    'italian': 'Italian_Switzerland.1252',
    'italian_italy': 'Italian_Italy.1252',
    'it': 'Italian_Italy.1252',
    'it-it': 'Italian_Italy.1252',
    'ita': 'Italian_Italy.1252',
    'italian_switzerland': 'Italian_Switzerland.1252',
    'it-ch': 'Italian_Switzerland.1252',
    'its': 'Italian_Switzerland.1252',
    'japanese': 'Japanese_Japan.932',
    'japanese_japan': 'Japanese_Japan.932',
    'ja': 'Japanese_Japan.932',
    'ja-jp': 'Japanese_Japan.932',
    'jpn': 'Japanese_Japan.932',
    'kannada': 'Kannada_India.1252',
    'kannada_india': 'Kannada_India.1252',
    'kn': 'Kannada_India.1252',
    'kn-in': 'Kannada_India.1252',
    'kdi': 'Kannada_India.1252',
    'kazakh': 'Kazakh_Kazakhstan.1252',
    'kazakh_kazakhstan': 'Kazakh_Kazakhstan.1252',
    'kk': 'Kazakh_Kazakhstan.1252',
    'kk-kz': 'Kazakh_Kazakhstan.1252',
    'kkz': 'Kazakh_Kazakhstan.1252',
    'khmer': 'Khmer_Cambodia.1252',
    'khmer_cambodia': 'Khmer_Cambodia.1252',
    'km': 'Khmer_Cambodia.1252',
    'km-kh': 'Khmer_Cambodia.1252',
    'khm': 'Khmer_Cambodia.1252',
    "k\'iche\'": "K\'iche\'_Guatemala.1252",
    "k\'iche\'_guatemala": "K\'iche\'_Guatemala.1252",
    'qut': "K\'iche\'_Guatemala.1252",
    'qut-gt': "K\'iche\'_Guatemala.1252",
    'kinyarwanda': 'Kinyarwanda_Rwanda.1252',
    'kinyarwanda_rwanda': 'Kinyarwanda_Rwanda.1252',
    'rw': 'Kinyarwanda_Rwanda.1252',
    'rw-rw': 'Kinyarwanda_Rwanda.1252',
    'kin': 'Kinyarwanda_Rwanda.1252',
    'kiswahili': 'Kiswahili_Kenya.1252',
    'kiswahili_kenya': 'Kiswahili_Kenya.1252',
    'sw': 'Kiswahili_Kenya.1252',
    'sw-ke': 'Kiswahili_Kenya.1252',
    'swk': 'Kiswahili_Kenya.1252',
    'konkani': 'Konkani_India.1252',
    'konkani_india': 'Konkani_India.1252',
    'kok': 'Konkani_India.1252',
    'kok-in': 'Konkani_India.1252',
    'korean': 'Korean_Korea.949',
    'korean_korea': 'Korean_Korea.949',
    'ko': 'Korean_Korea.949',
    'ko-kr': 'Korean_Korea.949',
    'kor': 'Korean_Korea.949',
    'kyrgyz': 'Kyrgyz_Kyrgyzstan.1251',
    'kyrgyz_kyrgyzstan': 'Kyrgyz_Kyrgyzstan.1251',
    'ky': 'Kyrgyz_Kyrgyzstan.1251',
    'ky-kg': 'Kyrgyz_Kyrgyzstan.1251',
    'kyr': 'Kyrgyz_Kyrgyzstan.1251',
    'lao': 'Lao_Laos.1252',
    'lao_laos': 'Lao_Laos.1252',
    'lo': 'Lao_Laos.1252',
    'lo-la': 'Lao_Laos.1252',
    'latvian': 'Latvian_Latvia.1257',
    'latvian_latvia': 'Latvian_Latvia.1257',
    'lv': 'Latvian_Latvia.1257',
    'lv-lv': 'Latvian_Latvia.1257',
    'lvi': 'Latvian_Latvia.1257',
    'lithuanian': 'Lithuanian_Lithuania.1257',
    'lithuanian_lithuania': 'Lithuanian_Lithuania.1257',
    'lt': 'Lithuanian_Lithuania.1257',
    'lt-lt': 'Lithuanian_Lithuania.1257',
    'lth': 'Lithuanian_Lithuania.1257',
    'lower sorbian': 'Lower Sorbian_Germany.1252',
    'lower sorbian_germany': 'Lower Sorbian_Germany.1252',
    'dsb': 'Lower Sorbian_Germany.1252',
    'dsb-de': 'Lower Sorbian_Germany.1252',
    'luxembourgish': 'Luxembourgish_Luxembourg.1252',
    'luxembourgish_luxembourg': 'Luxembourgish_Luxembourg.1252',
    'lb': 'Luxembourgish_Luxembourg.1252',
    'lb-lu': 'Luxembourgish_Luxembourg.1252',
    'lxb': 'Luxembourgish_Luxembourg.1252',
    # expanded form not working correctly for Macedonian
    'macedonian': 'mki',
    'mk': 'mki',
    'mki': 'Macedonian_Macedonia, FYRO.1251',
    # expanded form not working correctly
    'mk-mk': 'mki',
    'malay': 'Malay_Brunei.1252',
    'malay_malaysia': 'Malay_Malaysia.1252',
    'ms': 'Malay_Malaysia.1252',
    'ms-my': 'Malay_Malaysia.1252',
    'msl': 'Malay_Malaysia.1252',
    'malay_brunei': 'Malay_Brunei.1252',
    'ms-bn': 'Malay_Brunei.1252',
    'msb': 'Malay_Brunei.1252',
    'malayalam': 'Malayalam_India.1252',
    'malayalam_india': 'Malayalam_India.1252',
    'ml': 'Malayalam_India.1252',
    'ml-in': 'Malayalam_India.1252',
    'mym': 'Malayalam_India.1252',
    'maltese': 'Maltese_Malta.1252',
    'maltese_malta': 'Maltese_Malta.1252',
    'mt': 'Maltese_Malta.1252',
    'mt-mt': 'Maltese_Malta.1252',
    'mlt': 'Maltese_Malta.1252',
    'maori': 'Maori_New Zealand.1252',
    'maori_new zealand': 'Maori_New Zealand.1252',
    'mi': 'Maori_New Zealand.1252',
    'mi-nz': 'Maori_New Zealand.1252',
    'mri': 'Maori_New Zealand.1252',
    'mapudungun': 'Mapudungun_Chile.1252',
    'mapudungun_chile': 'Mapudungun_Chile.1252',
    'arn': 'Mapudungun_Chile.1252',
    'arn-cl': 'Mapudungun_Chile.1252',
    'mpd': 'Mapudungun_Chile.1252',
    'marathi': 'Marathi_India.1252',
    'marathi_india': 'Marathi_India.1252',
    'mr': 'Marathi_India.1252',
    'mr-in': 'Marathi_India.1252',
    'mar': 'Marathi_India.1252',
    'mohawk': 'Mohawk_Canada.1252',
    'mohawk_canada': 'Mohawk_Canada.1252',
    'moh': 'Mohawk_Canada.1252',
    'moh-ca': 'Mohawk_Canada.1252',
    'mwk': 'Mohawk_Canada.1252',
    'mongolian': 'Mongolian (Traditional Mongolian)_China.1252',
    'mongolian_mongolia': 'Mongolian_Mongolia.1251',
    'mn': 'Mongolian_Mongolia.1251',
    'mn-cyrl': 'Mongolian_Mongolia.1251',
    'mn-mn': 'Mongolian_Mongolia.1251',
    'mnn': 'Mongolian_Mongolia.1251',
    'Mongolian (traditional mongolian)':
    'Mongolian (Traditional Mongolian)_China.1252',
    'mongolian (traditional mongolian)_china':
    'Mongolian (Traditional Mongolian)_China.1252',
    'mn-mong': 'Mongolian (Traditional Mongolian)_China.1252',
    'mn-mong-cn': 'Mongolian (Traditional Mongolian)_China.1252',
    'mng': 'Mongolian (Traditional Mongolian)_China.1252',
    'nepali': 'Nepali_Nepal.1252',
    'nepali_nepal': 'Nepali_Nepal.1252',
    'ne': 'Nepali_Nepal.1252',
    'ne-np': 'Nepali_Nepal.1252',
    'nep': 'Nepali_Nepal.1252',
    'norwegian': 'Norwegian_Norway.1252',
    'norwegian_norway': 'Norwegian_Norway.1252',
    'no': 'Norwegian_Norway.1252',
    'nor': 'Norwegian Bokm\xe5l_Norway.1252',
    # Unicode input invalid due to Python bug
    # 'norwegian bokm\xe5l': 'Norwegian Bokm\xe5l_Norway.1252',
    # 'norwegian bokm\xe5l_norway': 'Norwegian Bokm\xe5l_Norway.1252',
    #'nb': 'Norwegian Bokm\xe5l_Norway.1252',
    #'nb-no': 'Norwegian Bokm\xe5l_Norway.1252',
    'norwegian nynorsk': 'Norwegian Nynorsk_Norway.1252',
    'norwegian nynorsk_norway': 'Norwegian Nynorsk_Norway.1252',
    'nn': 'Norwegian Nynorsk_Norway.1252',
    'nn-no': 'Norwegian Nynorsk_Norway.1252',
    'non': 'Norwegian Nynorsk_Norway.1252',
    'occitan': 'Occitan_France.1252',
    'occitan_france': 'Occitan_France.1252',
    'oc': 'Occitan_France.1252',
    'oc-fr': 'Occitan_France.1252',
    'oci': 'Occitan_France.1252',
    'odia': 'Odia_India.1252',
    'odia_india': 'Odia_India.1252',
    'or': 'Odia_India.1252',
    'or-in': 'Odia_India.1252',
    'ori': 'Odia_India.1252',
    'pashto': 'Pashto_Afghanistan.1252',
    'pashto_afghanistan': 'Pashto_Afghanistan.1252',
    'ps': 'Pashto_Afghanistan.1252',
    'ps-af': 'Pashto_Afghanistan.1252',
    'pas': 'Pashto_Afghanistan.1252',
    'persian': 'Persian_Iran.1256',
    'persian_iran': 'Persian_Iran.1256',
    'fa': 'Persian_Iran.1256',
    'fa-ir': 'Persian_Iran.1256',
    'far': 'Persian_Iran.1256',
    'polish': 'Polish_Poland.1250',
    'polish_poland': 'Polish_Poland.1250',
    'pl': 'Polish_Poland.1250',
    'pl-pl': 'Polish_Poland.1250',
    'plk': 'Polish_Poland.1250',
    'portuguese': 'Portuguese_Brazil.1252',
    'portuguese_brazil': 'Portuguese_Brazil.1252',
    'pt': 'Portuguese_Brazil.1252',
    'pt-br': 'Portuguese_Brazil.1252',
    'ptb': 'Portuguese_Brazil.1252',
    'portuguese_portugal': 'Portuguese_Portugal.1252',
    'pt-pt': 'Portuguese_Portugal.1252',
    'ptg': 'Portuguese_Portugal.1252',
    'punjabi': 'Punjabi_India.1252',
    'punjabi_india': 'Punjabi_India.1252',
    'pa': 'Punjabi_India.1252',
    'pa-in': 'Punjabi_India.1252',
    'pan': 'Punjabi_India.1252',
    'quechua': 'Quechua_Peru.1252',
    'quechua_bolivia': 'Quechua_Bolivia.1252',
    'quz': 'Quechua_Bolivia.1252',
    'quz-bo': 'Quechua_Bolivia.1252',
    'qub': 'Quechua_Bolivia.1252',
    'quechua_peru': 'Quechua_Peru.1252',
    'quz-pe': 'Quechua_Peru.1252',
    # Note the spelling difference in Ecuador
    'que': 'Quichua_Ecuador.1252',
    'quz-ec': 'Quichua_Ecuador.1252',
    'quichua_ecuador': 'Quichua_Ecuador.1252',
    'romanian': 'Romanian_Romania.1250',
    'romanian_romania': 'Romanian_Romania.1250',
    'ro': 'Romanian_Romania.1250',
    'ro-ro': 'Romanian_Romania.1250',
    'rom': 'Romanian_Romania.1250',
    'romansh': 'Romansh_Switzerland.1252',
    'romansh_switzerland': 'Romansh_Switzerland.1252',
    'rm': 'Romansh_Switzerland.1252',
    'rm-ch': 'Romansh_Switzerland.1252',
    'rmc': 'Romansh_Switzerland.1252',
    'russian': 'Russian_Russia.1251',
    'russian_russia': 'Russian_Russia.1251',
    'ru': 'Russian_Russia.1251',
    'ru-ru': 'Russian_Russia.1251',
    'rus': 'Russian_Russia.1251',
    'Sami (inari)': 'Sami (Inari)_Finland.1252',
    'sami': 'Sami (Northern)_Sweden.1252',
    'sami (inari)_finland': 'Sami (Inari)_Finland.1252',
    'smn': 'Sami (Inari)_Finland.1252',
    'smn-fi': 'Sami (Inari)_Finland.1252',
    'Sami (Lule)': 'Sami (Lule)_Sweden.1252',
    'sami (lule)_sweden': 'Sami (Lule)_Sweden.1252',
    'smk': 'Sami (Lule)_Sweden.1252',
    'smj-se': 'Sami (Lule)_Sweden.1252',
    'sami (lule)_norway': 'Sami (Lule)_Norway.1252',
    'smj': 'Sami (Lule)_Norway.1252',
    'smj-no': 'Sami (Lule)_Norway.1252',
    'Sami (northern)': 'Sami (Northern)_Finland.1252',
    'Sami (northern)_finland': 'Sami (Northern)_Finland.1252',
    # naming inconsistency
    'northern sami': 'Northern Sami_Norway.1252',
    'northern sami_norway': 'Northern Sami_Norway.1252',
    'se-no': 'Northern Sami_Norway.1252',
    'sme': 'Northern Sami_Norway.1252',
    'sami (skolt)': 'Sami (Skolt)_Finland.1252',
    'sami (skolt)_finland': 'Sami (Skolt)_Finland.1252',
    'sms': 'Sami (Skolt)_Finland.1252',
    'se-fi': 'Sami (Skolt)_Finland.1252',
    'sms-fi': 'Sami (Skolt)_Finland.1252',
    'se': 'Sami (Southern)_Norway.1252',
    'Sami (Southern)': 'Sami (Southern)_Sweden.1252',
    'sami (southern)_sweden': 'Sami (Southern)_Sweden.1252',
    'smb': 'Sami (Southern)_Sweden.1252',
    'sma-se': 'Sami (Southern)_Sweden.1252',
    'sami (southern)_norway': 'Sami (Southern)_Norway.1252',
    'sma': 'Sami (Southern)_Norway.1252',
    'sma-no': 'Sami (Southern)_Norway.1252',
    'sami (northern)_sweden': 'Sami (Northern)_Sweden.1252',
    'se-se': 'Sami (Northern)_Sweden.1252',
    'smf': 'Sami (Northern)_Sweden.1252',
    'sanskrit': 'Sanskrit_India.1252',
    'sanskrit_india': 'Sanskrit_India.1252',
    'sa': 'Sanskrit_India.1252',
    'sa-in': 'Sanskrit_India.1252',
    'san': 'Sanskrit_India.1252',
    'scottish gaelic': 'Scottish Gaelic_United Kingdom.1252',
    'scottish gaelic_united kingdom':
    'Scottish Gaelic_United Kingdom.1252',
    'gd': 'Scottish Gaelic_United Kingdom.1252',
    'gd-gb': 'Scottish Gaelic_United Kingdom.1252',
    'gla': 'Scottish Gaelic_United Kingdom.1252',
    'serbian': 'Serbian (Latin)_Serbia.1250',
    'serbian_serbia': 'Serbian_Serbia.1250',
    'sr': 'Serbian_Serbia.1250',
    'sr-cyrl-cs': 'Serbian_Serbia.1250',
    'srb': 'Serbian_Serbia.1250',
    'Serbian (Latin)': 'Serbian (Latin)_Serbia.1250',
    'serbian (latin)_serbia': 'Serbian (Latin)_Serbia.1250',
    'srm': 'Serbian (Latin)_Serbia.1250',
    'sr-latn': 'Serbian (Latin)_Serbia.1250',
    'sr-latn-rs': 'Serbian (Latin)_Serbia.1250',
    'Serbian (Cyrillic)': 'Serbian (Cyrillic)_Serbia.1251',
    'serbian (cyrillic)_serbia': 'Serbian (Cyrillic)_Serbia.1251',
    'sr-cyrl': 'Serbian (Cyrillic)_Serbia.1251',
    'sr-cyrl-rs': 'Serbian (Cyrillic)_Serbia.1251',
    'sro': 'Serbian (Cyrillic)_Serbia.1251',
    'serbian (cyrillic)_bosnia and herzegovina':
    'Serbian (Cyrillic)_Bosnia and Herzegovina.1251',
    'sr-cyrl-ba': 'Serbian (Cyrillic)_Bosnia and Herzegovina.1251',
    'srn': 'Serbian (Cyrillic)_Bosnia and Herzegovina.1251',
    'serbian (cyrillic)_montenegro':
    'Serbian (Cyrillic)_Montenegro.1251',
    'sr-cyrl-me': 'Serbian (Cyrillic)_Montenegro.1251',
    'srq': 'Serbian (Cyrillic)_Montenegro.1251',
    'serbian (latin)_bosnia and herzegovina':
    'Serbian (Latin)_Bosnia and Herzegovina.1250',
    'sr-latn-ba': 'Serbian (Latin)_Bosnia and Herzegovina.1250',
    'srs': 'Serbian (Latin)_Bosnia and Herzegovina.1250',
    'serbian (latin)_montenegro': 'Serbian (Latin)_Montenegro.1251',
    'sr-latn-me': 'Serbian (Latin)_Montenegro.1251',
    'srp': 'Serbian (Latin)_Montenegro.1250',
    'sesotho sa leboa': 'Sesotho sa Leboa_South Africa.1252',
    'sesotho sa leboa_south africa':
    'Sesotho sa Leboa_South Africa.1252',
    'nso': 'Sesotho sa Leboa_South Africa.1252',
    'nso-za': 'Sesotho sa Leboa_South Africa.1252',
    'setswana': 'Setswana_South Africa.1252',
    'setswana_south africa': 'Setswana_South Africa.1252',
    'tn': 'Setswana_South Africa.1252',
    'tn-za': 'Setswana_South Africa.1252',
    'tsn': 'Setswana_South Africa.1252',
    'sinhala': 'Sinhala_Sri Lanka.1252',
    'sinhala_sri lanka': 'Sinhala_Sri Lanka.1252',
    'si': 'Sinhala_Sri Lanka.1252',
    'si-lk': 'Sinhala_Sri Lanka.1252',
    'sin': 'Sinhala_Sri Lanka.1252',
    'slovak': 'Slovak_Slovakia.1250',
    'slovak_slovakia': 'Slovak_Slovakia.1250',
    'sk': 'Slovak_Slovakia.1250',
    'sk-sk': 'Slovak_Slovakia.1250',
    'sky': 'Slovak_Slovakia.1250',
    'slovenian': 'Slovenian_Slovenia.1250',
    'slovenian_slovenia': 'Slovenian_Slovenia.1250',
    'sl': 'Slovenian_Slovenia.1250',
    'sl-si': 'Slovenian_Slovenia.1250',
    'slv': 'Slovenian_Slovenia.1250',
    'spanish': 'Spanish_Venezuela.1252',
    'spanish_spain': 'Spanish_Spain.1252',
    'es': 'Spanish_Spain.1252',
    'es-es': 'Spanish_Spain.1252',
    'esn': 'Spanish_Spain.1252',
    'esp': 'Spanish_Spain.1252',
    'spanish_argentina': 'Spanish_Argentina.1252',
    'es-ar': 'Spanish_Argentina.1252',
    'ess': 'Spanish_Argentina.1252',
    'spanish_bolivia': 'Spanish_Bolivia.1252',
    'es-bo': 'Spanish_Bolivia.1252',
    'esb': 'Spanish_Bolivia.1252',
    'spanish_chile': 'Spanish_Chile.1252',
    'es-cl': 'Spanish_Chile.1252',
    'esl': 'Spanish_Chile.1252',
    'spanish_colombia': 'Spanish_Colombia.1252',
    'es-co': 'Spanish_Colombia.1252',
    'eso': 'Spanish_Colombia.1252',
    'spanish_costa rica': 'Spanish_Costa Rica.1252',
    'es-cr': 'Spanish_Costa Rica.1252',
    'esc': 'Spanish_Costa Rica.1252',
    'spanish_dominican republic': 'Spanish_Dominican Republic.1252',
    'es-do': 'Spanish_Dominican Republic.1252',
    'esd': 'Spanish_Dominican Republic.1252',
    'spanish_ecuador': 'Spanish_Ecuador.1252',
    'es-ec': 'Spanish_Ecuador.1252',
    'esf': 'Spanish_Ecuador.1252',
    'spanish_el salvador': 'Spanish_El Salvador.1252',
    'es-sv': 'Spanish_El Salvador.1252',
    'ese': 'Spanish_El Salvador.1252',
    'spanish_guatemala': 'Spanish_Guatemala.1252',
    'es-gt': 'Spanish_Guatemala.1252',
    'esg': 'Spanish_Guatemala.1252',
    'spanish_honduras': 'Spanish_Honduras.1252',
    'es-hn': 'Spanish_Honduras.1252',
    'esh': 'Spanish_Honduras.1252',
    'spanish_mexico': 'Spanish_Mexico.1252',
    'es-mx': 'Spanish_Mexico.1252',
    'esm': 'Spanish_Mexico.1252',
    'spanish_nicaragua': 'Spanish_Nicaragua.1252',
    'es-ni': 'Spanish_Nicaragua.1252',
    'esi': 'Spanish_Nicaragua.1252',
    'spanish_panama': 'Spanish_Panama.1252',
    'es-pa': 'Spanish_Panama.1252',
    'esa': 'Spanish_Panama.1252',
    'spanish_paraguay': 'Spanish_Paraguay.1252',
    'es-py': 'Spanish_Paraguay.1252',
    'esz': 'Spanish_Paraguay.1252',
    'spanish_peru': 'Spanish_Peru.1252',
    'es-pe': 'Spanish_Peru.1252',
    'esr': 'Spanish_Peru.1252',
    'spanish_puerto rico': 'Spanish_Puerto Rico.1252',
    'es-pr': 'Spanish_Puerto Rico.1252',
    'esu': 'Spanish_Puerto Rico.1252',
    'spanish_united states': 'Spanish_United States.1252',
    'es-us': 'Spanish_United States.1252',
    'est': 'Spanish_United States.1252',
    'spanish_uruguay': 'Spanish_Uruguay.1252',
    'es-uy': 'Spanish_Uruguay.1252',
    'esy': 'Spanish_Uruguay.1252',
    'spanish_venezuela': 'Spanish_Venezuela.1252',
    'es-ve': 'Spanish_Venezuela.1252',
    'esv': 'Spanish_Venezuela.1252',
    'swedish': 'Swedish_Finland.1252',
    'swedish_sweden': 'Swedish_Sweden.1252',
    'sv': 'Swedish_Sweden.1252',
    'sve': 'Swedish_Sweden.1252',
    'sv-se': 'Swedish_Sweden.1252',
    'swedish_finland': 'Swedish_Finland.1252',
    'sv-fi': 'Swedish_Finland.1252',
    'svf': 'Swedish_Finland.1252',
    'syriac': 'Syriac_Syria.1252',
    'syriac_syria': 'Syriac_Syria.1252',
    'syr': 'Syriac_Syria.1252',
    'syr-sy': 'Syriac_Syria.1252',
    'tajik': 'Tajik (Cyrillic)_Tajikistan.1251',
    'tajik_tajikistan': 'Tajik_Tajikistan.1251',
    'tg': 'Tajik_Tajikistan.1251',
    'taj': 'Tajik_Tajikistan.1251',
    'Tajik (Cyrillic)': 'Tajik (Cyrillic)_Tajikistan.1251',
    'tajik (cyrillic)_tajikistan':
    'Tajik (Cyrillic)_Tajikistan.1251',
    'tg-cyrl': 'Tajik (Cyrillic)_Tajikistan.1251',
    'tg-cyrl-tj': 'Tajik (Cyrillic)_Tajikistan.1251',
    'Central Atlas Tamazight (Tifinagh)':
    'Central Atlas Tamazight (Tifinagh)_Morocco.1252',
    'central atlas tamazight':
    'Central Atlas Tamazight (Latin)_Algeria.1252',
    'central atlas tamazight (tifinagh)_morocco':
    'Central Atlas Tamazight (Tifinagh)_Morocco.1252',
    'tzm': 'Central Atlas Tamazight (Tifinagh)_Morocco.1252',
    'Central Atlas Tamazight (Latin)':
    'Central Atlas Tamazight (Latin)_Algeria.1252',
    'central atlas tamazight (latin)_algeria':
    'Central Atlas Tamazight (Latin)_Algeria.1252',
    'tzm-dza': 'Central Atlas Tamazight (Latin)_Algeria.1252',
    'tzm-latn': 'Central Atlas Tamazight (Latin)_Algeria.1252',
    'tzm-latn-dz': 'Central Atlas Tamazight (Latin)_Algeria.1252',
    'tamil': 'Tamil_India.1252',
    'tamil_india': 'Tamil_India.1252',
    'ta': 'Tamil_India.1252',
    'ta-in': 'Tamil_India.1252',
    'tam': 'Tamil_Sri Lanka.1252',
    'ta-lk': 'Tamil_Sri Lanka.1252',
    'tatar': 'Tatar_Russia.1251',
    'tatar_russia': 'Tatar_Russia.1251',
    'tt': 'Tatar_Russia.1251',
    'tt-ru': 'Tatar_Russia.1251',
    'ttt': 'Tatar_Russia.1251',
    'telugu': 'Telugu_India.1252',
    'telugu_india': 'Telugu_India.1252',
    'te': 'Telugu_India.1252',
    'te-in': 'Telugu_India.1252',
    'tel': 'Telugu_India.1252',
    'thai': 'Thai_Thailand.874',
    'thai_thailand': 'Thai_Thailand.874',
    'th': 'Thai_Thailand.874',
    'th-th': 'Thai_Thailand.874',
    'tha': 'Thai_Thailand.874',
    'tibetan': 'Tibetan_China.1252',
    'tibetan_china': 'Tibetan_China.1252',
    'bo': 'Tibetan_China.1252',
    'bo-cn': 'Tibetan_China.1252',
    'bob': 'Tibetan_China.1252',
    'turkish': 'Turkish_Turkey.1254',
    'turkish_turkey': 'Turkish_Turkey.1254',
    'tr': 'Turkish_Turkey.1254',
    'tr-tr': 'Turkish_Turkey.1254',
    'trk': 'Turkish_Turkey.1254',
    'turkmen': 'Turkmen_Turkmenistan.1250',
    'turkmen_turkmenistan': 'Turkmen_Turkmenistan.1250',
    'tk': 'Turkmen_Turkmenistan.1250',
    'tk-tm': 'Turkmen_Turkmenistan.1250',
    'tuk': 'Turkmen_Turkmenistan.1250',
    'ukrainian': 'Ukrainian_Ukraine.1251',
    'ukrainian_ukraine': 'Ukrainian_Ukraine.1251',
    'uk': 'Ukrainian_Ukraine.1251',
    'uk-ua': 'Ukrainian_Ukraine.1251',
    'ukr': 'Ukrainian_Ukraine.1251',
    'upper sorbian': 'Upper Sorbian_Germany.1252',
    'upper sorbian_germany': 'Upper Sorbian_Germany.1252',
    'hsb': 'Upper Sorbian_Germany.1252',
    'hsb-de': 'Upper Sorbian_Germany.1252',
    'urdu': 'Urdu_Pakistan.1256',
    'urdu_pakistan': 'Urdu_Pakistan.1256',
    'ur': 'Urdu_Pakistan.1256',
    'ur-pk': 'Urdu_Pakistan.1256',
    'urd': 'Urdu_Pakistan.1256',
    'uyghur': 'Uyghur_China.1256',
    'uyghur_china': 'Uyghur_China.1256',
    'ug': 'Uyghur_China.1256',
    'ug-cn': 'Uyghur_China.1256',
    'uig': 'Uyghur_China.1256',
    'Uzbek (Cyrillic)': 'Uzbek (Cyrillic)_Uzbekistan.1251',
    'uzbek': 'Uzbek (Latin)_Uzbekistan.1254',
    'uzbek (cyrillic)_uzbekistan':
    'Uzbek (Cyrillic)_Uzbekistan.1251',
    'uz-cyrl': 'Uzbek (Cyrillic)_Uzbekistan.1251',
    'uz-cyrl-uz': 'Uzbek (Cyrillic)_Uzbekistan.1251',
    'Uzbek (Latin)': 'Uzbek (Latin)_Uzbekistan.1254',
    'uzbek (latin)_uzbekistan': 'Uzbek (Latin)_Uzbekistan.1254',
    'uz': 'Uzbek (Latin)_Uzbekistan.1254',
    'uz-latn': 'Uzbek (Latin)_Uzbekistan.1254',
    'uz-latn-uz': 'Uzbek (Latin)_Uzbekistan.1254',
    'ubz': 'Uzbek (Latin)_Uzbekistan.1254',
    'vietnamese': 'Vietnamese_Vietnam.1258',
    'vietnamese_vietnam': 'Vietnamese_Vietnam.1258',
    'vi': 'Vietnamese_Vietnam.1258',
    'vi-vn': 'Vietnamese_Vietnam.1258',
    'vit': 'Vietnamese_Vietnam.1258',
    'welsh': 'Welsh_United Kingdom.1252',
    'welsh_united kingdom': 'Welsh_United Kingdom.1252',
    'cy': 'Welsh_United Kingdom.1252',
    'cy-gb': 'Welsh_United Kingdom.1252',
    'cym': 'Welsh_United Kingdom.1252',
    'wolof': 'Wolof_Senegal.1252',
    'wolof_senegal': 'Wolof_Senegal.1252',
    'wo': 'Wolof_Senegal.1252',
    'wo-sn': 'Wolof_Senegal.1252',
    'wol': 'Wolof_Senegal.1252',
    'sakha': 'Sakha_Russia.1251',
    'sakha_russia': 'Sakha_Russia.1251',
    'sah': 'Sakha_Russia.1251',
    'sah-ru': 'Sakha_Russia.1251',
    'yi': 'Yi_China.1252',
    'yi_china': 'Yi_China.1252',
    'ii': 'Yi_China.1252',
    'ii-cn': 'Yi_China.1252',
    'iii': 'Yi_China.1252',
    'yoruba': 'Yoruba_Nigeria.1252',
    'yoruba_nigeria': 'Yoruba_Nigeria.1252',
    'yo': 'Yoruba_Nigeria.1252',
    'yo-ng': 'Yoruba_Nigeria.1252',
    'yor': 'Yoruba_Nigeria.1252'
}

win_locale_alias = copy.deepcopy(win_nls_locales)

# append non-NLS mapped pages
for (name, qualified_locale) in win_non_nls_locales.items():
    win_locale_alias[name] = qualified_locale


# Windows specific codepages. These are codepages
# which are supported by Windows which also have a relevant
# mapping in Python. Most of the unmapped codepages are EBCDIC,
# but also some of the ISO-2022 pages which didn't have a clear mapping
# to the codecs available.

# https://msdn.microsoft.com/en-us/library/windows/desktop/dd317756(v=vs.85).aspx

win_codepages = {
    '37': 'cp37',
    '437': 'cp437',
    '500': 'cp500',
    '720': 'cp720',
    '775': 'cp775',
    '737': 'cp737',
    '850': 'cp850',
    '852': 'cp852',
    '855': 'cp855',
    '857': 'cp857',
    '858': 'cp858',
    '860': 'cp860',
    '861': 'cp861',
    '862': 'cp862',
    '863': 'cp863',
    '864': 'cp864',
    '865': 'cp865',
    '866': 'cp866',
    '869': 'cp869',
    '874': 'cp874',
    '875': 'cp875',
    '932': 'cp932',
    '936': 'gbk',
    '949': 'cp949',
    '950': 'cp950',
    '1026': 'cp1026',
    '1140': 'cp1140',
    '1250': 'cp1250',
    '1251': 'cp1251',
    '1252': 'cp1252',
    '1253': 'cp1253',
    '1254': 'cp1254',
    '1255': 'cp1255',
    '1256': 'cp1256',
    '1257': 'cp1257',
    '1258': 'cp1258',
    '1361': 'jobhab',
    '10000': 'mac_roman',
    '10001': 'shift_jis',
    '10002': 'big5',
    '10003': 'euc_kr',
    '10004': 'mac_arabic',
    '10006': 'mac_greek',
    '10007': 'mac_cyrillic',
    '10008': 'gb2312',
    '10029': 'mac_latin2',
    '10079': 'mac_iceland',
    '10081': 'mac_turkish',
    '10082': 'mac_croatian',
    '20127': 'ascii',
    '20273': 'cp273',
    '20424': 'cp424',
    '20866': 'koi8_r',
    '20932': 'euc_jp',
    '20936': 'gb2312',
    '21866': 'koi8_u',
    '28591': 'iso8859_1',
    '28592': 'iso8859_2',
    '28593': 'iso8859_3',
    '28594': 'iso8859_4',
    '28595': 'iso8859_5',
    '28596': 'iso8859_6',
    '28597': 'iso8859_7',
    '28598': 'iso8859_8',
    '28599': 'iso8859_9',
    '28603': 'iso8859_13',
    '28605': 'iso8859_15',
    '50220': 'iso2022_jp',
    '50225': 'iso2022_kr',
    '51932': 'euc_jp',
    '52936': 'gb2312',
    '54936': 'gp18030',
    '65000': 'utf_7',
    '65001': 'utf_8'
}

#
# This maps Windows language identifiers to locale strings.
#
# This list has been updated from
# http://web.archive.org/web/20111105093934/http://msdn.microsoft.com:80/en-us/library/dd318693
# to include every locale up to Windows Vista.
#
# NOTE: this mapping is incomplete.  If your language is missing, please
# submit a bug report to the Python bug tracker at http://bugs.python.org/
# Make sure you include the missing language identifier and the suggested
# locale code.
#

windows_locale = {
    0x0436: "af_ZA", # Afrikaans
    0x041c: "sq_AL", # Albanian
    0x0484: "gsw_FR",# Alsatian - France
    0x045e: "am_ET", # Amharic - Ethiopia
    0x0401: "ar_SA", # Arabic - Saudi Arabia
    0x0801: "ar_IQ", # Arabic - Iraq
    0x0c01: "ar_EG", # Arabic - Egypt
    0x1001: "ar_LY", # Arabic - Libya
    0x1401: "ar_DZ", # Arabic - Algeria
    0x1801: "ar_MA", # Arabic - Morocco
    0x1c01: "ar_TN", # Arabic - Tunisia
    0x2001: "ar_OM", # Arabic - Oman
    0x2401: "ar_YE", # Arabic - Yemen
    0x2801: "ar_SY", # Arabic - Syria
    0x2c01: "ar_JO", # Arabic - Jordan
    0x3001: "ar_LB", # Arabic - Lebanon
    0x3401: "ar_KW", # Arabic - Kuwait
    0x3801: "ar_AE", # Arabic - United Arab Emirates
    0x3c01: "ar_BH", # Arabic - Bahrain
    0x4001: "ar_QA", # Arabic - Qatar
    0x042b: "hy_AM", # Armenian
    0x044d: "as_IN", # Assamese - India
    0x042c: "az_AZ", # Azeri - Latin
    0x082c: "az_AZ", # Azeri - Cyrillic
    0x046d: "ba_RU", # Bashkir
    0x042d: "eu_ES", # Basque - Russia
    0x0423: "be_BY", # Belarusian
    0x0445: "bn_IN", # Begali
    0x201a: "bs_BA", # Bosnian - Cyrillic
    0x141a: "bs_BA", # Bosnian - Latin
    0x047e: "br_FR", # Breton - France
    0x0402: "bg_BG", # Bulgarian
#    0x0455: "my_MM", # Burmese - Not supported
    0x0403: "ca_ES", # Catalan
    0x0004: "zh_CHS",# Chinese - Simplified
    0x0404: "zh_TW", # Chinese - Taiwan
    0x0804: "zh_CN", # Chinese - PRC
    0x0c04: "zh_HK", # Chinese - Hong Kong S.A.R.
    0x1004: "zh_SG", # Chinese - Singapore
    0x1404: "zh_MO", # Chinese - Macao S.A.R.
    0x7c04: "zh_CHT",# Chinese - Traditional
    0x0483: "co_FR", # Corsican - France
    0x041a: "hr_HR", # Croatian
    0x101a: "hr_BA", # Croatian - Bosnia
    0x0405: "cs_CZ", # Czech
    0x0406: "da_DK", # Danish
    0x048c: "gbz_AF",# Dari - Afghanistan
    0x0465: "div_MV",# Divehi - Maldives
    0x0413: "nl_NL", # Dutch - The Netherlands
    0x0813: "nl_BE", # Dutch - Belgium
    0x0409: "en_US", # English - United States
    0x0809: "en_GB", # English - United Kingdom
    0x0c09: "en_AU", # English - Australia
    0x1009: "en_CA", # English - Canada
    0x1409: "en_NZ", # English - New Zealand
    0x1809: "en_IE", # English - Ireland
    0x1c09: "en_ZA", # English - South Africa
    0x2009: "en_JA", # English - Jamaica
    0x2409: "en_CB", # English - Caribbean
    0x2809: "en_BZ", # English - Belize
    0x2c09: "en_TT", # English - Trinidad
    0x3009: "en_ZW", # English - Zimbabwe
    0x3409: "en_PH", # English - Philippines
    0x4009: "en_IN", # English - India
    0x4409: "en_MY", # English - Malaysia
    0x4809: "en_IN", # English - Singapore
    0x0425: "et_EE", # Estonian
    0x0438: "fo_FO", # Faroese
    0x0464: "fil_PH",# Filipino
    0x040b: "fi_FI", # Finnish
    0x040c: "fr_FR", # French - France
    0x080c: "fr_BE", # French - Belgium
    0x0c0c: "fr_CA", # French - Canada
    0x100c: "fr_CH", # French - Switzerland
    0x140c: "fr_LU", # French - Luxembourg
    0x180c: "fr_MC", # French - Monaco
    0x0462: "fy_NL", # Frisian - Netherlands
    0x0456: "gl_ES", # Galician
    0x0437: "ka_GE", # Georgian
    0x0407: "de_DE", # German - Germany
    0x0807: "de_CH", # German - Switzerland
    0x0c07: "de_AT", # German - Austria
    0x1007: "de_LU", # German - Luxembourg
    0x1407: "de_LI", # German - Liechtenstein
    0x0408: "el_GR", # Greek
    0x046f: "kl_GL", # Greenlandic - Greenland
    0x0447: "gu_IN", # Gujarati
    0x0468: "ha_NG", # Hausa - Latin
    0x040d: "he_IL", # Hebrew
    0x0439: "hi_IN", # Hindi
    0x040e: "hu_HU", # Hungarian
    0x040f: "is_IS", # Icelandic
    0x0421: "id_ID", # Indonesian
    0x045d: "iu_CA", # Inuktitut - Syllabics
    0x085d: "iu_CA", # Inuktitut - Latin
    0x083c: "ga_IE", # Irish - Ireland
    0x0410: "it_IT", # Italian - Italy
    0x0810: "it_CH", # Italian - Switzerland
    0x0411: "ja_JP", # Japanese
    0x044b: "kn_IN", # Kannada - India
    0x043f: "kk_KZ", # Kazakh
    0x0453: "kh_KH", # Khmer - Cambodia
    0x0486: "qut_GT",# K'iche - Guatemala
    0x0487: "rw_RW", # Kinyarwanda - Rwanda
    0x0457: "kok_IN",# Konkani
    0x0412: "ko_KR", # Korean
    0x0440: "ky_KG", # Kyrgyz
    0x0454: "lo_LA", # Lao - Lao PDR
    0x0426: "lv_LV", # Latvian
    0x0427: "lt_LT", # Lithuanian
    0x082e: "dsb_DE",# Lower Sorbian - Germany
    0x046e: "lb_LU", # Luxembourgish
    0x042f: "mk_MK", # FYROM Macedonian
    0x043e: "ms_MY", # Malay - Malaysia
    0x083e: "ms_BN", # Malay - Brunei Darussalam
    0x044c: "ml_IN", # Malayalam - India
    0x043a: "mt_MT", # Maltese
    0x0481: "mi_NZ", # Maori
    0x047a: "arn_CL",# Mapudungun
    0x044e: "mr_IN", # Marathi
    0x047c: "moh_CA",# Mohawk - Canada
    0x0450: "mn_MN", # Mongolian - Cyrillic
    0x0850: "mn_CN", # Mongolian - PRC
    0x0461: "ne_NP", # Nepali
    0x0414: "nb_NO", # Norwegian - Bokmal
    0x0814: "nn_NO", # Norwegian - Nynorsk
    0x0482: "oc_FR", # Occitan - France
    0x0448: "or_IN", # Oriya - India
    0x0463: "ps_AF", # Pashto - Afghanistan
    0x0429: "fa_IR", # Persian
    0x0415: "pl_PL", # Polish
    0x0416: "pt_BR", # Portuguese - Brazil
    0x0816: "pt_PT", # Portuguese - Portugal
    0x0446: "pa_IN", # Punjabi
    0x046b: "quz_BO",# Quechua (Bolivia)
    0x086b: "quz_EC",# Quechua (Ecuador)
    0x0c6b: "quz_PE",# Quechua (Peru)
    0x0418: "ro_RO", # Romanian - Romania
    0x0417: "rm_CH", # Romansh
    0x0419: "ru_RU", # Russian
    0x243b: "smn_FI",# Sami Finland
    0x103b: "smj_NO",# Sami Norway
    0x143b: "smj_SE",# Sami Sweden
    0x043b: "se_NO", # Sami Northern Norway
    0x083b: "se_SE", # Sami Northern Sweden
    0x0c3b: "se_FI", # Sami Northern Finland
    0x203b: "sms_FI",# Sami Skolt
    0x183b: "sma_NO",# Sami Southern Norway
    0x1c3b: "sma_SE",# Sami Southern Sweden
    0x044f: "sa_IN", # Sanskrit
    0x0c1a: "sr_SP", # Serbian - Cyrillic
    0x1c1a: "sr_BA", # Serbian - Bosnia Cyrillic
    0x081a: "sr_SP", # Serbian - Latin
    0x181a: "sr_BA", # Serbian - Bosnia Latin
    0x045b: "si_LK", # Sinhala - Sri Lanka
    0x046c: "ns_ZA", # Northern Sotho
    0x0432: "tn_ZA", # Setswana - Southern Africa
    0x041b: "sk_SK", # Slovak
    0x0424: "sl_SI", # Slovenian
    0x040a: "es_ES", # Spanish - Spain
    0x080a: "es_MX", # Spanish - Mexico
    0x0c0a: "es_ES", # Spanish - Spain (Modern)
    0x100a: "es_GT", # Spanish - Guatemala
    0x140a: "es_CR", # Spanish - Costa Rica
    0x180a: "es_PA", # Spanish - Panama
    0x1c0a: "es_DO", # Spanish - Dominican Republic
    0x200a: "es_VE", # Spanish - Venezuela
    0x240a: "es_CO", # Spanish - Colombia
    0x280a: "es_PE", # Spanish - Peru
    0x2c0a: "es_AR", # Spanish - Argentina
    0x300a: "es_EC", # Spanish - Ecuador
    0x340a: "es_CL", # Spanish - Chile
    0x380a: "es_UR", # Spanish - Uruguay
    0x3c0a: "es_PY", # Spanish - Paraguay
    0x400a: "es_BO", # Spanish - Bolivia
    0x440a: "es_SV", # Spanish - El Salvador
    0x480a: "es_HN", # Spanish - Honduras
    0x4c0a: "es_NI", # Spanish - Nicaragua
    0x500a: "es_PR", # Spanish - Puerto Rico
    0x540a: "es_US", # Spanish - United States
#    0x0430: "", # Sutu - Not supported
    0x0441: "sw_KE", # Swahili
    0x041d: "sv_SE", # Swedish - Sweden
    0x081d: "sv_FI", # Swedish - Finland
    0x045a: "syr_SY",# Syriac
    0x0428: "tg_TJ", # Tajik - Cyrillic
    0x085f: "tmz_DZ",# Tamazight - Latin
    0x0449: "ta_IN", # Tamil
    0x0444: "tt_RU", # Tatar
    0x044a: "te_IN", # Telugu
    0x041e: "th_TH", # Thai
    0x0851: "bo_BT", # Tibetan - Bhutan
    0x0451: "bo_CN", # Tibetan - PRC
    0x041f: "tr_TR", # Turkish
    0x0442: "tk_TM", # Turkmen - Cyrillic
    0x0480: "ug_CN", # Uighur - Arabic
    0x0422: "uk_UA", # Ukrainian
    0x042e: "wen_DE",# Upper Sorbian - Germany
    0x0420: "ur_PK", # Urdu
    0x0820: "ur_IN", # Urdu - India
    0x0443: "uz_UZ", # Uzbek - Latin
    0x0843: "uz_UZ", # Uzbek - Cyrillic
    0x042a: "vi_VN", # Vietnamese
    0x0452: "cy_GB", # Welsh
    0x0488: "wo_SN", # Wolof - Senegal
    0x0434: "xh_ZA", # Xhosa - South Africa
    0x0485: "sah_RU",# Yakut - Cyrillic
    0x0478: "ii_CN", # Yi - PRC
    0x046a: "yo_NG", # Yoruba - Nigeria
    0x0435: "zu_ZA", # Zulu
}

def _print_locale():

    """ Test function.
    """
    categories = {}
    def _init_categories(categories=categories):
        for k,v in globals().items():
            if k[:3] == 'LC_':
                categories[k] = v
    _init_categories()
    del categories['LC_ALL']

    print('Locale defaults as determined by getdefaultlocale():')
    print('-'*72)
    lang, enc = getdefaultlocale()
    print('Language: ', lang or '(undefined)')
    print('Encoding: ', enc or '(undefined)')
    print()

    print('Locale settings on startup:')
    print('-'*72)
    for name,category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()

    print()
    print('Locale settings after calling resetlocale():')
    print('-'*72)
    resetlocale()
    for name,category in categories.items():
        print(name, '...')
        lang, enc = getlocale(category)
        print('   Language: ', lang or '(undefined)')
        print('   Encoding: ', enc or '(undefined)')
        print()

    try:
        setlocale(LC_ALL, "")
    except:
        print('NOTE:')
        print('setlocale(LC_ALL, "") does not support the default locale')
        print('given in the OS environment variables.')
    else:
        print()
        print('Locale settings after calling setlocale(LC_ALL, ""):')
        print('-'*72)
        for name,category in categories.items():
            print(name, '...')
            lang, enc = getlocale(category)
            print('   Language: ', lang or '(undefined)')
            print('   Encoding: ', enc or '(undefined)')
            print()

###

try:
    LC_MESSAGES
except NameError:
    pass
else:
    __all__.append("LC_MESSAGES")

if __name__=='__main__':
    print('Locale aliasing:')
    print()
    _print_locale()
    print()
    print('Number formatting:')
    print()
    _test()
