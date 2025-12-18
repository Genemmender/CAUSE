import string


class Calc:
    @classmethod
    def parse(cls, s: str):
        allow_functions = ['int', 'float']

        orig_string = s
        for func in allow_functions:
            s = s.replace(func, '')

        allow_chars = string.digits + '(). +-*/'
        string_set = set(s)
        for char in string_set:
            if char not in allow_chars:
                raise ValueError(f'Invalid calculation: {orig_string}')

        return eval(orig_string)
