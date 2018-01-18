from .utils import validate_id, _unique_id, pretty_repr
import functools


@pretty_repr
class CfgObject:
    def __init__(self, func, lmp_type, required_keys=None):

        self.func = func

        # use default keys and update if there is anything else
        # __call__ will overwrite code except for ions
        self.odict = dict.fromkeys(('code', 'type'), [])
        self.odict['type'] = lmp_type
        if required_keys:
            self.odict.update(dict.fromkeys(required_keys))

        # add dunder attrs from func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):

        func = self.func

        if getattr(self, '_unique_id', False):
            uid = _unique_id(func, *args)
            self.odict['uid'] = uid
            func = functools.partial(self.func, uid)

        self.odict.update(func(*args, **kwargs))
        if not isinstance(self.odict['code'], list):
            raise TypeError("'code' should be a list of strings.")

        return self.odict.copy()


class Ions(CfgObject):
    # need to handle this in the class namespace
    _ids = set()

    def __call__(self, *args, **kwargs):
        self.odict = super().__call__(*args, **kwargs)

        # if function, charge, mass and rigid are the same it's probably the
        # same ions definition. Don't increment the set count.
        charge, mass = self.odict['charge'], self.odict['mass']
        rigid = self.odict.get('rigid', False)

        uid = _unique_id(self.func, charge, mass, rigid)
        Ions._ids.add(uid)

        self.odict['uid'] = len(Ions._ids)

        return self.odict.copy()


class Variable(CfgObject):

    def __call__(self, *args, **kwargs):
        # only support fix type variables
        # var type variables are easier to add with custom code

        vs = kwargs['variables']
        allowed = {'id', 'x', 'y', 'z', 'vx', 'vy', 'vz'}
        if not set(vs).issubset(allowed):
            prefix = [item.startswith('v_') for item in vs]
            if not all(prefix):
                raise TypeError(
                    f'Use only {allowed} as variables or previously defined '
                    "variables with the prefix 'v_'.")

        self.odict = super().__call__(*args, **kwargs)

        # vtype can only be 'fix' or 'var'
        # prefix = {'fix': 'f_', 'var': 'v_'}
        # vtype - self.odict['vtype']

        # name = self.odict['uid']

        # this is not necessary anymore
        # output = ' '.join([f'{prefix[vtype]}{name}[{i}]'
        #                    for i in range(1, len(vs))])

        # self.odict.update({'output': output})

        return self.odict.copy()


class lammps:

    @validate_id
    def fix(func):
        return CfgObject(func, 'fix')

    def command(func):
        return CfgObject(func, 'command')

    # def group(func):
    #     return CfgObject(func, 'group')

    def variable(vtype):
        @validate_id
        # @validate_vars  # todo need kwarg variables?
        def decorator(func):
            return Variable(func, 'variable',
                            required_keys=['output', 'vtype'])
        return decorator

    def ions(func):
        return Ions(func, 'ions',
                    required_keys=['charge', 'mass', 'positions'])
