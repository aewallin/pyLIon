import h5py
import signal
import pexpect
import jinja2 as j2
import json
import inspect
from datetime import datetime
import sys
import time

__version__ = '0.3.2'


class SimulationError(Exception):
    """Custom error class for Simulation."""
    pass


class Attributes:
    """Container for attributes with h5 friendly json values."""

    def __setattr__(self, name, value):
        super().__setattr__(name, json.dumps(value))

    def save(self, filename):
        with h5py.File(filename, 'r+') as f:
            print(f'Saving attributes to {filename}', self.__dict__)
            f.attrs.update(self.__dict__)

    def load(self, filename):
        with h5py.File(filename, 'r') as f:
            return {k: json.loads(v) for k, v in f.attrs.items()}


class Simulation(list):

    def __init__(self, name='pylion'):
        super().__init__()

        # keep track of uids for list function overrides
        self._uids = []

        # slugify 'name' to use for filename
        name = name.replace(' ', '_').lower()

        self.attrs = {}
        self.attrs['executable'] = '/Applications/lammps-31Mar17/src/lmp_serial'
        self.attrs['timestep'] = 1e-6
        self.attrs['domain'] = [1e-3, 1e-3, 1e-3]  # length, width, height
        self.attrs['name'] = name
        self.attrs['neighbour'] = {'skin': 1, 'list': 'nsq'}
        self.attrs['coulombcutoff'] = 10
        self.attrs['template'] = 'simulation.j2'

        # make the h5 file so all other operations can append
        with h5py.File(self.attrs['name'] + '.h5', 'w') as f:
            pass

    def _saveattrs(self):
        with h5py.File(self.attrs['name'] + '.h5', 'r+') as f:
            # serialise them before saving so that h5 is happy no matter
            # what you throw at it
            f.attrs.update({k: json.dumps(v)
                            for k, v in self.attrs.items()})

    def _loadattrs(self):
        with h5py.File(self.attrs['name'] + '.h5', 'r') as f:
            return {k: json.loads(v) for k, v in f.attrs.items()}

    def __contains__(self, this):
        # raise SimulationError("Element does not have 'uid' key.")
        return this['uid'] in self._uids

    def append(self, this):
        # only allow for dicts in the list
        assert isinstance(this, dict)
        try:
            self._uids.append(this['uid'])
            # ions will always be included first so to sort you have
            # to give 1-count 'priority' keys to the rest
            if this.get('type') == 'ions':
                this['priority'] = 0
        except KeyError:
            # append None to make sure len(self._uids) == len(self.data)
            self._uids.append(None)

        timestep = this.get('timestep', 1e12)
        if timestep < self.attrs['timestep']:
            print(f'Reducing timestep to {timestep} sec')
            self.attrs['timestep'] = timestep

        super().append(this)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def index(self, this):
        return self._uids.index(this['uid'])

    def remove(self, this):
        # use del if you really want to delete something or better yet don't
        # add it to the simulation in the first place
        code = ['\n# Deleting a fix', f"unfix {this['uid']}\n"]
        self.append({'code': code, 'type': 'command'})

    def sort(self):
        # sort with 'priority' keys if found otherwise do nothing
        try:
            super().sort(key=lambda item: item['priority'])
        except KeyError:
            pass
            # Not all elements have 'priority' keys. Cannot sort list

    def _writeinputfile(self):
        self.attrs['version'] = __version__
        self.attrs.setdefault('rigid', {'exists': False})

        self.sort()  # if 'priority' keys exist

        odict = {key: [] for key in ['species', 'simulation']}
        for idx, item in enumerate(self):
            if item.get('type') == 'ions':
                odict['species'].append(item)
                if item.get('rigid'):
                    self.attrs['rigid'] = {'exists': True}
                    self.attrs['rigid'].setdefault('groups', []).append(idx+1)
            else:
                odict['simulation'].append(item)

        # do a couple of checks
        # check for uids clashing
        uids = list(filter(None.__ne__, self._uids))
        if len(uids) > len(set(uids)):
            raise SimulationError(
                "There are identical 'uids'. Although this is allowed in some "
                " cases, 'lammps' is probably not going to like it.")

        # make sure species will behave
        maxuid = max(odict['species'], key=lambda item: item['uid'])['uid']
        if maxuid > len(odict['species']):
            raise SimulationError(
                "Max 'uid' of species is larger than the number of species. "
                "Calling '@lammps.ions' decorated functions "
                "always increments the 'uid' count.")

        # load jinja2 template
        env = j2.Environment(loader=j2.PackageLoader('pylion', 'templates'),
                             trim_blocks=True)
        template = env.get_template(self.attrs['template'])
        rendered = template.render({**self.attrs, **odict})

        with open(self.attrs['name'] + '.lammps', 'w') as f:
            f.write(rendered)

        # get a few more attrs
        self.attrs['time'] = datetime.now().isoformat()

        # and the name of the output files
        fixes = filter(lambda item: item.get('type') == 'fix',
                       odict['simulation'])
        self.attrs['output_files'] = [line.split()[5] for fix in fixes
                                      for line in fix['code']
                                      if line.startswith('dump')]

        # save attrs and scripts to h5 file
        self._saveattrs()
        self._savecallersource()
        self._savescriptsource(self.attrs['name'] + '.lammps')

        # give it some time to write everything to the h5 file
        time.sleep(1)

    def execute(self):

        if getattr(self, '_hasexecuted', False):
            print('Simulation has executed already. Do not run it again.')
            return

        self._writeinputfile()

        def signal_handler(sig, frame):
            print('Simulation terminated by the user.')
            child.terminate()
            # sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        child = pexpect.spawn(' '.join([self.attrs['executable'], '-in',
                              self.attrs['name'] + '.lammps']), timeout=30,
                              encoding='utf8')

        self._process_stdout(child)
        child.close()

        self._hasexecuted = True

        for filename in self.attrs['output_files'] + ['log.lammps']:
            self._savescriptsource(filename)

    def _process_stdout(self, child):
        atoms = 0
        for line in child:
            line = line.rstrip('\r\n')
            if line == 'Created 1 atoms':
                atoms += 1
                continue
            elif line == 'Created 0 atoms':
                raise SimulationError(
                    'lammps created 0 atoms - perhaps you placed ions '
                    'with positions outside the simulation domain?')

            if atoms:
                print(f'Created {atoms} atoms.')
                atoms = False
                continue

            print(line)

    def _savescriptsource(self, script):
        with h5py.File(self.attrs['name'] + '.h5', 'r+') as f:
            with open(script, 'rb') as pf:
                lines = pf.readlines()
                f.create_dataset(script, data=lines)

    def _savecallersource(self):
        # inspect the first four frames of the stack to find the correct
        # filename. This covers calling from execute() or _writeinputfile().
        # if the stack is indeed larger than this it's probably the REPL.
        stack = inspect.stack()[:4]
        for frame in stack:
            if sys.argv[0] == frame.filename:
                self._savescriptsource(frame.filename)
                return

        # cannot save on the h5 file if using the repl
        print('Caller source not saved. '
              'Are you running the simulation from the repl?')
