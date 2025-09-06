import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()

    def register_state(self, name, initial_value):
        self._state_names.append(name)
        setattr(self, name, initial_value)

    def states(self):
        for name in self._state_names:
            yield name, getattr(self, name)
        for m in self.children():
            if isinstance(m, Module):
                for state in m.states():
                    yield state

    def reset_states(self):
        for name in self._state_names:
            setattr(self, name, self._state_defaults[name])
        for m in self.children():
            if isinstance(m, Module):
                m.reset_states()

class ModuleList(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleList, self).__init__(modules)

    def reset_states(self):
        for m in self.children():
            if isinstance(m, Module):
                m.reset_states()