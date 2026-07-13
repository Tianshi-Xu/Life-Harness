# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from typing import Optional

import textworld

import textworld.envs
from textworld.core import EnvInfos, Wrapper


class GenericEnvironment(Wrapper):

    def __init__(self, infos: Optional[EnvInfos] = None) -> None:
        super().__init__()
        self.infos = infos
        self._last_backend = None

    def load(self, gamefile: str) -> None:
        backend = textworld.envs._guess_backend(gamefile)
        if self._last_backend != backend:
            self._wrap(textworld.start(gamefile, self.infos))
            self._last_backend = backend
        else:
            self._wrapped_env.load(gamefile)
