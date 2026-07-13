# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


# -*- coding: utf-8 -*-
import os
import re
import sys
import textwrap

from typing import Mapping, Union, Tuple, List

import numpy as np

from io import StringIO

import textworld
from textworld.core import GameState
from textworld.core import GameNotRunningError
from textworld.generator.game2 import Game, GameProgression
from textworld.generator.inform7 import Inform7Game


class TextWorldEnv(textworld.Environment):
    """
    Environment for playing games by TextWorld.
    """

    def load(self, game: textworld.Game) -> None:
        self._game = game
        self._game_progression = None

    def _gather_infos(self):
        self.state["game"] = self._game
        self.state["command_templates"] = self._game.command_templates
        self.state["verbs"] = self._game.verbs
        self.state["entities"] = self._game.entity_names
        self.state["objective"] = self._game.objective
        self.state["max_score"] = self._game.max_score

        # for k, v in self._game.extras.items():
        #     self.state["extra.{}".format(k)] = v

        self.state["_game_progression"] = self._game_progression
        self.state["_facts"] = list(self._game_progression.state.facts)

        self.state["won"] = self._game_progression.state.check_goal()
        self.state["lost"] = False  # Can an ALFRED game be lost?

        self.state["_winning_policy"] = self._current_winning_policy
        # if self.infos.policy_commands:
        #     self.state["policy_commands"] = []
        #     if self._game_progression.winning_policy is not None:
        #         self.state["policy_commands"] = self._inform7.gen_commands_from_actions(self._current_winning_policy)

        if self.infos.intermediate_reward:
            self.state["intermediate_reward"] = 0
            if self.state["won"]:
                # The last action led to winning the game.
                self.state["intermediate_reward"] = 1

            elif self.state["lost"]:
                # The last action led to losing the game.
                self.state["intermediate_reward"] = -1

            elif self._previous_winning_policy is None:
                self.state["intermediate_reward"] = 0

            else:
                diff = len(self._previous_winning_policy) - len(self._current_winning_policy)
                self.state["intermediate_reward"] = int(diff > 0) - int(diff < 0)  # Sign function.

        # if self.infos.facts:
        #     self.state["facts"] = list(map(self._inform7.get_human_readable_fact, self.state["_facts"]))

        self.state["last_action"] = None
        self.state["_last_action"] = self._last_action
        # if self.infos.last_action and self._last_action is not None:
        #     self.state["last_action"] = self._inform7.get_human_readable_action(self._last_action)

        self.state["_valid_actions"] = self._game_progression.valid_actions

        mapping = {k: info.name for k, info in self._game.infos.items()}
        self.state["_valid_commands"] = []
        for action in self._game_progression.valid_actions:

            context = {
                "state": self._game_progression.state,
                "facts": list(self._game_progression.state.facts),
                "variables": {ph.name: self._game.infos[var.name] for ph, var in action.mapping.items()},
                "mapping": action.mapping,
                "entity_infos": self._game.infos,
            }
            backup = action.command_template
            action.command_template = self._game.state._logic.grammar.derive(action.command_template, context)
            #print(action.command_template)
            self.state["_valid_commands"].append(action.format_command(mapping))

        # self.state["_valid_commands"] = [a.format_command(mapping) for a in self._game_progression.valid_actions]

        # To guarantee the order from one execution to another, we sort the commands.
        # Remove any potential duplicate commands (they would lead to the same result anyway).
        self.state["admissible_commands"] = sorted(set(self.state["_valid_commands"]))

        if self.infos.moves:
            self.state["moves"] = self._moves

    def reset(self):
        self.prev_state = None
        self.state = GameState()
        track_quests = (self.infos.intermediate_reward or self.infos.policy_commands)
        self._game_progression = GameProgression(self._game, track_quests=track_quests)
        self._last_action = None
        self._previous_winning_policy = None
        self._current_winning_policy = self._game_progression.winning_policy
        self._moves = 0

        context = {
            "state": self._game_progression.state,
            "facts": list(self._game.state.facts),
            "variables": {},
            "mapping": {}, #dict(self._game.kb.types.constants_mapping),
            "entity_infos": self._game.infos,
        }

        self.state.feedback = self._game.state._logic.grammar.derive("#intro#", context)
        self.state.raw = self.state.feedback
        self._gather_infos()
        return self.state

    def step(self, command: str):
        command = command.strip()
        self.prev_state = self.state

        self.state = GameState()
        self.state.last_command = command
        self._previous_winning_policy = self._current_winning_policy

        self._last_action = None
        try:
            # Find the action corresponding to the command.
            idx = self.prev_state["_valid_commands"].index(command)
            self._last_action = self._game_progression.valid_actions[idx]
            # An action that affects the state of the game.
            self.state.effects = self._game_progression.update(self._last_action)
            self._current_winning_policy = self._game_progression.winning_policy

            context = {
                "state": self._game_progression.state,
                "facts": list(self._game_progression.state.facts),
                "variables": {ph.name: self._game.infos[var.name] for ph, var in self._last_action.mapping.items()},
                "mapping": self._last_action.mapping,
                "entity_infos": self._game.infos,
            }

            self.state.feedback = self._game.state._logic.grammar.derive(self._last_action.feedback_rule, context)
            self._moves += 1
        except ValueError:
            # TODO: handle error messages
            self.state.effects = []
            self.state.feedback = "Nothing happens."
            # print("Unknown command: {}".format(command))
            pass  # We assume nothing happened in the game.

        self.state.raw = self.state.feedback
        self._gather_infos()
        self.state["score"] = self._game_progression
        self.state["done"] = self.state["won"] or self.state["lost"]
        return self.state, self.state["score"], self.state["done"]

    def copy(self) -> "TextWorldEnv":
        """ Return a soft copy. """
        env = TextWorldEnv()

        env.state = self.state
        env.infos = self.infos

        env._gamefile = self._gamefile
        env._game = self._game
        env._inform7 = self._inform7

        env.prev_state = self.prev_state
        env._last_action = self._last_action
        env._previous_winning_policy = self._previous_winning_policy
        env._current_winning_policy = self._current_winning_policy
        env._moves = self._moves
        env._game_progression = self._game_progression.copy()
        return env

    def render(self, mode: str = "human") -> None:
        outfile = StringIO() if mode in ['ansi', "text"] else sys.stdout

        msg = self.state.feedback.rstrip() + "\n"
        if self.display_command_during_render and self.state.last_command is not None:
            msg = '> ' + self.state.last_command + "\n" + msg

        # Wrap each paragraph.
        if mode == "human":
            paragraphs = msg.split("\n")
            paragraphs = ["\n".join(textwrap.wrap(paragraph, width=80)) for paragraph in paragraphs]
            msg = "\n".join(paragraphs)

        outfile.write(msg + "\n")

        if mode == "text":
            outfile.seek(0)
            return outfile.read()

        if mode == 'ansi':
            return outfile
