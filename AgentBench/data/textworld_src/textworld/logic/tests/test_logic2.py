import textworld

from textworld.logic.logic2 import GameLogic, _parse_and_convert
from textworld.envs.tw import TextWorldEnv


def build_game(kb):
    options = textworld.GameOptions()
    options.seed = 1234
    options.kb = kb
    options.grammar.theme = "new"

    M = textworld.GameMaker(options)

    # Create a room.
    kitchen = M.new_room("kitchen")
    M.set_player(kitchen)

    # Add a supporter to the kitchen.
    supporter = M.new(type='s', name='table')
    supporter.infos.indefinite = "a"
    M.add_fact("supporter", supporter)
    kitchen.add(supporter)

    # Add a container to the kitchen.
    container = M.new(type='c', name='chest')
    container.infos.indefinite = "a"
    M.add_fact("container", container)
    kitchen.add(container)
    M.add_fact("open", container)
    fridge = M.new(type='c', name='fridge')
    fridge.infos.indefinite = "a"
    kitchen.add(container, fridge)
    M.add_fact("container", fridge)
    M.add_fact("closed", fridge)

    # Add some item in the fridge.
    milk = M.new(type='f', name="glass of milk")
    milk.infos.indefinite = "a"
    M.new_fact("portable", milk)
    M.new_fact("drinkable", milk)
    fridge.add(milk)

    # Add some item in the inventory.
    apple = M.new(type='f', name="apple")
    apple.infos.indefinite = "an"
    M.new_fact("portable", apple)
    M.new_fact("eatable", apple)

    fork = M.new(type='o', name="fork")
    fork.infos.indefinite = "a"
    M.new_fact("portable", apple)

    M.inventory.add(apple, fork)

    # Add a bathroom.
    bathroom = M.new_room("bathroom")

    path = M.connect(kitchen.north, bathroom.south)
    door = M.new_door(path, name="glass door")
    door.infos.indefinite = "a"
    M.add_fact("door", door)
    M.add_fact("closed", door)

    toilet = M.new(type='c', name='toilet')
    toilet.infos.indefinite = "a"
    M.add_fact("container", toilet)
    M.add_fact("closed", toilet)
    bathroom.add(toilet)

    bedroom = M.new_room("bedroom")
    livingroom = M.new_room("livingroom")
    hallway = M.new_room("hallway")

    M.connect(kitchen.east, bedroom.west)
    M.connect(kitchen.south, livingroom.north)
    M.connect(kitchen.west, hallway.east)

    game = M.build()
    return game

from textworld.generator.data import KB

def test_loading_twl2():
    path = "/home/macote/src/TextWorld/textworld/generator/data/logic/look.twl2"
    with open(path) as f:
        document = f.read()

    actions, grammar = _parse_and_convert(document, rule_name="start2")

    print("Missing:")
    for name, rule in KB.logic.rules.items():
        if name not in actions:
            print("  ", name)
            rule.feedback_rule = "notImplemented"
        else:
            rule.feedback_rule = actions[name].feedback_rule

    game = build_game(KB)
    game.grammar = grammar

    env = TextWorldEnv()
    env.load(game=game)
    state = env.reset()

    while True:
        print(state.feedback)
        cmd = input("> ")
        state, _, _ = env.step(cmd)

    from ipdb import set_trace; set_trace()


if __name__ == "__main__":
    test_loading_twl2()
