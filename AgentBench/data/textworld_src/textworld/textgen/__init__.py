import json
from copy import deepcopy
from collections import defaultdict

from tatsu.model import NodeWalker
from typing import Iterable, Optional, Tuple, List, Dict

from textworld.utils import check_flag
from textworld.textgen.model import TextGrammarModelBuilderSemantics
from textworld.textgen.parser import TextGrammarParser

class Symbol:
    def __init__(self, symbol: str, context: Dict = {}):
        self.symbol = symbol
        self.context = context

    def __str__(self):
        return str(self.symbol)

    def copy(self, context):
        copy = deepcopy(self)
        copy.context = copy_context(context)
        return copy


class TerminalSymbol(Symbol):

    def __repr__(self):
        return "TerminalSymbol('{}')".format(self.symbol)


class NonterminalSymbol(Symbol):

    def __repr__(self):
        return "NonterminalSymbol('#{}#')".format(self.symbol)


from itertools import tee, chain

def join(sep, iterable):
    iterable = iter(iterable)
    yield next(iterable)

    for e in iterable:
        yield sep
        yield e


def copy_context(context):
    return {
        "state": context["state"],
        "facts": context["facts"],
        "variables": deepcopy(context["variables"]),
        "mapping": deepcopy(context["mapping"]),
        "entity_infos": context["entity_infos"],
    }


def display_list(l, context):
    if len(l) == 0:
        return [NonterminalSymbol("list_empty", context)]

    if len(l) == 1:
        return [l[0]]

    list_separator = NonterminalSymbol("list_separator", context)
    list_last_separator = NonterminalSymbol("list_last_separator", context)
    # "#list_separator#".join(l[:-1]) + "#list_last_separator#" + l[-1]
    return list(join(list_separator, l[:-1])) + [list_last_separator] + [l[-1]]


def query(expression, context):
    from textworld.logic import Predicate, Rule
    rule = Rule(
        name="query",
        preconditions=[Predicate.parse(e.strip()) for e in expression.split("&")],
        postconditions=[],
    )

    contexts = []
    for mapping in context["state"].all_assignments(rule, context["mapping"]):
        context_ = copy_context(context)
        new_variables = {ph.name: context_["entity_infos"][var.name] for ph, var in mapping.items()}
        context_["variables"].update(new_variables)
        context_["mapping"].update(mapping)
        contexts.append(context_)

    return contexts



def evaluate(expression, context):
    from textworld.logic import Predicate, Rule, _parse_and_convert, dnf

    expression = _parse_and_convert(expression, rule_name="onlyExpression")

    for conjunction in dnf(expression):
        rule = Rule(
            name="query",
            preconditions=list(conjunction),
            postconditions=[],
        )

        mappings = list(context["state"].all_assignments(rule, context["mapping"]))
        if len(mappings) > 0:
            return True

        # if len(mappings) > 1:
        #     assert False

        # if len(mappings) == 1:
        #     mapping = mappings[0]
        #     context_ =  copy_context(context)
        #     new_variables = {ph.name: context_["entity_infos"][var.name] for ph, var in mapping.items()}
        #     context_["variables"].update(new_variables)
        #     context_["mapping"].update(mapping)
        #     return context_

    return False


class EvalSymbol(Symbol):
    def __init__(self, expression: str, context: Dict = {}):
        super().__init__(expression, context)
        self.expression = expression

    def __repr__(self):
        return "EvalSymbol('{{{}}}')".format(str(self.expression))

    def derive(self, context=None):
        context = context or self.context
        locals().update(context["variables"])
        res = eval(self.expression)
        if isinstance(res, list):
            assert False
            return res

        return [TerminalSymbol(res)]


class ListSymbol(Symbol):
    def __init__(self, symbol: Symbol, context: Dict = {}):
        super().__init__(symbol, context)

    def __repr__(self):
        return "ListSymbol('[{!r}]')".format(self.symbol)

    def derive(self, context=None):
        context = context or self.context
        self.symbol.context = context
        derivation = self.symbol.derive()
        return display_list(derivation, context)


# class PythonSymbol(Symbol):
#     def __init__(self, expression: str, context: Dict = {}):
#         super().__init__(expression, context)
#         self.expression = expression

#     def derive(self, context=None):
#         context = context or self.context
#         res = eval(self.expression)
#         if isinstance(res, list):
#             return res

#         return [res]


class SpecialSymbol(Symbol):

    def __init__(self, expression: str, given: str, context: Dict = {}):
        text = expression
        if given:
            text += "|" + given

        super().__init__(text, context)
        self.expression = expression
        self.given = given

    def derive(self):
        assert False
        from textworld.logic import Predicate, Rule

        if len(self.context) == 0:
            raise ValueError("Empty context")

        context = copy_context(self.context)

        given = []
        if self.given:
            given += [Predicate.parse(self.given)]  # TODO: support conjonction

        is_nonterminal = False
        if self.expression.startswith("#"):
            is_nonterminal = True

        query = []
        try:
            query = [Predicate.parse(self.expression.strip("#"))]
        except:
            pass
            #query = [Predicate.parse("query({})".format(query.split(".", 1)[0]))]

        rules = [
            Rule(
                name="query",
                preconditions=given + query,
                postconditions=query
            )
        ]

        mappings = [assignment for rule in rules for assignment in context["state"].all_assignments(rule, context["mapping"])]
        mapping = mappings[0]  #TODO: support more than one possible mappings?"

        new_variables = {ph.name: context["entity_infos"][var.name] for ph, var in mapping.items()}
        context["variables"].update(new_variables)
        context["mapping"].update(mapping)

        if is_nonterminal:
            return [NonterminalSymbol(self.expression.strip("#"), context)]

        locals().update(context["variables"])
        text = str(eval(self.expression))
        return [TerminalSymbol(text)]


class ConditionalSymbol(Symbol):

    def __init__(self, expression: Symbol, given: str, context: Dict = {}):
        super().__init__(str(expression), context)
        self.expression = expression
        self.given = given

    def __repr__(self):
        return "ConditionalSymbol('{{{}|{}}}')".format(str(self.expression), str(self.given))

    def derive(self):
        from textworld.logic import Predicate, Rule

        if len(self.context) == 0:
            raise ValueError("Empty context")

        context = copy_context(self.context)

        contexts = [context]
        if self.given:
            contexts = query(self.given, context)

        # res = display_list([self.expression.copy(context) for context in contexts], context)
        res = [self.expression.copy(context) for context in contexts]
        return res


class ProductionRule:
    """ Production rule for a context-sensitive grammar. """

    # TODO: support multiple symbols for the rhs?
    def __init__(self, lhs: str, rhs: List[Symbol], weight=1, condition=None):
        """
        Arguments:
            rhs: symbol that will be transformed by this production rule.
            lhs: list of symbols generated by this production rule.
            weight: prevalence of this production.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight
        self.condition = condition

    def __repr__(self):
        return "ProductionRule(lhs={!r}, rhs={!r}, weight={!r}, condition={!r})".format(self.lhs, self.rhs, self.weight, self.condition)


class _Converter(NodeWalker):

    def __init__(self, grammar = None):
        self.grammar = grammar

    def walk_list(self, node):
        return [self.walk(child) for child in node]

    def walk_str(self, node):
        return node.replace("\\n", "\n")

    def walk_TerminalSymbol(self, node):
        return TerminalSymbol(node.literal)

    def walk_NonterminalSymbol(self, node):
        return NonterminalSymbol(node.symbol)

    def walk_ConditionalSymbol(self, node):
        return ConditionalSymbol(self.walk(node.expression), node.given)

    def walk_SpecialSymbol(self, node):
        return self.walk(node.statement)

    # def walk_PythonSymbol(self, node):
    #     return PythonSymbol(node.statement)

    def walk_EvalSymbol(self, node):
        return EvalSymbol(node.statement)

    def walk_ListSymbol(self, node):
        return ListSymbol(self.walk(node.symbol.statement))

    def walk_String(self, node):
        return self.walk(node.symbols)

    def walk_ProductionRule(self, node):
        for string in node.alternatives:
            if string is None:
                assert False
                continue

            rule = ProductionRule(node.symbol, self.walk(string))
            self.grammar.add_rule(rule)

    def walk_TextGrammar(self, node):
        self.walk(node.rules)


_PARSER = TextGrammarParser(semantics=TextGrammarModelBuilderSemantics(), parseinfo=True)


def _parse_and_convert(*args, **kwargs):
    model = _PARSER.parse(*args, **kwargs)
    return _Converter().walk(model)


class CSGUnknownSymbolError(Exception):
    def __init__(self, symbol: Symbol):
        msg = "Can't find symbol '#{}#' in the set of production rules."
        # TODO: mention closest match?
        super().__init__(msg.format(symbol))


# class ContextSensitiveGrammar:
class TextGrammar:

    def __init__(self):
        self._rules = defaultdict(list)

    def update(self, grammar: "TextGrammar"):
        for k, v in grammar._rules.items():
            self._rules[k].extend(v)

    @classmethod
    def parse(cls, text: str, filename: Optional[str] = None):
        data = json.loads(text)

        grammar = cls()
        for name, rules in data.items():
            rules = [ProductionRule(lhs=name,
                                    rhs=_parse_and_convert(rule["rhs"], rule_name="String"),
                                    weight=rule.get("weight", 1),
                                    condition=rule.get("condition", ""))
                     for rule in rules]

            for rule in rules:
                # print(repr(rule))
                grammar.add_rule(rule)

        # model = _PARSER.parse(text, filename=filename)
        # _Converter(grammar).walk(model)
        return grammar

    def add_rule(self, rule: ProductionRule):
        self._rules[rule.lhs].append(rule)

    def replace(self, start: Symbol) -> List[Symbol]:
        rules = self._rules.get(str(start))
        if not rules:
            raise CSGUnknownSymbolError(start)

        def _applicable(rule):
            if not rule.condition:
                return True

            return evaluate(rule.condition, start.context)

        rules = list(filter(_applicable, rules))

        # TODO: deal with multiple alternatives
        # TODO: deal with context
        symbols = deepcopy(rules[0].rhs)
        for symbol in symbols:
            symbol.context = copy_context(start.context)

        return symbols

    def derive(self, start: str, context={}) -> str:
        derivation = _parse_and_convert(start, rule_name="String", trace=check_flag("TW_CSG_TRACE"))
        derivation = derivation[::-1]  # Reverse to build a derivation stack.

        for symbol in derivation:
            symbol.context = copy_context(context)

        derived = []
        while len(derivation) > 0:
            if check_flag("TW_CSG_DEBUG"):
                print(derivation)

            symbol = derivation.pop()
            if isinstance(symbol, TerminalSymbol):
                derived.append(symbol)

            elif isinstance(symbol, NonterminalSymbol):
                derivation += self.replace(symbol)[::-1]  # Reverse to add on top of the derivation stack.

            elif isinstance(symbol, (ConditionalSymbol, EvalSymbol, ListSymbol)):
                derivation += symbol.derive()[::-1]  # Reverse to add on top of the derivation stack.

            else:
                raise NotImplementedError("Unknown symbol: {}".format(type(symbol)))

        return "".join(map(str, derived))
