"""Empty H2-H5 interfaces. Public conversation only; new instance per episode."""
class Harness:
    def h3(self, tools):
        """Once before interaction: edit only tool/function descriptions."""
        return tools

    def h5(self, messages):
        """Once after initial public instruction: return at most one skill hint."""
        return []

    def h4(self, messages, remaining):
        """After execution/before next turn: at most one feedback/budget hint."""
        return []

    def h2(self, message, history):
        """Before execution: deterministic repair/validation of model action."""
        return message
