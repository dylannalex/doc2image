from dataclasses import dataclass


@dataclass
class Prompt:
    """
    A dataclass representing a prompt for a language model.

    Attributes:
        parameters (list[str]): A list of parameter names to be used in the prompt.
        messages (list[dict[str, str]]): A list of messages, each containing a role and content.
    """

    parameters: list[str]
    messages: list[dict[str, str]]

    def format(self, values: dict[str, str]) -> list[dict[str, str]]:
        """
        Format the prompt messages with the given values.

        Args:
            values (dict[str, str]): A dictionary of values to format the prompt messages.

        Returns:
            list[dict[str, str]]: The formatted prompt messages.

        Raises:
            ValueError: If any required parameter is missing in the values dictionary.
        """
        for param in self.parameters:
            if param not in values:
                raise ValueError(f"Missing required parameter: {param}")

        formatted_messages = []
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            formatted_content = content.format(**values)
            formatted_messages.append({"role": role, "content": formatted_content})

        return formatted_messages
