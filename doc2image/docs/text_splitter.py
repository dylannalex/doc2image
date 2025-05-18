import re
from typing import Optional, Union, Literal, Iterable


def _split_text_with_regex(
    text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> list[str]:
    """Split text using regex and return the resulting chunks.
    
    Args:
        text (str): The input text to be split.
        separator (str): The regex pattern used for splitting the text.
        keep_separator (Union[bool, Literal["start", "end"]]): Whether to keep the
            separator and where to place it in each corresponding chunk (True='start')

    Returns:
        list[str]: A list of text chunks obtained after splitting.
    """
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class TextSplitter:
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
        is_separator_regex: bool,
        keep_separator: Union[bool, Literal["start", "end"]],
        strip_whitespace: bool,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size (int): Maximum size of chunks to return
            chunk_overlap (int): Overlap in characters between chunks
            separators (list[str]): List of separators to use for splitting
            is_separator_regex (bool): Whether the separators are regex patterns
            keep_separator (Union[bool, Literal["start", "end"]]): Whether to keep the
                separator and where to place it in each corresponding chunk (True='start')
            strip_whitespace (bool): If `True`, strips whitespace from the start and end of
                every document
        """
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._strip_whitespace = strip_whitespace

    def _join_docs(self, docs: list[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        """Merge smaller splits into larger chunks."""
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = len(separator)

        docs = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            _len = len(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print(
                        "WARNING: "
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= len(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if len(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            list[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)
