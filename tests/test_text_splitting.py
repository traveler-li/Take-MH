"""流式文本切分测试：句子边界、残句刷新与重置。"""

from __future__ import annotations

from takemh.text_splitting import SentenceSplitter


def test_push_returns_complete_sentences_and_buffers_remainder():
    splitter = SentenceSplitter()

    assert splitter.push("你好，") == []
    assert splitter.push("世界。这是下一句") == ["你好，世界。"]
    assert splitter.flush() == "这是下一句"


def test_ellipsis_and_newline_are_sentence_boundaries():
    splitter = SentenceSplitter()

    assert splitter.push("第一句…") == ["第一句…"]
    assert splitter.push("第二句！\n") == ["第二句！"]


def test_reset_clears_buffered_text():
    splitter = SentenceSplitter()
    splitter.push("还没说完")
    splitter.reset()

    assert splitter.flush() == ""
