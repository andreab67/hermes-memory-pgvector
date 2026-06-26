# Forked from andreab67/hermes-hexus (BSD-3-Clause)
"""Conformance tests to ensure HexusMemoryProvider matches the MemoryProvider plugin interface contract."""

from __future__ import annotations

import inspect
from unittest.mock import patch


def test_hexus_memory_provider_conforms():
    with patch("hexus.MemoryProvider", new=object):
        from hexus import HexusMemoryProvider

        # Verify class exists
        assert inspect.isclass(HexusMemoryProvider)

        # Verify name property
        assert isinstance(HexusMemoryProvider.name, property)

        provider = HexusMemoryProvider(config={})
        assert provider.name == "hexus"

        # Helper to get signature
        def check_signature(method_name, expected_params, check_var_kwargs=False):
            assert hasattr(HexusMemoryProvider, method_name), (
                f"Missing method {method_name}"
            )
            method = getattr(HexusMemoryProvider, method_name)
            assert inspect.isfunction(method), f"{method_name} is not a function/method"
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            # First param must be self
            assert params[0] == "self"

            # Check other params
            for p in expected_params:
                assert p in params, f"Method {method_name} is missing parameter '{p}'"

            if check_var_kwargs:
                # Check for **kwargs
                has_var_keyword = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in sig.parameters.values()
                )
                assert has_var_keyword, f"Method {method_name} must accept **kwargs"

        # Verify lifecycle methods
        check_signature("is_available", [])
        check_signature("initialize", ["session_id"], check_var_kwargs=True)
        check_signature("shutdown", [])
        check_signature("on_session_switch", ["new_session_id"], check_var_kwargs=True)
        check_signature(
            "on_turn_start", ["turn_number", "message"], check_var_kwargs=True
        )
        check_signature("on_session_end", ["messages"])

        # Verify ambient recall & prompt methods
        check_signature("system_prompt_block", [])
        check_signature("prefetch", ["query"])

        # Verify hook and tool methods
        check_signature("on_memory_write", ["action", "target", "content", "metadata"])
        check_signature(
            "handle_tool_call", ["tool_name", "args"], check_var_kwargs=True
        )
