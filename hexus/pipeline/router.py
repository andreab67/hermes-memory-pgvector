# Forked from andreab67/hermes-hexus (BSD-3-Clause)
import re
import json

class ContentRouter:
    """Pre-processing pipeline to route and compress large memory payloads (> 200 tokens)."""

    def __init__(self, threshold_tokens: int = 200):
        # 1 token ≈ 4 characters on average
        self.threshold_chars = threshold_tokens * 4

    def maybe_compress(self, text: str) -> str | None:
        """Compress text if it exceeds the token threshold.
        
        Returns the compressed version, or None if the text is under the threshold.
        """
        if not text or len(text) <= self.threshold_chars:
            return None

        # Detect the content type and compress accordingly
        if self._is_json(text):
            return self._compress_json(text)
        elif self._is_log(text):
            return self._compress_log(text)
        elif self._is_code(text):
            return self._compress_code(text)
        else:
            return self._compress_text(text)

    def _is_json(self, text: str) -> bool:
        text_strip = text.strip()
        return (text_strip.startswith("{") and text_strip.endswith("}")) or \
               (text_strip.startswith("[") and text_strip.endswith("]"))

    def _is_log(self, text: str) -> bool:
        log_indicators = [
            r'\b(?:INFO|ERROR|WARN|WARNING|DEBUG|FATAL|CRITICAL)\b',
            r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',
            r'\[\d{2}:\d{2}:\d{2}\]',
        ]
        matches = sum(1 for ind in log_indicators if re.search(ind, text, re.IGNORECASE))
        return matches >= 1 and len(text.splitlines()) > 5

    def _is_code(self, text: str) -> bool:
        code_indicators = [
            r'\b(def|class|import|from|function|const|let|var|public|private|return|async|await)\b',
            r'[{}]',
            r'\b(if|for|while)\s*\(.*\)\s*\{',
        ]
        matches = sum(1 for ind in code_indicators if re.search(ind, text))
        return matches >= 2

    def _compress_json(self, text: str) -> str:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                keys = list(data.keys())
                truncated = {k: data[k] for k in keys[:3]}
                return f"[Compressed JSON Object] keys: {', '.join(keys)}\nSample: {json.dumps(truncated)}"
            elif isinstance(data, list):
                return f"[Compressed JSON Array] length: {len(data)}\nFirst item: {json.dumps(data[0]) if data else 'empty'}"
        except Exception:
            pass
        return text[:self.threshold_chars] + "\n... [Truncated JSON]"

    def _compress_log(self, text: str) -> str:
        lines = text.splitlines()
        first_n = lines[:3]
        last_n = lines[-3:]

        error_lines = []
        for line in lines[3:-3]:
            if re.search(r'\b(?:ERROR|FATAL|CRITICAL|WARNING|WARN|EXCEPTION|FAIL)\b', line, re.IGNORECASE):
                error_lines.append(line)

        filtered = []
        filtered.extend(first_n)
        if error_lines:
            filtered.append(f"... [Truncated log: showing {len(error_lines)} errors/warnings] ...")
            filtered.extend(error_lines[:20])  # Cap at 20 critical lines
        else:
            filtered.append("... [Truncated log: no critical patterns found] ...")
        filtered.extend(last_n)

        return "\n".join(filtered)

    def _compress_code(self, text: str) -> str:
        lines = text.splitlines()
        compressed_lines = []
        for line in lines:
            if re.match(r'^\s*(def|class|import|from|async\s+def)\b', line):
                compressed_lines.append(line)
            elif re.match(r'^\s*#.*', line) and len(compressed_lines) < 10:
                compressed_lines.append(line)

        if len(compressed_lines) < 3:
            return text[:self.threshold_chars] + "\n... [Truncated Code]"

        return "\n".join(compressed_lines) + "\n... [Truncated Code: definitions only]"

    def _compress_text(self, text: str) -> str:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            summary = paragraphs[0]
            if len(paragraphs) > 1:
                summary += "\n\n" + paragraphs[1][:200] + "..."
            return summary + "\n... [Truncated Text]"
        return text[:self.threshold_chars] + "\n... [Truncated Text]"
