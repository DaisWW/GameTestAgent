_ANDROID_SHELL_META = frozenset(r'&;|<>()$`!*?[]{}\^~#')


def escape_for_adb_input(text: str) -> tuple[str, bool]:
    """将字符串转义为 `adb shell input text` 可安全接受的格式。

    Returns:
        (escaped_text, has_non_ascii)
    """
    result = []
    has_non_ascii = False
    for ch in text:
        if ch == '%':
            result.append('%%')      # 必须先于空格处理，防止 %s → 空格
        elif ch == ' ':
            result.append('%s')
        elif ch in _ANDROID_SHELL_META or ch in ("'", '"'):
            result.append(f'\\{ch}')
        else:
            if ord(ch) > 127:
                has_non_ascii = True
            result.append(ch)
    return ''.join(result), has_non_ascii
