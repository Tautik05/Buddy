"""
Input Handler Module
Cross-platform keyboard input handling
"""

import sys

# Platform-specific imports
if sys.platform == 'win32':
    import msvcrt
    USE_MSVCRT = True
    select = None
else:
    import select
    USE_MSVCRT = False


class InputHandler:
    """Handles cross-platform keyboard input"""
    
    def __init__(self):
        self.input_buffer = ""
        self.use_msvcrt = USE_MSVCRT
    
    def check_input(self) -> tuple[bool, str]:
        """
        Check for keyboard input non-blocking
        Returns: (enter_pressed, current_buffer)
        """
        if self.use_msvcrt:
            return self._check_windows()
        else:
            return self._check_unix()
    
    def _check_windows(self) -> tuple[bool, str]:
        """Windows input handling"""
        try:
            if msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8', errors='ignore')
                if char == '\r':
                    result = self.input_buffer
                    self.input_buffer = ""
                    return True, result
                elif char == '\b':
                    if self.input_buffer:
                        self.input_buffer = self.input_buffer[:-1]
                        print('\b \b', end="", flush=True)
                elif char.isprintable():
                    self.input_buffer += char
                    print(char, end="", flush=True)
        except:
            pass
        return False, self.input_buffer
    
    def _check_unix(self) -> tuple[bool, str]:
        """Unix/Linux input handling"""
        if select and select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char == '\n':
                result = self.input_buffer
                self.input_buffer = ""
                return True, result
            elif char == '\x7f':  # backspace
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
                    print('\b \b', end="", flush=True)
            elif char.isprintable():
                self.input_buffer += char
                print(char, end="", flush=True)
        return False, self.input_buffer
    
    def clear(self):
        """Clear input buffer"""
        self.input_buffer = ""