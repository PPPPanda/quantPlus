"""GUI input with foreground verification and SendInput."""
import ctypes
import ctypes.wintypes
import time
import struct

user32 = ctypes.windll.user32

# Check foreground window before and after click
def get_fg_title():
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return hwnd, buf.value

# Check what window is at a point
def window_at(x, y):
    point = ctypes.wintypes.POINT(x, y)
    hwnd = user32.WindowFromPoint(point)
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return hwnd, buf.value

print(f"Before click - Foreground: {get_fg_title()}")
print(f"Window at (300, 78): {window_at(300, 78)}")

# Force CTA to foreground first
hwnd_at = user32.WindowFromPoint(ctypes.wintypes.POINT(300, 78))
user32.keybd_event(0x12, 0, 0, 0)  # Alt down
user32.ShowWindow(hwnd_at, 3)  # SW_SHOWMAXIMIZED
user32.SetForegroundWindow(hwnd_at)
user32.keybd_event(0x12, 0, 2, 0)  # Alt up
time.sleep(0.5)

print(f"After SetForeground - Foreground: {get_fg_title()}")

# Click on input field
user32.SetCursorPos(300, 78)
time.sleep(0.1)
user32.mouse_event(0x0002, 0, 0, 0, 0)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.5)

print(f"After click - Foreground: {get_fg_title()}")

# Press Shift to switch IME
user32.keybd_event(0x10, 0, 0, 0)
time.sleep(0.02)
user32.keybd_event(0x10, 0, 2, 0)
time.sleep(0.3)

# Ctrl+A
user32.keybd_event(0x11, 0, 0, 0)
user32.keybd_event(0x41, 0, 0, 0)
time.sleep(0.02)
user32.keybd_event(0x41, 0, 2, 0)
user32.keybd_event(0x11, 0, 2, 0)
time.sleep(0.3)

# Delete
user32.keybd_event(0x2E, 0, 0, 0)
time.sleep(0.02)
user32.keybd_event(0x2E, 0, 2, 0)
time.sleep(0.3)

# Type each character
for ch in 'p2505.DCE':
    vk = user32.VkKeyScanW(ord(ch))
    vk_code = vk & 0xFF
    need_shift = (vk >> 8) & 1
    if need_shift:
        user32.keybd_event(0x10, 0, 0, 0)
    user32.keybd_event(vk_code, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(vk_code, 0, 2, 0)
    if need_shift:
        user32.keybd_event(0x10, 0, 2, 0)
    time.sleep(0.05)

print(f"After typing - Foreground: {get_fg_title()}")
print("Done")
