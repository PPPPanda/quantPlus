"""Use PostMessage WM_CHAR to type directly into Qt widget, bypassing IME."""
import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_HOME = 0x24
VK_DELETE = 0x2E
VK_CONTROL = 0x11

hwnd = user32.GetForegroundWindow()
print(f"CTA hwnd: {hwnd}")

# Click on the input field
user32.SetCursorPos(300, 78)
time.sleep(0.1)
user32.mouse_event(0x0002, 0, 0, 0, 0)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.5)

# Select all: send Ctrl+A via PostMessage
user32.PostMessageW(hwnd, WM_KEYDOWN, VK_CONTROL, 0)
time.sleep(0.02)
user32.PostMessageW(hwnd, WM_KEYDOWN, 0x41, 0)  # A
time.sleep(0.02)
user32.PostMessageW(hwnd, WM_KEYUP, 0x41, 0)
user32.PostMessageW(hwnd, WM_KEYUP, VK_CONTROL, 0)
time.sleep(0.3)

# Send each character via WM_CHAR (bypasses IME completely)
text = 'p2505.DCE'
for ch in text:
    user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)
    time.sleep(0.03)

time.sleep(0.3)
print(f"Sent: {text}")
print("Done")
