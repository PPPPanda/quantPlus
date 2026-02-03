import ctypes, time

user32 = ctypes.windll.user32

def click(x, y):
    user32.SetCursorPos(x, y)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)

def press(vk):
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    user32.keybd_event(vk, 0, 2, 0)

def type_text(text):
    for ch in text:
        vk_result = user32.VkKeyScanW(ord(ch))
        vk = vk_result & 0xFF
        shift = (vk_result >> 8) & 1
        if shift:
            user32.keybd_event(0x10, 0, 0, 0)
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(vk, 0, 2, 0)
        if shift:
            user32.keybd_event(0x10, 0, 2, 0)
        time.sleep(0.03)

# Click on input field (right side where text is)
click(300, 78)
time.sleep(0.8)

# Toggle IME to English
press(0x10)  # Shift
time.sleep(0.3)

# Ctrl+A to select all
user32.keybd_event(0x11, 0, 0, 0)  # CTRL down
time.sleep(0.02)
user32.keybd_event(0x41, 0, 0, 0)  # A
time.sleep(0.02)
user32.keybd_event(0x41, 0, 2, 0)
user32.keybd_event(0x11, 0, 2, 0)
time.sleep(0.3)

# Delete
press(0x2E)  # VK_DELETE
time.sleep(0.3)

# Type new value
type_text('p2505.DCE')
time.sleep(0.3)

print('Done')
