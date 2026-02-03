"""Force IME to English, then type into CTA field."""
import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32
imm32 = ctypes.windll.imm32

# Get foreground window
hwnd = user32.GetForegroundWindow()

# Get IMM context
himc = imm32.ImmGetContext(hwnd)
print(f"IMM context: {himc}")

if himc:
    # Get conversion status
    conv = ctypes.wintypes.DWORD()
    sent = ctypes.wintypes.DWORD()
    imm32.ImmGetConversionStatus(himc, ctypes.byref(conv), ctypes.byref(sent))
    print(f"Conversion mode: {conv.value} (0=alphanumeric, 1=native/chinese)")
    
    # Force to alphanumeric (English) mode
    if conv.value & 1:  # If in native/Chinese mode
        print("Switching to English mode...")
        imm32.ImmSetConversionStatus(himc, 0, sent.value)
        time.sleep(0.3)
        imm32.ImmGetConversionStatus(himc, ctypes.byref(conv), ctypes.byref(sent))
        print(f"After switch - Conversion mode: {conv.value}")
    else:
        print("Already in English mode")
    
    imm32.ImmReleaseContext(hwnd, himc)
else:
    print("No IMM context - trying keyboard shortcut")
    # Some IME use Win+Space or Ctrl+Space
    pass

# Now click and type
user32.SetCursorPos(300, 78)
time.sleep(0.1)
user32.mouse_event(0x0002, 0, 0, 0, 0)
user32.mouse_event(0x0004, 0, 0, 0, 0)
time.sleep(0.5)

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

time.sleep(0.3)
print("Done typing")
